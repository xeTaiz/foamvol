"""
MORE (Multi-Organ REconstruction) dataset loader.

Two on-disk layouts (detected automatically):

1. **Test PNG layout** (extracted from test.zip):
   <data_dir>/test/{patient_id}_{slice_idx:04d}.png
   512×512 uint8 grayscale images, normalised to [0, 1].

2. **DICOM layout** (extracted from CT/{Organ}.zip):
   <data_dir>/{organ}/{patient_id}/full_1mm/*.dcm
   Raw DICOM CT slices (HU values).  1 mm reconstruction preferred.

Since MORE ships reconstructed CT images (no native sinograms), fan-beam
projections are simulated via ASTRA using a standard 3rd-generation CT
geometry (same parameters as the AAPM Mayo challenge):
  SOD = 595 mm, SDD = 1085.6 mm, 736 det @ 1.2858 mm pitch.

Output shapes (matching DataHandler convention):
  all_rays:         [N_angles, N_slices, N_det, 6]
  all_projections:  [N_angles, N_slices, N_det, 1]
"""

import math
import os
import glob
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode

# --- Standard AAPM fan-flat CT geometry ---
SOD_MM = 595.0
SDD_MM = 1085.6
N_DET = 736
DET_PIX_MM = 1.2858
N_ANGLES = 1000
ASSUMED_FOV_MM = 500.0
IMG_SIZE = 512
SCENE_SCALE_XY = ASSUMED_FOV_MM / 2.0
PIXEL_MM = 2.0 * SCENE_SCALE_XY / IMG_SIZE

SOD_NORM = SOD_MM / SCENE_SCALE_XY
SDD_NORM = SDD_MM / SCENE_SCALE_XY
DET_PIX_NORM = DET_PIX_MM / SCENE_SCALE_XY


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def _detect_layout(data_dir, split):
    """Return ('png', png_dir) or ('dicom', [list_of_patient_dirs], slice_thickness_mm)."""
    # PNG layout: data_dir/{split}/{patient}_{slice:04d}.png
    for candidate_split in (split, "train", "test"):
        png_dir = os.path.join(data_dir, candidate_split)
        if os.path.isdir(png_dir) and glob.glob(os.path.join(png_dir, "*.png")):
            return "png", png_dir, 1.0  # PNG assumed 1 mm spacing

    # DICOM layout: data_dir/{organ}/{patient_id}/full_1mm/*.dcm  (prefer 1 mm)
    patient_entries = []
    for organ_dir in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not organ_dir.is_dir():
            continue
        for patient_dir in sorted(os.scandir(organ_dir.path), key=lambda e: e.name):
            if not patient_dir.is_dir():
                continue
            # Prefer 1 mm; fall back to 3 mm
            series_1mm = os.path.join(patient_dir.path, "full_1mm")
            series_3mm = os.path.join(patient_dir.path, "full_3mm")
            if os.path.isdir(series_1mm) and glob.glob(os.path.join(series_1mm, "*.dcm")):
                patient_entries.append((patient_dir.path, series_1mm, 1.0))
            elif os.path.isdir(series_3mm) and glob.glob(os.path.join(series_3mm, "*.dcm")):
                patient_entries.append((patient_dir.path, series_3mm, 3.0))

    if patient_entries:
        return "dicom", patient_entries, None  # thickness per entry

    raise FileNotFoundError(
        f"Cannot detect MORE dataset layout in {data_dir}.\n"
        "Expected either:\n"
        "  {data_dir}/test/{{patient_id}}_{{slice:04d}}.png  (test.zip extracted), or\n"
        "  {data_dir}/{{organ}}/{{patient_id}}/full_1mm/*.dcm  (CT organ zip extracted)."
    )


# ---------------------------------------------------------------------------
# Slice loaders
# ---------------------------------------------------------------------------

def _load_png_patient(png_dir, patient_id):
    """Return (N, H, W) float32 in [0, 1] for a given patient."""
    files = sorted(glob.glob(os.path.join(png_dir, f"{patient_id}_*.png")))
    if not files:
        raise FileNotFoundError(f"No PNG slices for patient {patient_id} in {png_dir}")
    slices = []
    for f in files:
        img_t = read_image(f, mode=ImageReadMode.GRAY)  # (1, H, W) uint8
        slices.append(img_t.squeeze().numpy().astype(np.float32) / 255.0)
    return np.stack(slices, axis=0)


def _load_dicom_series(series_dir):
    """Return (N, H, W) float32 with density = (HU + 1000) / 1000, sorted by z."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for MORE DICOM data:\n"
            "  micromamba install -n radfoam pydicom"
        ) from exc

    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {series_dir}")

    slices_with_z = []
    for f in files:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        density = np.clip((hu + 1000.0) / 1000.0, 0.0, None)
        try:
            z = float(ds.ImagePositionPatient[2])
        except Exception:
            z = float(ds.InstanceNumber)
        slices_with_z.append((z, density))

    slices_with_z.sort(key=lambda x: x[0])
    return np.stack([s[1] for s in slices_with_z], axis=0)   # (N, H, W)


def _resize_volume(vol, target_size):
    if vol.shape[1] == target_size and vol.shape[2] == target_size:
        return vol
    t = torch.from_numpy(vol).float().unsqueeze(1)  # (N, 1, H, W)
    t = F.interpolate(t, size=(target_size, target_size), mode="area")
    return t.squeeze(1).numpy()


# ---------------------------------------------------------------------------
# ASTRA forward projection + ray building
# ---------------------------------------------------------------------------

def _astra_fanflat_sino(volume_slice, vol_geom, proj_geom):
    """2D fan-flat ASTRA forward projection. Returns (N_angles, N_det)."""
    import astra
    vid = astra.data2d.create("-vol", vol_geom, volume_slice)
    sid = astra.data2d.create("-sino", proj_geom)
    cfg = astra.astra_dict("FP_CUDA")
    cfg["VolumeDataId"] = vid
    cfg["ProjectionDataId"] = sid
    alg = astra.algorithm.create(cfg)
    astra.algorithm.run(alg)
    sino = astra.data2d.get(sid).astype(np.float32)
    astra.algorithm.delete(alg)
    astra.data2d.delete(vid)
    astra.data2d.delete(sid)
    return sino


def _build_rays(angles, det_norm, z_norm):
    """Build (N_a, N_z, N_d, 6) fan-beam ray array. Rays are horizontal (dz=0)."""
    N_a, N_z, N_d = len(angles), len(z_norm), len(det_norm)

    cos_a = np.cos(angles)[:, None, None]
    sin_a = np.sin(angles)[:, None, None]
    u = det_norm[None, None, :]
    z = z_norm[None, :, None]

    ox = np.broadcast_to(SOD_NORM * cos_a, (N_a, N_z, N_d)).copy()
    oy = np.broadcast_to(SOD_NORM * sin_a, (N_a, N_z, N_d)).copy()
    oz = np.broadcast_to(z,               (N_a, N_z, N_d)).copy()

    rdx = -SDD_NORM * cos_a + u * (-sin_a)
    rdy = -SDD_NORM * sin_a + u *   cos_a
    norm = np.sqrt(rdx**2 + rdy**2 + 1e-30)
    dx = np.broadcast_to(rdx / norm, (N_a, N_z, N_d)).copy()
    dy = np.broadcast_to(rdy / norm, (N_a, N_z, N_d)).copy()
    dz = np.zeros((N_a, N_z, N_d), dtype=np.float32)

    return np.stack([ox, oy, oz, dx, dy, dz], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class MOREDataset:
    """MORE multi-organ CT dataset with simulated fan-beam projections.

    Selects one patient study (sample_index) from the data directory,
    forward-projects all slices via ASTRA, and builds a 3D stacked ray set.

    Output shapes:
      all_rays:         [N_angles, N_slices, N_det, 6]
      all_projections:  [N_angles, N_slices, N_det, 1]
    """

    beam_type = "cone"

    def __init__(self, data_dir, split="train", sample_index=0,
                 num_angles=None, detector_size=None, **kwargs):
        import astra
        layout, location, default_thickness = _detect_layout(data_dir, split)

        if layout == "png":
            patients = sorted(set(
                os.path.basename(f).split("_")[0]
                for f in glob.glob(os.path.join(location, "*.png"))
            ))
            if sample_index >= len(patients):
                raise IndexError(
                    f"sample_index={sample_index} out of range "
                    f"({len(patients)} patients in {location})"
                )
            patient_id = patients[sample_index]
            volume = _load_png_patient(location, patient_id)
            slice_thickness_mm = default_thickness
            self.patient_id = patient_id

        else:  # dicom
            if sample_index >= len(location):
                raise IndexError(
                    f"sample_index={sample_index} out of range "
                    f"({len(location)} patient series found)"
                )
            patient_dir, series_dir, slice_thickness_mm = location[sample_index]
            volume = _load_dicom_series(series_dir)
            self.patient_id = os.path.basename(patient_dir)

        volume = _resize_volume(volume, IMG_SIZE)
        N_slices = volume.shape[0]

        n_angles = num_angles if num_angles is not None else N_ANGLES
        n_det = detector_size if detector_size is not None else N_DET
        det_pix_mm = DET_PIX_MM * (N_DET / n_det) if detector_size is not None else DET_PIX_MM

        angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
        vol_geom = astra.create_vol_geom(IMG_SIZE, IMG_SIZE)
        proj_geom = astra.create_proj_geom(
            "fanflat",
            det_pix_mm / PIXEL_MM,
            n_det,
            angles,
            SOD_MM / PIXEL_MM,
            (SDD_MM - SOD_MM) / PIXEL_MM,
        )

        sinos = np.stack([
            _astra_fanflat_sino(volume[k], vol_geom, proj_geom)
            for k in range(N_slices)
        ], axis=1)   # (N_angles, N_slices, N_det)

        pixel_size_norm = 2.0 / IMG_SIZE
        sinos = sinos * pixel_size_norm   # model-space integral units

        scene_scale_z = max(SCENE_SCALE_XY, N_slices * slice_thickness_mm / 2.0)
        self.scene_scale = max(SCENE_SCALE_XY, scene_scale_z)
        z_mm = (np.arange(N_slices) - (N_slices - 1) / 2.0) * slice_thickness_mm
        z_norm = (z_mm / self.scene_scale).astype(np.float32)

        i = np.arange(n_det, dtype=np.float32)
        u_mm = (i - (n_det - 1) / 2.0) * det_pix_mm
        det_norm = u_mm / SCENE_SCALE_XY

        rays = _build_rays(angles, det_norm, z_norm)

        self.all_rays = torch.from_numpy(rays)
        self.all_projections = torch.from_numpy(
            sinos[..., np.newaxis].astype(np.float32)
        )

        ang_per_pix = math.atan(det_pix_mm / SDD_MM)
        z_ang = slice_thickness_mm / self.scene_scale / 2.0 if N_slices > 1 else 0.0
        self.pixel_ang_size = (ang_per_pix, z_ang)

        self.c2ws = None
        self.fx = None
        self.fy = None

        self.n_slices = N_slices
        self.slice_thickness_mm = slice_thickness_mm
        self.layout = layout
