"""
AAPM Low-Dose CT Grand Challenge (Mayo Clinic) dataset loader.

Two on-disk layouts are supported — detected automatically:

1. **DICOM reconstruction** (typical TCIA download):
   <data_dir>/
     <patient_id>/
       <series>/
         *.dcm   (or *.IMA)

   Slices loaded via pydicom, HU → linear attenuation, then forward-projected
   with ASTRA using the real Mayo scanner geometry.

2. **Pre-processed NPY** (community preprocessing pipeline):
   <data_dir>/train/*.npy  and  <data_dir>/test/*.npy
   Each .npy is a single 512×512 float32 slice in [0,1].

In both cases the output is a 3D stacked fan-beam dataset using the Mayo
scanner geometry (identical to the AAPM geometry used in ct_utils.py):
  SOD = 595 mm, SDD = 1085.6 mm, 736 det @ 1.2858 mm pitch.

sample_index selects which patient/study to load.

Output shapes:
  all_rays:         [N_angles, N_slices, N_det, 6]
  all_projections:  [N_angles, N_slices, N_det, 1]
"""

import math
import os
import glob
import warnings

import numpy as np
import torch
import torch.nn.functional as F

# --- Mayo clinic scanner geometry (AAPM challenge) ---
SOD_MM = 595.0
SDD_MM = 1085.6
N_DET = 736
DET_PIX_MM = 1.2858
N_ANGLES = 1000
IMG_SIZE = 512
ASSUMED_FOV_MM = 500.0          # typical reconstruction FOV
SCENE_SCALE_XY = ASSUMED_FOV_MM / 2.0
PIXEL_MM = 2.0 * SCENE_SCALE_XY / IMG_SIZE

SOD_NORM = SOD_MM / SCENE_SCALE_XY
SDD_NORM = SDD_MM / SCENE_SCALE_XY
DET_PIX_NORM = DET_PIX_MM / SCENE_SCALE_XY

# Water linear attenuation at ~70 keV (μ_water ≈ 0.192 cm⁻¹ = 0.0192 mm⁻¹)
MU_WATER_PER_MM = 0.0192


def _detect_layout(data_dir, split):
    """Return ('npy', split_dir) or ('dicom', [list of patient dirs])."""
    for candidate_split in (split, "train", "test"):
        npy_dir = os.path.join(data_dir, candidate_split)
        if os.path.isdir(npy_dir) and glob.glob(os.path.join(npy_dir, "*.npy")):
            return "npy", npy_dir

    # DICOM: look for patient subdirs containing *.dcm or *.IMA files
    patient_dirs = []
    for entry in sorted(os.scandir(data_dir)):
        if not entry.is_dir():
            continue
        dcm_files = (
            glob.glob(os.path.join(entry.path, "**", "*.dcm"), recursive=True)
            or glob.glob(os.path.join(entry.path, "**", "*.IMA"), recursive=True)
        )
        if dcm_files:
            patient_dirs.append(entry.path)
    if patient_dirs:
        return "dicom", patient_dirs

    raise FileNotFoundError(
        f"Cannot detect AAPM-Mayo layout in {data_dir}.\n"
        "Expected either:\n"
        "  {data_dir}/{split}/*.npy         (pre-processed numpy slices), or\n"
        "  {data_dir}/<patient_id>/.../*.dcm  (DICOM reconstruction files).\n"
        "See https://www.aapm.org/GrandChallenge/LowDoseCT/#registration for download."
    )


def _load_npy_study(npy_dir, sample_index):
    """Load pre-processed numpy slices. Returns (N, H, W) float32."""
    files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files in {npy_dir}")
    # sample_index selects individual slices; group into a single volume
    # (the npy layout has one file per slice, all from a mixed-patient set)
    slices = [np.load(f).astype(np.float32) for f in files]
    # Stack only a contiguous block for 3D reconstruction; use all by default
    vol = np.stack(slices, axis=0)
    if vol.ndim == 4:   # (N, 1, H, W)
        vol = vol[:, 0]
    return vol   # (N_slices, H, W)


def _load_dicom_patient(patient_dir):
    """Load all DICOM slices for a patient. Returns (N, H, W) float32 in HU/1000 + 1."""
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required to load DICOM AAPM-Mayo data:\n"
            "  micromamba install -n radfoam pydicom"
        ) from exc

    dcm_files = (
        sorted(glob.glob(os.path.join(patient_dir, "**", "*.dcm"), recursive=True))
        or sorted(glob.glob(os.path.join(patient_dir, "**", "*.IMA"), recursive=True))
    )
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files under {patient_dir}")

    slices_with_pos = []
    for path in dcm_files:
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
        density = (hu + 1000.0) / 1000.0   # 0 = air, 1 = water, >1 = bone
        density = np.clip(density, 0.0, None)
        try:
            z = float(ds.ImagePositionPatient[2])
        except Exception:
            z = float(ds.InstanceNumber)
        slices_with_pos.append((z, density))

    slices_with_pos.sort(key=lambda x: x[0])
    vol = np.stack([s[1] for s in slices_with_pos], axis=0)   # (N, H, W)

    # Resize to IMG_SIZE if needed
    if vol.shape[1] != IMG_SIZE or vol.shape[2] != IMG_SIZE:
        t = torch.from_numpy(vol).float().unsqueeze(1)  # (N, 1, H, W)
        t = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="area")
        vol = t.squeeze(1).numpy()

    return vol


def _astra_fanflat_sino(volume_slice, angles, vol_geom, proj_geom):
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
    return sino   # (N_angles, N_det)


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


class AAPMMayoDataset:
    """AAPM-Mayo Low-Dose CT dataset with simulated fan-beam projections.

    Supports pre-processed NPY slices and raw DICOM reconstruction files.
    Forward-projects via ASTRA using the real Mayo scanner geometry.

    Output shapes:
      all_rays:         [N_angles, N_slices, N_det, 6]
      all_projections:  [N_angles, N_slices, N_det, 1]
    """

    beam_type = "cone"

    def __init__(self, data_dir, split="train", sample_index=0,
                 num_angles=None, detector_size=None, **kwargs):
        import astra
        layout, location = _detect_layout(data_dir, split)

        if layout == "npy":
            volume = _load_npy_study(location, sample_index)
            slice_thickness_mm = 1.0   # pre-processed NPY assumed 1 mm
        else:
            if sample_index >= len(location):
                raise IndexError(
                    f"sample_index={sample_index} out of range "
                    f"({len(location)} patients found)"
                )
            volume = _load_dicom_patient(location[sample_index])
            # Try to read slice thickness from first DICOM (best-effort)
            try:
                import pydicom
                dcm_files = (
                    sorted(glob.glob(os.path.join(location[sample_index], "**", "*.dcm"), recursive=True))
                    or sorted(glob.glob(os.path.join(location[sample_index], "**", "*.IMA"), recursive=True))
                )
                ds = pydicom.dcmread(dcm_files[0])
                slice_thickness_mm = float(ds.SliceThickness)
            except Exception:
                slice_thickness_mm = 1.0
                warnings.warn("Could not read SliceThickness from DICOM; assuming 1 mm.")

        N_slices, H, W = volume.shape
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

        # DICOM volume is already in density units (1 = water).
        # NPY pre-processed files are in [0, 1]; treat as fractional attenuation.
        # Both are forward-projected as-is; the model learns the scale.
        sinos = np.stack([
            _astra_fanflat_sino(volume[k], angles, vol_geom, proj_geom)
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
        z_ang = slice_thickness_mm / SCENE_SCALE_XY / 2.0 if N_slices > 1 else 0.0
        self.pixel_ang_size = (ang_per_pix, z_ang)

        self.c2ws = None
        self.fx = None
        self.fy = None

        self.n_slices = N_slices
        self.slice_thickness_mm = slice_thickness_mm
        self.layout = layout
