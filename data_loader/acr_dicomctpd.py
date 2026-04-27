"""Data loader for the ACR CT accreditation phantom from the TCIA LDCT collection.

Data format: DICOM-CT-PD (Siemens SOMATOM Definition Flash, helical fan-beam)
Source collection: LDCT-and-Projection-Data, PatientID='PatientID' (ACR_Phantom)

Geometry (decoded from Siemens private tags):
  - SOD  = 595.0 mm  (source-to-isocenter, tag 7031,1003)
  - SDD  = 1085.6 mm (source-to-detector, tag 7031,1031)
  - View angle per file: tag 7031,1001 (radians)
  - Table z position per file: tag 7031,1002 (mm)
  - Detector: 736 channels × 64 rows
  - DataCollectionDiameter: 500 mm (scan FOV)

Projection data:
  Each DICOM file stores one fan-beam view (736 channels × 64 rows).
  Rescaled values via (RescaleSlope, RescaleIntercept) are line integrals in nepers.
  No further dark/flat-field correction needed.

Trajectory: helical — source traces a helix around the rotation axis.
Ray geometry can be used directly with radfoam since only (origin, direction)
pairs matter (no assumption about the trajectory shape).

Usage:
  ds = ACRPhantomDataset('/path/to/dicom_dir/', split='train', every_nth=10)

  every_nth: use every N-th view to reduce dataset size (default 10 → ~1800 views)
  split='test': uses every_nth × 5 spacing (much sparser), ~360 views

NOTE: No vol_gt is provided (no reconstructed images available in this collection).
      For evaluation, run FDK reconstruction on the projections, e.g. with TIGRE.
"""

import os
import struct
import re
import numpy as np
import torch
import pydicom


# Known Siemens private tags (decoded empirically from the LDCT collection)
_TAG_ANGLE  = (0x7031, 0x1001)  # source angle, radians
_TAG_TABLE_Z = (0x7031, 0x1002)  # table z position, mm
_TAG_SOD    = (0x7031, 0x1003)  # source-to-isocenter distance, mm (constant)
_TAG_SDD    = (0x7031, 0x1031)  # source-to-detector distance, mm (constant)


def _read_float_tag(ds, tag, default=None):
    elem = ds.get(tag)
    if elem is None or len(elem.value) < 4:
        return default
    return struct.unpack('<f', elem.value[:4])[0]


def _sorted_dcm_files(dcm_dir):
    files = sorted(
        [f for f in os.listdir(dcm_dir) if f.endswith('.dcm')],
        key=lambda f: int(re.search(r'\d+', f).group())
    )
    return [os.path.join(dcm_dir, f) for f in files]


class ACRPhantomDataset:
    """Helical fan-beam CT of the ACR phantom (TCIA LDCT-and-Projection-Data).

    Args:
        data_dir: Directory containing the DICOM-CT-PD files (*.dcm).
        split:    'train' or 'test'.
        every_nth: Load every N-th view (default 10 → ~1800 of 18033 views).
                   Reduces memory and loading time substantially.
    """

    def __init__(self, data_dir, split='train', every_nth=10, **kwargs):
        all_files = _sorted_dcm_files(data_dir)
        if not all_files:
            raise FileNotFoundError(f"No .dcm files found in {data_dir}")

        # Use denser sampling for train, sparser for test
        if split == 'test':
            stride = every_nth * 5
        else:
            stride = every_nth
        files = all_files[::stride]

        # Read geometry from first file
        ref = pydicom.dcmread(files[0])
        SOD_phys = _read_float_tag(ref, _TAG_SOD, default=595.0)   # mm
        SDD_phys = _read_float_tag(ref, _TAG_SDD, default=1085.6)  # mm
        n_ch  = ref.Rows     # 736 fan-beam channels
        n_row = ref.Columns  # 64 axial detector rows
        rescale_slope = float(ref.RescaleSlope)
        rescale_intercept = float(ref.RescaleIntercept)
        fov_mm = float(getattr(ref, 'DataCollectionDiameter', 500.0))

        # Full z extent: estimate from first and last file
        first = pydicom.dcmread(all_files[0])
        last  = pydicom.dcmread(all_files[-1])
        z_first = _read_float_tag(first, _TAG_TABLE_Z, 0.0)
        z_last  = _read_float_tag(last,  _TAG_TABLE_Z, 0.0)
        total_z_mm = abs(z_last - z_first) + fov_mm  # approximate scan length

        # Scene normalisation: fit into [-1, 1]^3
        max_phys = max(fov_mm, total_z_mm)
        scene_scale = 2.0 / max_phys
        SOD_n = SOD_phys * scene_scale
        SDD_n = SDD_phys * scene_scale

        # Fan-beam detector geometry:
        # Channel i: fan angle φ_i measured from central ray
        # Row j: axial offset at detector
        # Physical fan half-angle = arcsin(fov/2 / SOD)
        fan_half_angle = np.arcsin((fov_mm / 2.0) / SOD_phys)
        # Assume uniform channel spacing across 2*fan_half_angle
        ch_angles = np.linspace(-fan_half_angle, fan_half_angle, n_ch, endpoint=True)  # [n_ch]

        # Axial row spacing at detector: derived from 0.625 mm slice thickness × M
        row_pitch_isocenter = 0.625  # mm per detector row at isocenter (Siemens standard 64-slice)
        row_pitch_det = row_pitch_isocenter * (SDD_phys / SOD_phys)  # at detector plane
        row_offsets = (np.arange(n_row) - (n_row - 1) / 2.0) * row_pitch_det  # [n_row] mm at detector

        all_rays = []
        all_projs = []

        for fpath in files:
            ds = pydicom.dcmread(fpath)
            angle = _read_float_tag(ds, _TAG_ANGLE, 0.0)   # radians
            table_z = _read_float_tag(ds, _TAG_TABLE_Z, 0.0)  # mm

            # Source position on helix
            src_x = SOD_phys * np.cos(angle)
            src_y = SOD_phys * np.sin(angle)
            src_z = table_z
            src = np.array([src_x, src_y, src_z], dtype=np.float64)

            # Projection data (736, 64): rows=channels, cols=detector rows
            px = ds.pixel_array.astype(np.float64)  # uint16
            proj = px * rescale_slope + rescale_intercept  # nepers
            proj = np.clip(proj, 0.0, None)

            rays = _build_helical_rays(
                angle, src, SDD_phys, ch_angles, row_offsets,
                n_ch, n_row, scene_scale
            )
            all_rays.append(rays)
            all_projs.append(proj)  # (n_ch, n_row) → kept as-is in nepers

        # Stack: [N_views, n_ch, n_row, 6] and [N_views, n_ch, n_row, 1]
        self.all_rays = torch.tensor(np.stack(all_rays, axis=0), dtype=torch.float32)
        projs_np = np.stack(all_projs, axis=0)[..., np.newaxis]  # [N, n_ch, n_row, 1]
        self.all_projections = torch.tensor(projs_np, dtype=torch.float32)

        self.vol_gt = None  # Not available in this collection
        self.pixel_size = None
        self.pixel_ang_size = None
        self.scene_scale = scene_scale
        self.SOD = SOD_n
        self.SDD = SDD_n


def _build_helical_rays(angle, src_phys, SDD_phys, ch_angles, row_offsets, n_ch, n_row, scene_scale):
    """Build [n_ch, n_row, 6] normalised ray tensor for one helical fan-beam view.

    Fan-beam geometry: source at src_phys, detector on the far side at SDD_phys.
    Channel i: fan angle ch_angles[i] (radians) from the central ray in the XY plane.
    Row j: axial offset row_offsets[j] (mm) at the detector plane.
    """
    # --- vectorised, no Python loops ---

    # Detector pixel XY positions: each channel sits at angle (θ+π+φ_i) from source
    total_angles = angle + np.pi + ch_angles  # [n_ch]
    det_x = src_phys[0] + SDD_phys * np.cos(total_angles)  # [n_ch]
    det_y = src_phys[1] + SDD_phys * np.sin(total_angles)  # [n_ch]

    # Broadcast to [n_ch, n_row] with axial offsets
    det_x = np.tile(det_x[:, None], (1, n_row))  # [n_ch, n_row]
    det_y = np.tile(det_y[:, None], (1, n_row))
    det_z = src_phys[2] + row_offsets[None, :]              # [n_ch, n_row]

    # Ray direction vectors
    dx = det_x - src_phys[0]
    dy = det_y - src_phys[1]
    dz = det_z - src_phys[2]
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx = dx / norm
    dy = dy / norm
    dz = dz / norm

    src_n = src_phys * scene_scale
    origins = np.full((n_ch, n_row, 3), src_n, dtype=np.float32)
    dirs = np.stack([dx, dy, dz], axis=-1).astype(np.float32)
    return np.concatenate([origins, dirs], axis=-1)
