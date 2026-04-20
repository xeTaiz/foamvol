import os
import math
import numpy as np
import torch
import tifffile


# --- Scanner constants (identical across all 3 modes) ---
SOD_MM = 431.019989          # source-to-object distance, mm
SDD_MM = 529.000488          # source-to-detector distance, mm
DET_PIX_MM = 0.0748          # physical detector pixel size, mm
N_ANGLES_RAW = 3601          # raw projections (0° … 360°, last duplicates first)
N_DET_RAW = 1912             # raw detector width, pixels

# The official reconstruction scripts bin the detector by 2 (sum adjacent pairs)
DET_BIN = 2
N_DET = N_DET_RAW // DET_BIN           # 956 pixels after binning
DET_PIX_BIN_MM = DET_PIX_MM * DET_BIN  # 0.1496 mm effective pixel size

# FOV and normalisation derived from the official scripts
FOV_MM = N_DET * DET_PIX_BIN_MM * SOD_MM / SDD_MM   # ≈ 116.5 mm
SCENE_SCALE = FOV_MM / 2.0                            # ≈ 58.25 mm  → [-1,1] model space

SOD_NORM = SOD_MM / SCENE_SCALE     # ≈ 7.40
SDD_NORM = SDD_MM / SCENE_SCALE     # ≈ 9.08
DET_PIX_NORM = DET_PIX_BIN_MM / SCENE_SCALE  # ≈ 0.00257

# Detector center-of-rotation correction (from official scripts).
# Slices 1–2830 and OOD early (5521–5870): 1.0 binned pixel shift.
# Slices 2831–5000 and OOD late (5871–6370): 0.0 shift.
CORR_SHIFT_BIG = 1.0    # binned pixels
CORR_SHIFT_SMALL = 0.0

SAMPLES_PER_FILE = 1    # each slice is its own directory


def _det_shift_for_slice(slice_id: int) -> float:
    """Return the detector shift in binned pixels for a given 1-based slice ID."""
    if (1 <= slice_id <= 2830) or (5521 <= slice_id <= 5870):
        return CORR_SHIFT_BIG
    return CORR_SHIFT_SMALL


class TwoDeteCTDataset:
    """2DeteCT fan-beam 2D CT dataset loader.

    Each 'sample' is one physical object slice. Three acquisition modes are
    available (mode1 = high-dose reference, mode2 = low-dose, mode3 = 60 kV).

    Preprocessing follows the official Reconstructions_2DeteCT.py script:
      1. 2× detector binning
      2. Dark/flat-field correction: T = (sino – dark) / (flat – dark)
      3. Detector shift interpolation (center-of-rotation correction)
      4. Beer-Lambert: proj = –log(T.clip(1e-6))
      5. Drop duplicate last angle → 3600 projections

    Ray geometry (fan-beam 2D, all rays in z = 0):
      Source at angle φ:  S = SOD_norm · [cos φ, sin φ, 0]
      Detector pixel u:   D = –(SDD_norm–SOD_norm)·[cos φ, sin φ, 0] + u·[–sin φ, cos φ, 0]
      Direction:          normalize(D – S)

    Output shapes (matching DataHandler convention):
      all_rays:         [3600, 1, 956, 6]
      all_projections:  [3600, 1, 956, 1]
    """

    beam_type = "cone"   # fan-beam uses angular (cone) jitter

    def __init__(self, data_dir, split="train", sample_index=1, mode=1,
                 split_override=None):
        """
        Parameters
        ----------
        data_dir : str
            Path to /mnt/hdd/2detectct (parent of slice00001/, slice00002/, …)
        split : ignored (all data is used for reconstruction of a single slice)
        sample_index : int
            1-based slice ID (1 … 6370).
        mode : {1, 2, 3}
            Acquisition mode. 1 = high-dose (90 kV / 90 W), 2 = low-dose,
            3 = 60 kV.
        split_override : ignored (kept for DataHandler compatibility)
        """
        slice_dir = os.path.join(data_dir, f"slice{sample_index:05d}", f"mode{mode}")
        if not os.path.isdir(slice_dir):
            raise FileNotFoundError(
                f"2DeteCT slice not found: {slice_dir}\n"
                f"Download with: python download_2detectct.py"
            )

        sinogram, projections = self._load_and_preprocess(slice_dir, sample_index)
        # sinogram: (3600, 956)

        angles = np.linspace(0, 2 * math.pi, N_ANGLES_RAW)[:-1]  # 3600 angles
        det_positions_norm = self._det_positions_norm()            # (956,)

        rays = self._build_rays(angles, det_positions_norm)        # (3600, 956, 6)

        # Add height dim: [3600, 1, 956, 6] / [3600, 1, 956, 1]
        self.all_rays = torch.from_numpy(rays.astype(np.float32)).unsqueeze(1)
        self.all_projections = torch.from_numpy(
            projections.astype(np.float32)
        ).unsqueeze(1).unsqueeze(-1)

        # Angular pixel size (for cone jitter): arctan(pixel / SDD)
        ang_per_pix = math.atan(DET_PIX_BIN_MM / SDD_MM)
        self.pixel_ang_size = (ang_per_pix, 0.0)   # no z-jitter (2D dataset)

        # c2ws / fx / fy not needed for 2D fan-beam (targeted sampling unsupported)
        self.c2ws = None
        self.fx = None
        self.fy = None

    # ------------------------------------------------------------------
    def _load_and_preprocess(self, slice_dir, slice_id):
        """Load TIFFs, bin, correct, and log-transform. Returns (3600, 956)."""
        sino  = tifffile.imread(os.path.join(slice_dir, "sinogram.tif")).astype(np.float32)
        dark  = tifffile.imread(os.path.join(slice_dir, "dark.tif")).astype(np.float32)
        flat1 = tifffile.imread(os.path.join(slice_dir, "flat1.tif")).astype(np.float32)
        flat2 = tifffile.imread(os.path.join(slice_dir, "flat2.tif")).astype(np.float32)
        flat  = (flat1 + flat2) / 2.0   # average flat fields

        # 2× detector binning (sum adjacent pixel pairs)
        sino = sino[:, 0::2] + sino[:, 1::2]   # (3601, 956)
        dark = dark[0, 0::2] + dark[0, 1::2]   # (956,)
        flat = flat[0, 0::2] + flat[0, 1::2]   # (956,)

        # Dark/flat correction
        data = (sino - dark) / (flat - dark + 1e-9)   # (3601, 956)

        # Drop duplicate last projection (360° == 0°)
        data = data[:-1]   # (3600, 956)

        # Center-of-rotation detector shift (linear interpolation)
        shift_pix = _det_shift_for_slice(slice_id)
        if shift_pix != 0.0:
            det_grid = np.arange(N_DET) * DET_PIX_BIN_MM
            det_grid_shifted = det_grid + shift_pix * DET_PIX_BIN_MM
            from scipy.interpolate import interp1d
            shift_fn = interp1d(
                det_grid, data, kind="linear", axis=1,
                bounds_error=False, fill_value="extrapolate"
            )
            data = shift_fn(det_grid_shifted).astype(np.float32)

        # Beer-Lambert: –log(transmission), clamp to avoid log(0)
        data = data.clip(1e-6, None)
        projections = -np.log(data)   # (3600, 956)

        return data, projections   # return transmission too for ref, but only proj used

    def _det_positions_norm(self):
        """Detector pixel centres in normalised model space (centred at 0)."""
        i = np.arange(N_DET, dtype=np.float32)
        u_mm = (i - (N_DET - 1) / 2.0) * DET_PIX_BIN_MM
        return u_mm / SCENE_SCALE   # (956,)

    def _build_rays(self, angles, det_norm):
        """
        Fan-beam ray geometry (all in z = 0 plane).

        Source at φ:        S = SOD_norm · [cos φ, sin φ, 0]
        Detector pixel u:   u · [–sin φ, cos φ, 0] – (SDD_norm–SOD_norm)·[cos φ, sin φ, 0]
        Direction:          normalize(det_pixel – source)
                          = normalize(–SDD_norm·[cos φ, sin φ] + u·[–sin φ, cos φ])
        """
        n_a = len(angles)
        n_d = len(det_norm)
        D = SDD_NORM
        R = SOD_NORM

        cos_a = np.cos(angles)[:, None]   # (n_a, 1)
        sin_a = np.sin(angles)[:, None]
        u = det_norm[None, :]              # (1, n_d)

        # Radial unit vector (source direction from centre)
        ox = R * cos_a * np.ones((n_a, n_d))
        oy = R * sin_a * np.ones((n_a, n_d))
        oz = np.zeros((n_a, n_d))

        # Raw direction vector (not yet normalised)
        rdx = -D * cos_a + u * (-sin_a)
        rdy = -D * sin_a + u *   cos_a
        rdz = np.zeros((n_a, n_d))

        norm = np.sqrt(rdx**2 + rdy**2 + 1e-30)
        dx = rdx / norm
        dy = rdy / norm
        dz = rdz

        return np.stack([ox, oy, oz, dx, dy, dz], axis=-1)   # (n_a, n_d, 6)
