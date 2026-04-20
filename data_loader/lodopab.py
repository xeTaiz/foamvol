import os
import math
import numpy as np
import torch
import h5py


# LoDoPaB-CT geometry constants (from ODL parallel_beam_geometry)
SCENE_SCALE = 0.13       # physical half-extent in meters → normalized to [-1, 1]
SOURCE_DIST = 2.0        # source distance in normalized space (well outside [-1,1]³)
NUM_ANGLES_FULL = 1000
NUM_DET_PIXELS = 513
DET_MIN = -0.18384776    # meters
DET_MAX = 0.18384776     # meters
ANGLE_MIN = math.pi / NUM_ANGLES_FULL          # first angle (≈ 0.00157 rad)
ANGLE_MAX = math.pi * (1 - 1 / NUM_ANGLES_FULL)  # last angle (≈ 3.14002 rad)
SAMPLES_PER_FILE = 128


class LoDoPaBDataset:
    """Loads a single LoDoPaB-CT slice for radfoam reconstruction.

    LoDoPaB uses 2D parallel-beam geometry (ODL convention):
    - 1000 angles uniformly spaced in (0, π)
    - 513 detector pixels spanning [-0.1838, 0.1838] m
    - Image domain [-0.13, 0.13] m, normalized to [-1, 1] in model space

    Ray geometry for angle θ, detector position s (normalized):
        direction = [-sin(θ), cos(θ), 0]
        origin    = [s·cos(θ) + D·sin(θ),  s·sin(θ) - D·cos(θ),  0]
    where D = SOURCE_DIST = 2.0.

    Shape convention matching DataHandler:
        all_rays:        [num_angles, 1, num_det, 6]
        all_projections: [num_angles, 1, num_det, 1]
    The height dimension is 1 (2D CT slice, no z-axis detector extent).
    """

    beam_type = "parallel"

    def __init__(self, data_dir, split="train", sample_index=0, split_override=None):
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory containing LoDoPaB HDF5 files.
        split : {"train", "validation", "test"}
            Which split to load from (passed by DataHandler).
        sample_index : int
            Index of the sample (CT slice) to reconstruct.
        split_override : str or None
            If set, always load from this HDF5 split regardless of `split`.
            Useful when only one split is downloaded (e.g. split_override="test").
        """
        num_angles = None  # always use all 1000 angles
        if split_override is not None:
            part = split_override
        elif split == "val":
            part = "validation"
        else:
            part = split

        sinogram = self._load_sinogram(data_dir, part, sample_index)  # (1000, 513)

        # Subsample angles. None / 0 / -1 / >= NUM_ANGLES_FULL → use all 1000.
        all_angle_indices = np.arange(NUM_ANGLES_FULL)
        if num_angles is not None and 0 < num_angles < NUM_ANGLES_FULL:
            step = NUM_ANGLES_FULL / num_angles
            sel = np.round(np.arange(num_angles) * step).astype(int)
            sel = np.clip(sel, 0, NUM_ANGLES_FULL - 1)
            sinogram = sinogram[sel]
            angle_indices = sel
        else:
            angle_indices = all_angle_indices

        n_angles = len(angle_indices)

        # Full angle array (0, π) exclusive, matching ODL's parallel_beam_geometry
        full_angles = np.linspace(ANGLE_MIN, ANGLE_MAX, NUM_ANGLES_FULL)
        angles = full_angles[angle_indices]  # (n_angles,)

        # Detector pixel centers in normalized model space
        det_coords_m = np.linspace(DET_MIN, DET_MAX, NUM_DET_PIXELS)
        det_coords_norm = det_coords_m / SCENE_SCALE  # → [-1.414, 1.414]

        self.pixel_size = (DET_MAX - DET_MIN) / (NUM_DET_PIXELS - 1) / SCENE_SCALE
        self.angle_step = (ANGLE_MAX - ANGLE_MIN) / (NUM_ANGLES_FULL - 1)

        rays = self._build_rays(angles, det_coords_norm)  # (n_angles, 513, 6)

        # Add height dimension: [n_angles, 1, 513, 6]
        self.all_rays = torch.tensor(rays, dtype=torch.float32).unsqueeze(1)

        # Projections: (n_angles, 513) → (n_angles, 1, 513, 1)
        proj = sinogram.astype(np.float32)
        self.all_projections = torch.tensor(proj, dtype=torch.float32).unsqueeze(1).unsqueeze(-1)

    def _load_sinogram(self, data_dir, part, sample_index):
        """Load a single sinogram (1000, 513) from the HDF5 files."""
        file_idx = sample_index // SAMPLES_PER_FILE
        in_file_idx = sample_index % SAMPLES_PER_FILE

        obs_path = os.path.join(
            data_dir, f"observation_{part}_{file_idx:03d}.hdf5"
        )
        if not os.path.exists(obs_path):
            raise FileNotFoundError(
                f"LoDoPaB observation file not found: {obs_path}\n"
                f"Run 'python setup_lodopab.py --path {data_dir}' to download the dataset."
            )

        with h5py.File(obs_path, "r") as f:
            sinogram = f["data"][in_file_idx]  # (1000, 513)

        return sinogram  # numpy array

    def _build_rays(self, angles, det_coords_norm):
        """Compute ray origins and directions for all (angle, detector) pairs.

        Parameters
        ----------
        angles : np.ndarray, shape (n_angles,)
            Rotation angles in radians.
        det_coords_norm : np.ndarray, shape (n_det,)
            Normalized detector pixel centers.

        Returns
        -------
        rays : np.ndarray, shape (n_angles, n_det, 6)
            [ox, oy, oz, dx, dy, dz] for each ray.
        """
        n_angles = len(angles)
        n_det = len(det_coords_norm)
        D = SOURCE_DIST

        sin_t = np.sin(angles)   # (n_angles,)
        cos_t = np.cos(angles)   # (n_angles,)

        # Broadcast: (n_angles, 1) and (1, n_det)
        sin_t = sin_t[:, None]
        cos_t = cos_t[:, None]
        s = det_coords_norm[None, :]  # (1, n_det)

        # Ray direction: [-sin(θ), cos(θ), 0]  (ODL convention)
        dx = -sin_t * np.ones_like(s)   # (n_angles, n_det)
        dy = cos_t * np.ones_like(s)
        dz = np.zeros_like(dx)

        # Ray origin: s·[cos(θ), sin(θ)] - D·[-sin(θ), cos(θ)]
        #           = [s·cos(θ) + D·sin(θ),  s·sin(θ) - D·cos(θ),  0]
        ox = s * cos_t + D * sin_t
        oy = s * sin_t - D * cos_t
        oz = np.zeros_like(ox)

        rays = np.stack([ox, oy, oz, dx, dy, dz], axis=-1)  # (n_angles, n_det, 6)
        return rays
