"""Data loader for Siemens Inveon preclinical cone-beam CT data.

Reads the Inveon `.cat` (raw projections) and `.ct.img` (reconstructed volume)
file pair produced by the Siemens Inveon PET/CT scanner.

Binary layout of `.cat` file:
  [50560-byte header]
  [dark frame:  ny × nx int16]
  [light frame: ny × nx int16]
  [bed-1 projections: 121 × ny × nx int16]
  [bed-2 projections: 121 × ny × nx int16]
  [bed-3 projections: 121 × ny × nx int16]

Geometry (from header):
  SDD = 402.99 mm  (source to detector)
  SOD = 344.27 mm  (source to isocenter)
  nx = 768, ny = 512 (detector columns × rows, 4× binned)
  detector pixel pitch ≈ 0.267 mm (physical, derived from reconstruction)
  rotation: 0° → 220° over 121 projections (counter-clockwise, partial rotation)
  3 bed positions, Z step ≈ ±8 mm from centre

Flat-field correction:
  transmission = (raw - dark) / (light - dark), clipped to (1e-6, 1)
  line_integral = -log(transmission) = attenuation × path_length

Usage:
  ds = InveonDataset('/path/to/NEMA_S6_...')  (pass prefix without extension)
  ds = InveonDataset('/path/to/NEMA_S6_...', beds=[1])  (single bed)
"""

import os
import re
import numpy as np
import torch


def _parse_header(hdr_path):
    with open(hdr_path) as f:
        text = f.read()
    def get(key, cast=float, default=None):
        m = re.search(rf'^{key}\s+(.+)$', text, re.M)
        if m is None:
            return default
        try:
            return cast(m.group(1).strip())
        except ValueError:
            return default
    return {
        'hdr_size':  int(get('ct_header_size', int)),
        'nx':        int(get('ct_proj_size_transaxial', int)),
        'ny':        int(get('ct_proj_size_axial', int)),
        'n_proj':    int(get('number_of_projections', int)),
        'n_beds':    int(get('number_of_bed_positions', int)),
        'SDD':       get('ct_source_to_detector'),     # mm
        'SOD':       get('ct_source_to_crot'),         # mm
        'rot_start': get('rotating_stage_start_position', default=0.0),   # deg
        'rot_stop':  get('rotating_stage_stop_position', default=220.0),  # deg
        'rot_dir':   int(get('rotation_direction', int, default=1)),
    }


def _parse_vol_header(hdr_path):
    with open(hdr_path) as f:
        text = f.read()
    def get(key, cast=float, default=None):
        m = re.search(rf'^{key}\s+(.+)$', text, re.M)
        if m is None:
            return default
        try:
            return cast(m.group(1).strip())
        except ValueError:
            return default
    return {
        'nx': int(get('x_dimension', int)),
        'ny': int(get('y_dimension', int)),
        'nz': int(get('z_dimension', int)),
        'data_type': int(get('data_type', int, default=2)),
        'pixel_size_x': get('pixel_size_x'),  # mm
        'pixel_size_z': get('pixel_size_z'),  # mm
    }


_DTYPE_MAP = {2: np.int16, 4: np.float32, 6: np.int32}


class InveonDataset:
    """Cone-beam CT dataset from Siemens Inveon preclinical scanner.

    Args:
        prefix: Path to the data files without extension, e.g.
                '/data/NEMA_S6_..._v1' (loader appends .cat, .cat.hdr, etc.)
        split:  'train' or 'test'. Training uses the CT projections;
                'test' generates 20 held-out angles (same geometry, no real data).
        beds:   Which bed positions to include (1-indexed). Default: all [1,2,3].
    """

    def __init__(self, data_dir, split='train', beds=None, **kwargs):
        prefix = data_dir  # data_dir is the file prefix (without extension)
        cat_hdr  = prefix + '.cat.hdr'
        cat_file = prefix + '.cat'
        img_hdr  = prefix + '.ct.img.hdr'
        img_file = prefix + '.ct.img'

        h  = _parse_header(cat_hdr)
        vh = _parse_vol_header(img_hdr)

        SDD = h['SDD']
        SOD = h['SOD']
        nx  = h['nx']
        ny  = h['ny']
        n_proj_per_bed = h['n_proj']
        n_beds = h['n_beds']
        if beds is None:
            beds = list(range(1, n_beds + 1))

        # Detector pixel pitch (physical, mm) derived from reconstruction:
        #   recon_voxel = det_pitch / (SDD/SOD)  →  det_pitch = voxel × (SDD/SOD)
        voxel_xy = vh['pixel_size_x']  # mm
        det_pitch = voxel_xy * (SDD / SOD)  # mm

        # Total volume extent
        n_recon_z = vh['nz']
        voxel_z   = vh.get('pixel_size_z') or voxel_xy
        total_z_mm = n_recon_z * voxel_z          # mm
        total_xy_mm = vh['nx'] * voxel_xy          # mm (X = Y)

        # Scene normalisation: scale to fit the volume into [-1, 1]^3
        scene_scale = 2.0 / max(total_xy_mm, total_z_mm)

        SDD_n = SDD * scene_scale
        SOD_n = SOD * scene_scale
        det_pitch_n = det_pitch * scene_scale

        # Bed Z offsets: beds evenly distributed over Z range
        axial_fov_per_bed = ny * voxel_xy  # FOV at isocenter per bed position
        if n_beds > 1:
            bed_step = (total_z_mm - axial_fov_per_bed) / (n_beds - 1)
        else:
            bed_step = 0.0
        z_offsets_mm = [-(n_beds - 1) / 2.0 * bed_step + (b - 1) * bed_step
                        for b in range(1, n_beds + 1)]
        z_offsets_n = [z * scene_scale for z in z_offsets_mm]

        # Rotation angles
        rot_start = np.radians(h['rot_start'])
        rot_stop  = np.radians(h['rot_stop'])
        angles_per_bed = np.linspace(rot_start, rot_stop, n_proj_per_bed, endpoint=False)
        if h['rot_dir'] == 1:  # CCW
            angles_per_bed = -angles_per_bed

        # Read dark and light frames from .cat
        hdr_size = h['hdr_size']
        with open(cat_file, 'rb') as f:
            f.seek(hdr_size)
            dark  = np.frombuffer(f.read(nx * ny * 2), dtype=np.int16).reshape(ny, nx).astype(np.float32)
            light = np.frombuffer(f.read(nx * ny * 2), dtype=np.int16).reshape(ny, nx).astype(np.float32)
            ct_frames = []
            for _ in range(n_beds * n_proj_per_bed):
                raw = np.frombuffer(f.read(nx * ny * 2), dtype=np.int16).reshape(ny, nx).astype(np.float32)
                ct_frames.append(raw)
        ct_frames = np.stack(ct_frames, axis=0)  # [n_beds*n_proj, ny, nx]

        # Flat-field correction → line integrals
        denom = np.maximum(light - dark, 1.0)  # avoid divide-by-zero
        projs_all = []
        for i in range(len(ct_frames)):
            transmission = (ct_frames[i] - dark) / denom
            transmission = np.clip(transmission, 1e-6, 1.0)
            projs_all.append(-np.log(transmission))
        projs_all = np.stack(projs_all, axis=0)  # [n_beds*n_proj, ny, nx]

        # Build rays for selected beds
        all_rays = []
        all_projs = []
        rng = np.random.default_rng(42)

        if split == 'train':
            for bed_idx in beds:
                b = bed_idx - 1  # 0-indexed
                z0_n = z_offsets_n[b]
                proj_slice = projs_all[b * n_proj_per_bed:(b + 1) * n_proj_per_bed]  # [121, ny, nx]

                for i, angle in enumerate(angles_per_bed):
                    rays = _build_cone_rays(angle, SOD_n, SDD_n, nx, ny, det_pitch_n, z0_n)
                    all_rays.append(rays)
                    all_projs.append(proj_slice[i])  # physical nepers, no scene_scale
        else:
            # 20 held-out angles from bed 2 (centre)
            b = (n_beds // 2)
            z0_n = z_offsets_n[b]
            test_idx = np.linspace(0, n_proj_per_bed - 1, 20, dtype=int)
            proj_slice = projs_all[b * n_proj_per_bed:(b + 1) * n_proj_per_bed]
            for i in test_idx:
                rays = _build_cone_rays(angles_per_bed[i], SOD_n, SDD_n, nx, ny, det_pitch_n, z0_n)
                all_rays.append(rays)
                all_projs.append(proj_slice[i])

        self.all_rays = torch.tensor(np.stack(all_rays, axis=0), dtype=torch.float32)
        self.all_projections = torch.tensor(
            np.stack(all_projs, axis=0)[..., np.newaxis], dtype=torch.float32)

        # Load and normalise ground-truth volume (HU → [0,1])
        dt = _DTYPE_MAP.get(vh['data_type'], np.float32)
        vol = np.fromfile(img_file, dtype=dt).reshape(vh['nz'], vh['ny'], vh['nx'])
        vol = vol.astype(np.float32)
        vol = np.clip(vol, -1000.0, 3000.0)
        vol = (vol + 1000.0) / 4000.0  # HU: [-1000,3000] → [0,1]
        # .ct.img is stored (Z, Y, X); vol_gt convention is (X, Y, Z)
        self.vol_gt = torch.tensor(np.transpose(vol, (2, 1, 0)), dtype=torch.float32)

        # Expose pixel size for jittering (isotropic)
        self.pixel_size = None
        self.pixel_ang_size = None  # we don't use angular jitter here

        # Scene metadata
        self.scene_scale = scene_scale
        self.SDD = SDD_n
        self.SOD = SOD_n


def _build_cone_rays(angle, SOD, SDD, nx, ny, det_pitch, z_offset):
    """Build [ny, nx, 6] ray tensor for one cone-beam projection.

    The source orbits in the XY plane at radius SOD from the rotation axis.
    The detector is at distance (SDD - SOD) on the far side.

    Coordinate system: right-handed, Z is the rotation axis.
    Angle=0 → source on the +X axis.
    """
    # Source position
    src = np.array([SOD * np.cos(angle), SOD * np.sin(angle), z_offset], dtype=np.float32)

    # Detector centre (opposite side from source)
    DSD = SDD
    det_centre = np.array([-( DSD - SOD) * np.cos(angle),
                            -(DSD - SOD) * np.sin(angle),
                            z_offset], dtype=np.float32)

    # Detector basis vectors
    det_u = np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=np.float32)   # transaxial
    det_v = np.array([0.0, 0.0, 1.0], dtype=np.float32)                         # axial

    # Pixel grid: u = transaxial (columns), v = axial (rows)
    iu = np.arange(nx, dtype=np.float32)
    iv = np.arange(ny, dtype=np.float32)
    iv_g, iu_g = np.meshgrid(iv, iu, indexing='ij')  # [ny, nx]

    u_offset = (iu_g + 0.5 - nx / 2.0) * det_pitch
    v_offset = (iv_g + 0.5 - ny / 2.0) * det_pitch

    # Detector pixel world positions
    px = det_centre[0] + u_offset * det_u[0] + v_offset * det_v[0]
    py = det_centre[1] + u_offset * det_u[1] + v_offset * det_v[1]
    pz = det_centre[2] + u_offset * det_u[2] + v_offset * det_v[2]

    # Ray directions (source → pixel), normalised
    dx = px - src[0]
    dy = py - src[1]
    dz = pz - src[2]
    norm = np.sqrt(dx**2 + dy**2 + dz**2)
    dx /= norm; dy /= norm; dz /= norm

    origins = np.stack([np.full_like(dx, src[0]),
                         np.full_like(dy, src[1]),
                         np.full_like(dz, src[2])], axis=-1)
    dirs = np.stack([dx, dy, dz], axis=-1)
    return np.concatenate([origins, dirs], axis=-1).astype(np.float32)
