import math
import torch
import numpy as np


class CTSyntheticDataset:
    """Generates parallel-beam CT projection data from an analytical sphere phantom.

    Phantom: 3D sphere of radius R centered at origin with uniform density=1.0.
    Analytical line integral for a ray at perpendicular distance d from center:
        projection = 2 * sqrt(R^2 - d^2) if d < R, else 0.

    Rays are organized as [num_angles, det_h, det_w, 6] where each ray is
    (origin_x, origin_y, origin_z, dir_x, dir_y, dir_z).
    """

    def __init__(self, data_dir=None, split="train", downsample=1,
                 num_angles=180, detector_size=128, radius=1.0,
                 source_distance=5.0):
        self.radius = radius
        self.source_distance = source_distance
        self.num_angles = num_angles
        self.det_h = detector_size // downsample
        self.det_w = detector_size // downsample
        self.detector_extent = 2.0 * radius * 1.5  # cover the sphere with margin
        self.pixel_size = self.detector_extent / detector_size
        self.beam_type = "parallel"

        if split == "train":
            angles = torch.linspace(0, math.pi, num_angles + 1)[:-1]
        else:
            # Offset test angles by half a step
            step = math.pi / num_angles
            angles = torch.linspace(step / 2, math.pi - step / 2, max(num_angles // 4, 1))

        self.all_rays, self.all_projections = self._generate(angles)

    def _generate(self, angles):
        """Generate parallel-beam rays and their analytical projections."""
        R = self.radius
        det_h = self.det_h
        det_w = self.det_w
        extent = self.detector_extent
        dist = self.source_distance

        # Detector pixel coordinates (centered)
        u = torch.linspace(-extent / 2, extent / 2, det_w)
        v = torch.linspace(-extent / 2, extent / 2, det_h)
        uu, vv = torch.meshgrid(u, v, indexing="xy")  # [det_h, det_w]

        all_rays = []
        all_projections = []

        for theta in angles:
            ct = torch.cos(theta)
            st = torch.sin(theta)

            # Parallel beam: direction is (cos theta, sin theta, 0)
            direction = torch.tensor([ct.item(), st.item(), 0.0])

            # Perpendicular direction in the XY plane
            perp = torch.tensor([-st.item(), ct.item(), 0.0])

            # Origins: on a plane perpendicular to direction, at -dist along direction
            # u indexes the perpendicular direction, v indexes z
            origins = (
                -dist * direction.unsqueeze(0).unsqueeze(0)  # [1,1,3]
                + uu.unsqueeze(-1) * perp.unsqueeze(0).unsqueeze(0)  # [det_h, det_w, 3]
                + vv.unsqueeze(-1) * torch.tensor([0.0, 0.0, 1.0]).unsqueeze(0).unsqueeze(0)
            )  # [det_h, det_w, 3]

            directions = direction.unsqueeze(0).unsqueeze(0).expand(det_h, det_w, 3)

            rays = torch.cat([origins, directions], dim=-1)  # [det_h, det_w, 6]

            # Analytical projection: perpendicular distance to z-axis
            # For a sphere at origin, the perpendicular distance of a ray to
            # the sphere center is sqrt(u^2 + v^2) where u, v are detector coords
            d_sq = uu ** 2 + vv ** 2
            inside = d_sq < R ** 2
            proj = torch.zeros(det_h, det_w)
            proj[inside] = 2.0 * torch.sqrt(R ** 2 - d_sq[inside])

            all_rays.append(rays)
            all_projections.append(proj)

        all_rays = torch.stack(all_rays, dim=0)  # [num_angles, det_h, det_w, 6]
        all_projections = torch.stack(all_projections, dim=0)  # [num_angles, det_h, det_w]
        all_projections = all_projections.unsqueeze(-1)  # [num_angles, det_h, det_w, 1]

        return all_rays, all_projections
