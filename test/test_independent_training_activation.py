"""GPU smoke for iteration-zero independent initialization and resume."""
import importlib.util
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from split_voxelize import voxelize_split
SPEC = importlib.util.spec_from_file_location(
    "forward_fixture",
    os.path.join(os.path.dirname(__file__), "test_thin_surface_independent_forward_cuda.py"),
)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def optimizer_args():
    return SimpleNamespace(
        points_lr_init=2e-4, points_lr_final=5e-6,
        density_lr_init=5e-2, density_lr_final=1e-2,
        freeze_points=100, points_hard_freeze_at=0,
        thin_surface_density_mode="independent",
        thin_surface_relative_delta=False,
        thin_surface_raw_side_lr_init=2e-4,
        thin_surface_raw_side_lr_final=2e-5,
        thin_surface_start=0, thin_surface_K=4,
        thin_surface_lr_scale=1.0,
        thin_surface_delta_lr_scale=0.0,
        thin_surface_quat_lr_scale=0.0,
        thin_surface_sites_lr_scale=0.0,
        thin_surface_heights_lr_scale=0.0,
        thin_surface_delta_clip=2.0,
        thin_surface_grad_clip=1.0,
        thin_surface_delta_max_frac=.5,
        warmup_steps=0,
    )


def activate(model, args):
    # Fixture already declares an optimizer; replace it to exercise production order.
    model.declare_optimizer(args, warmup=0, max_iterations=100)
    model.initialize_thin_surface(args, K=4)
    return model


def main():
    assert torch.cuda.is_available()
    args = optimizer_args()
    model = activate(fixture._make_scene(device="cuda"), args)
    assert model._thin_surface_active
    assert model._thin_surface_density_mode == "independent"
    assert all(hasattr(model, n) for n in (
        "raw_plus", "raw_minus", "quaternions", "texel_sites_2d", "texel_heights"))
    names = {g["name"] for g in model.optimizer.param_groups}
    assert "density" not in names
    assert {"raw_plus", "raw_minus", "quaternions", "texel_sites_2d", "texel_heights"} <= names
    assert model.density.requires_grad is False

    rays = fixture._make_test_rays(model)
    start = model.get_starting_point(rays, model.primal_points, model.aabb_tree)
    out_before = model(rays, start)[0].detach()
    loss = model(rays, start)[0].sum()
    model.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert model.raw_plus.grad is not None and model.raw_plus.grad.isfinite().all()
    assert model.raw_minus.grad is not None and model.raw_minus.grad.isfinite().all()
    assert model.density.grad is None

    tmp = os.path.join(tempfile.mkdtemp(), "independent.pt")
    model.save_pt(tmp)
    saved = torch.load(tmp, map_location="cpu")
    assert saved["thin_surface"]["active"] is True
    assert saved["thin_surface"]["density_mode"] == "independent"
    assert "density_delta" not in saved
    for name in ("raw_plus", "raw_minus", "quaternions", "texel_sites_2d", "texel_heights"):
        assert name in saved

    resumed = fixture._make_scene(device="cuda")
    resumed.load_pt(tmp)
    resumed.declare_optimizer(args, warmup=0, max_iterations=100)
    resumed.initialize_thin_surface(args, K=4)
    assert resumed._thin_surface_active
    for name in ("quaternions", "texel_sites_2d", "texel_heights"):
        assert torch.equal(getattr(resumed, name).detach().cpu(), saved[name])
    resumed_names = {g["name"] for g in resumed.optimizer.param_groups}
    assert "density" not in resumed_names
    assert {"raw_plus", "raw_minus"} <= resumed_names
    start2 = resumed.get_starting_point(rays, resumed.primal_points, resumed.aabb_tree)
    out_after = resumed(rays, start2)[0].detach()
    assert torch.allclose(out_before, out_after, atol=1e-6, rtol=1e-6), (
        out_before - out_after).abs().max().item()

    # Hard-side voxelizer must reproduce scalar/relative-zero/independent-zero.
    scalar_data = dict(saved)
    for name in ("raw_plus", "raw_minus", "quaternions",
                 "texel_sites_2d", "texel_heights", "thin_surface"):
        scalar_data.pop(name, None)
    relative_data = dict(saved)
    relative_data.pop("raw_plus")
    relative_data.pop("raw_minus")
    relative_data["density_delta"] = torch.zeros_like(saved["density"])
    relative_data["thin_surface"] = dict(saved["thin_surface"])
    relative_data["thin_surface"].update({
        "active": True, "density_mode": "relative",
        "relative_delta": True, "delta_max_frac": .5,
    })
    scalar_path = os.path.join(os.path.dirname(tmp), "scalar.pt")
    relative_path = os.path.join(os.path.dirname(tmp), "relative.pt")
    torch.save(scalar_data, scalar_path)
    torch.save(relative_data, relative_path)
    volumes = []
    for label, path in (("scalar", scalar_path), ("relative", relative_path),
                        ("independent", tmp)):
        out_path = os.path.join(os.path.dirname(tmp), f"{label}.npy")
        volumes.append(voxelize_split(
            path, resolution=12, output_path=out_path, supersample=2))
    assert np.max(np.abs(volumes[0] - volumes[1])) < 1e-6
    assert np.max(np.abs(volumes[0] - volumes[2])) < 1e-6

    resumed.enforce_hard_point_freeze(0)
    point_group = next(g for g in resumed.optimizer.param_groups
                       if g["name"] == "primal_points")
    assert point_group["lr"] == 0.0
    assert resumed.primal_points.requires_grad is False
    print("SUMMARY: INDEPENDENT ITERATION-ZERO TRAIN/RESUME SMOKE PASSED")


if __name__ == "__main__":
    main()
