"""GPU finite-difference checks for CUDA-native independent side densities.

Covers crossing/non-crossing, dp signs, both side logits, asymmetric and
near-air raw values. Run: micromamba run -n radfoam python test/test_thin_surface_independent_backward_cuda.py
"""
import os, sys, importlib.util
import torch
import torch.nn as nn
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_spec = importlib.util.spec_from_file_location("adv", os.path.join(os.path.dirname(__file__), "test_thin_surface_crossing_dp_adv.py"))
adv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(adv)


def activate(model, raw_plus, raw_minus, scale=1.7):
    n = model.primal_points.shape[0]
    model.raw_plus = nn.Parameter(torch.full((n,1), raw_plus, device="cuda"))
    model.raw_minus = nn.Parameter(torch.full((n,1), raw_minus, device="cuda"))
    model.density.requires_grad_(False)
    model._thin_surface_density_mode = "independent"
    model._thin_surface_active = True
    model.activation_scale = scale
    return model


def render_loss(model, rays):
    start = model.get_starting_point(rays, model.primal_points, model.aabb_tree)
    return model(rays, start)[0].sum()


def fd_one(model, rays, name, eps=1e-3):
    for p in (model.raw_plus, model.raw_minus, model.primal_points,
              model.quaternions, model.texel_sites_2d, model.texel_heights):
        p.grad = None
    render_loss(model, rays).backward()
    p = getattr(model, name); g = p.grad.detach().flatten()
    idx = int(g.abs().argmax()); analytic = float(g[idx])
    with torch.no_grad():
        orig = float(p.flatten()[idx])
        p.flatten()[idx] = orig + eps; lp = float(render_loss(model, rays))
        p.flatten()[idx] = orig - eps; lm = float(render_loss(model, rays))
        p.flatten()[idx] = orig
    fd = (lp-lm)/(2*eps)
    rel = abs(analytic-fd)/max(abs(analytic),abs(fd),1e-8)
    return idx, analytic, fd, rel


def require_fd(label, model, rays, names=("raw_plus","raw_minus"), tol=0.02):
    print("\n--", label, "--")
    for name in names:
        idx,a,f,r = fd_one(model,rays,name)
        print(f"{name} idx={idx} analytic={a:.7g} fd={f:.7g} rel={r:.4g}")
        assert abs(a) > 1e-7 or abs(f) > 1e-7, f"{label}: {name} has no observable gradient"
        assert r < tol, f"{label}: {name} rel_err {r} >= {tol}"


def main():
    assert torch.cuda.is_available()
    rays = adv._linear_x_axis_rays("cuda")

    # dp>0, flat internal plane crossing the cell chord.
    m = activate(adv._build_scalar_or_thin_scene("cuda", True), .3, -.2)
    require_fd("crossing dp>0 asymmetric", m, rays)

    # dp<0 with the plane restored to the cell interior: crossing branch.
    m = activate(adv._build_scene_minus_side("cuda"), .3, -.2)
    with torch.no_grad(): m.texel_heights.zero_()
    require_fd("crossing dp<0 asymmetric", m, rays)

    # dp<0 non-crossing plus-side and minus-side branches.
    m = activate(adv._build_scene_minus_side("cuda"), .3, -.2)
    require_fd("noncrossing dp<0 plus-side", m, rays, ("raw_plus",))
    m = activate(adv._build_scene_minus_side("cuda"), .3, -.2)
    with torch.no_grad(): m.texel_heights.fill_(5.0)
    require_fd("noncrossing dp<0 minus-side", m, rays, ("raw_minus",))

    # Low-density air regime: softplus derivative is small but usable/correct.
    m = activate(adv._build_scalar_or_thin_scene("cuda", True), -.5, -.5)
    require_fd("near-air crossing", m, rays, tol=.03)

    # Symmetric grazing fixture: the fallback weights both equal raw sides by
    # exactly one half, so their complete gradient tensors must match.
    m = activate(adv._build_scalar_or_thin_scene("cuda", True), -.2, -.2)
    dirs = torch.tensor([[0., 1., 0.]] * 3, device="cuda")
    grazing_rays = torch.cat([-1.5 * dirs, dirs], dim=-1)
    for p in (m.raw_plus, m.raw_minus):
        p.grad = None
    render_loss(m, grazing_rays).backward()
    assert torch.allclose(m.raw_plus.grad, m.raw_minus.grad,
                          atol=1e-7, rtol=1e-7)
    print("symmetric grazing raw-side gradients: PASS")

    # Base density is excluded from independent rendering/autograd.
    assert m.density.grad is None
    print("\nSUMMARY: ALL INDEPENDENT-SIDE BACKWARD CUDA GRADCHECKS PASSED")


if __name__ == "__main__": main()
