import torch


class ErrorBox:
    def __init__(self):
        self.ray_error = None
        self.point_error = None


class TraceRays(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        pipeline,
        _points,
        _density,
        _point_adjacency,
        _point_adjacency_offsets,
        rays,
        start_point,
        return_contribution,
        _density_grad=None,
        _gradient_max_slope=5.0,
        _interpolation_mode=False,
        _idw_sigma=0.01,
        _idw_sigma_v=0.1,
        _per_cell_sigma=False,
        _per_neighbor_sigma=False,
        _cell_radius=None,
        _gaussian_mode=False,
        _density_peak=None,
        _delta_raw=None,
        _cov_raw=None,
        _thin_surface_mode=False,
        _density_delta=None,
        _quaternions=None,
        _texel_sites_2d=None,
        _texel_heights=None,
        _thin_K=4,
        _thin_temp=10.0,
        _thin_height_eps=1e-4,
        # M5 relative-delta rescue: when True the kernel computes
        # delta = rho * mu_bar * tanh(raw_delta) instead of treating
        # density_delta as a raw additive offset.  Bounded and nonneg-safe.
        _thin_surface_relative_delta=False,
        _thin_surface_delta_max_frac=0.5,
        # LC64 plan v3 Commit 2A -- independent-side raw logits
        # (each (N,1)).  When _thin_surface_independent_mode is True
        # the kernel reads raw_plus/raw_minus and computes
        # mu_plus = activation_scale * softplus(raw_plus, beta=10),
        # mu_minus = activation_scale * softplus(raw_minus, beta=10)
        # independently.  Geometry / crossing / dp-sign semantics are
        # reused from the legacy thin-surface branch.  Backward is NOT
        # implemented in this commit; TraceRays.backward raises.
        _raw_plus=None,
        _raw_minus=None,
        _thin_surface_independent_mode=False,
        _thin_surface_activation_scale=1.0,
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.pipeline = pipeline
        ctx.points = _points
        ctx.density = _density
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets
        ctx.has_density_grad = _density_grad is not None
        ctx.gradient_max_slope = _gradient_max_slope
        ctx.interpolation_mode = _interpolation_mode
        ctx.idw_sigma = _idw_sigma
        ctx.idw_sigma_v = _idw_sigma_v
        ctx.per_cell_sigma = _per_cell_sigma
        ctx.per_neighbor_sigma = _per_neighbor_sigma
        ctx.cell_radius = _cell_radius
        ctx.gaussian_mode = _gaussian_mode
        ctx.has_gaussian = _density_peak is not None
        ctx.thin_surface_mode = _thin_surface_mode
        ctx.has_thin_surface = _density_delta is not None
        ctx.thin_K = _thin_K
        ctx.thin_temp = _thin_temp
        ctx.thin_height_eps = _thin_height_eps
        ctx.thin_surface_relative_delta = _thin_surface_relative_delta
        ctx.thin_surface_delta_max_frac = _thin_surface_delta_max_frac
        # LC64 plan v3 Commit 2A -- independent-side raw logits.
        # Saved verbatim across the autograd context so backward can
        # gate on the discriminator and raise if the user calls
        # backward() under independent mode before Commit 2B.
        ctx.thin_surface_independent_mode = _thin_surface_independent_mode
        ctx.thin_surface_activation_scale = _thin_surface_activation_scale
        ctx.has_raw_plus = _raw_plus is not None
        ctx.has_raw_minus = _raw_minus is not None
        if _thin_surface_independent_mode:
            ctx.raw_plus = _raw_plus
            ctx.raw_minus = _raw_minus
        if ctx.has_density_grad:
            ctx.density_grad = _density_grad
        if ctx.has_gaussian:
            ctx.density_peak = _density_peak
            ctx.delta_raw = _delta_raw
            ctx.cov_raw = _cov_raw
        if ctx.has_thin_surface:
            ctx.density_delta = _density_delta
            ctx.quaternions = _quaternions
            ctx.texel_sites_2d = _texel_sites_2d
            ctx.texel_heights = _texel_heights

        results = pipeline.trace_forward(
            _points,
            _density,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            return_contribution=return_contribution,
            density_grad=_density_grad,
            gradient_max_slope=_gradient_max_slope,
            interpolation_mode=_interpolation_mode,
            idw_sigma=_idw_sigma,
            idw_sigma_v=_idw_sigma_v,
            per_cell_sigma=_per_cell_sigma,
            per_neighbor_sigma=_per_neighbor_sigma,
            cell_radius=_cell_radius,
            gaussian_mode=_gaussian_mode,
            density_peak=_density_peak,
            delta_raw=_delta_raw,
            cov_raw=_cov_raw,
            thin_surface_mode=_thin_surface_mode,
            density_delta=_density_delta,
            quaternions=_quaternions,
            texel_sites_2d=_texel_sites_2d,
            texel_heights=_texel_heights,
            thin_K=_thin_K,
            thin_temp=_thin_temp,
            thin_height_eps=_thin_height_eps,
            thin_surface_relative_delta=_thin_surface_relative_delta,
            thin_surface_delta_max_frac=_thin_surface_delta_max_frac,
            raw_plus=_raw_plus,
            raw_minus=_raw_minus,
            thin_surface_independent_mode=_thin_surface_independent_mode,
            thin_surface_activation_scale=_thin_surface_activation_scale,
        )

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["projection"],
            results.get("contribution", None),
            results.get("hit_count", None),
            results["num_intersections"],
            errbox,
        )

    @staticmethod
    def backward(
        ctx,
        grad_projection,
        grad_contribution,
        grad_hit_count,
        grad_num_intersections,
        errbox_grad,
    ):
        del grad_contribution
        del grad_hit_count
        del grad_num_intersections
        del errbox_grad

        # LC64 plan v3 Commit 2A -- independent-side backward is NOT
        # implemented in this commit.  Fail fast with a clear error
        # before any kernel launch (the C++ binding also rejects this
        # as a defensive second guard).  Calling backward under
        # independent mode is a configuration error, not a runtime
        # regression; the forward contract is the only supported path
        # until Commit 2B.
        if getattr(ctx, "thin_surface_independent_mode", False):
            raise NotImplementedError(
                "Independent-side backward is not implemented in "
                "LC64 plan v3 Commit 2A. The forward path renders "
                "raw_plus / raw_minus via the CUDA dispatch in "
                "ct_independent_forward, but the backward path lands "
                "in Commit 2B. Until then, calling backward() under "
                "thin_surface_independent_mode=True is a hard error "
                "(no silent scalar / legacy-delta fallback)."
            )

        rays = ctx.rays
        start_point = ctx.start_point
        pipeline = ctx.pipeline
        _points = ctx.points
        _density = ctx.density
        _point_adjacency = ctx.point_adjacency
        _point_adjacency_offsets = ctx.point_adjacency_offsets
        has_density_grad = ctx.has_density_grad
        _density_grad = ctx.density_grad if has_density_grad else None
        gradient_max_slope = ctx.gradient_max_slope
        interpolation_mode = ctx.interpolation_mode
        idw_sigma = ctx.idw_sigma
        idw_sigma_v = ctx.idw_sigma_v
        per_cell_sigma = ctx.per_cell_sigma
        per_neighbor_sigma = ctx.per_neighbor_sigma
        cell_radius = ctx.cell_radius
        gaussian_mode = ctx.gaussian_mode
        has_gaussian = ctx.has_gaussian
        _density_peak = ctx.density_peak if has_gaussian else None
        _delta_raw = ctx.delta_raw if has_gaussian else None
        _cov_raw = ctx.cov_raw if has_gaussian else None
        thin_surface_mode = ctx.thin_surface_mode
        has_thin_surface = ctx.has_thin_surface
        thin_K = ctx.thin_K
        thin_temp = ctx.thin_temp
        thin_height_eps = ctx.thin_height_eps
        thin_surface_relative_delta = ctx.thin_surface_relative_delta
        thin_surface_delta_max_frac = ctx.thin_surface_delta_max_frac
        _density_delta = ctx.density_delta if has_thin_surface else None
        _quaternions = ctx.quaternions if has_thin_surface else None
        _texel_sites_2d = ctx.texel_sites_2d if has_thin_surface else None
        _texel_heights = ctx.texel_heights if has_thin_surface else None
        # LC64 plan v3 Commit 2A -- independent-side raw logits are
        # NOT read by backward (the binding rejects independent-mode
        # backward above).  Pass None for legacy arg ordering.
        _raw_plus = None
        _raw_minus = None
        thin_surface_independent_mode = False
        thin_surface_activation_scale = 1.0

        results = pipeline.trace_backward(
            _points,
            _density,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            grad_projection,
            ctx.errbox.ray_error,
            density_grad=_density_grad,
            gradient_max_slope=gradient_max_slope,
            interpolation_mode=interpolation_mode,
            idw_sigma=idw_sigma,
            idw_sigma_v=idw_sigma_v,
            per_cell_sigma=per_cell_sigma,
            per_neighbor_sigma=per_neighbor_sigma,
            cell_radius=cell_radius,
            gaussian_mode=gaussian_mode,
            density_peak=_density_peak,
            delta_raw=_delta_raw,
            cov_raw=_cov_raw,
            thin_surface_mode=thin_surface_mode,
            density_delta=_density_delta,
            quaternions=_quaternions,
            texel_sites_2d=_texel_sites_2d,
            texel_heights=_texel_heights,
            thin_K=thin_K,
            thin_temp=thin_temp,
            thin_height_eps=thin_height_eps,
            thin_surface_relative_delta=thin_surface_relative_delta,
            thin_surface_delta_max_frac=thin_surface_delta_max_frac,
            raw_plus=_raw_plus,
            raw_minus=_raw_minus,
            thin_surface_independent_mode=thin_surface_independent_mode,
            thin_surface_activation_scale=thin_surface_activation_scale,
        )
        points_grad = results["points_grad"]
        attr_grad = results["attr_grad"]
        density_grad_grad = results.get("density_grad_grad", None)
        density_peak_grad = results.get("density_peak_grad", None)
        delta_raw_grad = results.get("delta_raw_grad", None)
        cov_raw_grad = results.get("cov_raw_grad", None)
        density_delta_grad = results.get("density_delta_grad", None)
        quaternions_grad = results.get("quaternions_grad", None)
        texel_sites_2d_grad = results.get("texel_sites_2d_grad", None)
        texel_heights_grad = results.get("texel_heights_grad", None)
        ctx.errbox.point_error = results.get("point_error", None)

        # Autograd-contract guard: every returned thin-surface grad MUST match
        # the corresponding forward-input parameter shape, else AccumulateGrad
        # cannot accumulate it (a silent or hard failure). This catches any
        # future regression of the C++ grad-tensor allocation shapes.
        if has_thin_surface:
            _expected = {
                "density_delta_grad": (ctx.density_delta, density_delta_grad),
                "quaternions_grad": (ctx.quaternions, quaternions_grad),
                "texel_sites_2d_grad": (ctx.texel_sites_2d, texel_sites_2d_grad),
                "texel_heights_grad": (ctx.texel_heights, texel_heights_grad),
            }
            for _name, (_param, _grad) in _expected.items():
                assert _grad is not None, (
                    f"{_name} is None but thin-surface mode is active "
                    f"(has_thin_surface=True)")
                assert tuple(_grad.shape) == tuple(_param.shape), (
                    f"{_name} shape {tuple(_grad.shape)} != param "
                    f"{tuple(_param.shape)}; autograd cannot accumulate")

        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0
        if density_grad_grad is not None:
            density_grad_grad[~density_grad_grad.isfinite()] = 0
        if density_peak_grad is not None:
            density_peak_grad[~density_peak_grad.isfinite()] = 0
        if delta_raw_grad is not None:
            delta_raw_grad[~delta_raw_grad.isfinite()] = 0
        if cov_raw_grad is not None:
            cov_raw_grad[~cov_raw_grad.isfinite()] = 0
        if density_delta_grad is not None:
            density_delta_grad[~density_delta_grad.isfinite()] = 0
        if quaternions_grad is not None:
            quaternions_grad[~quaternions_grad.isfinite()] = 0
        if texel_sites_2d_grad is not None:
            texel_sites_2d_grad[~texel_sites_2d_grad.isfinite()] = 0
        if texel_heights_grad is not None:
            texel_heights_grad[~texel_heights_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.pipeline,
            ctx.points,
            ctx.density,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
            ctx.has_density_grad,
            ctx.gradient_max_slope,
            ctx.interpolation_mode,
            ctx.idw_sigma,
            ctx.idw_sigma_v,
            ctx.per_cell_sigma,
            ctx.per_neighbor_sigma,
            ctx.cell_radius,
            ctx.gaussian_mode,
            ctx.has_gaussian,
            ctx.thin_surface_mode,
            ctx.has_thin_surface,
            ctx.thin_K,
            ctx.thin_temp,
            ctx.thin_height_eps,
            ctx.thin_surface_relative_delta,
            ctx.thin_surface_delta_max_frac,
            ctx.thin_surface_independent_mode,
            ctx.thin_surface_activation_scale,
            ctx.has_raw_plus,
            ctx.has_raw_minus,
        )
        if has_density_grad:
            del ctx.density_grad
        if has_gaussian:
            del ctx.density_peak, ctx.delta_raw, ctx.cov_raw
        if has_thin_surface:
            del ctx.density_delta, ctx.quaternions, ctx.texel_sites_2d, ctx.texel_heights
        # LC64 plan v3 Commit 2A -- independent-side raw logits saved
        # on ctx during forward; release here so the autograd context
        # does not hold Parameter references past the backward call.
        if getattr(ctx, "thin_surface_independent_mode", False):
            del ctx.raw_plus, ctx.raw_minus

        return (
            None,               # pipeline
            points_grad,        # _points
            attr_grad,          # _density
            None,               # _point_adjacency
            None,               # _point_adjacency_offsets
            None,               # rays
            None,               # start_point
            None,               # return_contribution
            density_grad_grad,  # _density_grad
            None,               # _gradient_max_slope
            None,               # _interpolation_mode
            None,               # _idw_sigma
            None,               # _idw_sigma_v
            None,               # _per_cell_sigma
            None,               # _per_neighbor_sigma
            None,               # _cell_radius
            None,               # _gaussian_mode
            density_peak_grad,  # _density_peak
            delta_raw_grad,     # _delta_raw
            cov_raw_grad,       # _cov_raw
            None,               # _thin_surface_mode
            density_delta_grad, # _density_delta
            quaternions_grad,   # _quaternions
            texel_sites_2d_grad, # _texel_sites_2d
            texel_heights_grad, # _texel_heights
            None,               # _thin_K
            None,               # _thin_temp
            None,               # _thin_height_eps
            None,               # _thin_surface_relative_delta (M5)
            None,               # _thin_surface_delta_max_frac (M5)
            None,               # _raw_plus (Commit 2A)
            None,               # _raw_minus (Commit 2A)
            None,               # _thin_surface_independent_mode (Commit 2A)
            None,               # _thin_surface_activation_scale (Commit 2A)
        )
