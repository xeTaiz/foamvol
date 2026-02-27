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
        if ctx.has_density_grad:
            ctx.density_grad = _density_grad

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
        )

        errbox = ErrorBox()
        ctx.errbox = errbox

        return (
            results["projection"],
            results.get("contribution", None),
            results["num_intersections"],
            errbox,
        )

    @staticmethod
    def backward(
        ctx,
        grad_projection,
        grad_contribution,
        grad_num_intersections,
        errbox_grad,
    ):
        del grad_contribution
        del grad_num_intersections
        del errbox_grad

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
        )
        points_grad = results["points_grad"]
        attr_grad = results["attr_grad"]
        density_grad_grad = results.get("density_grad_grad", None)
        ctx.errbox.point_error = results.get("point_error", None)

        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0
        if density_grad_grad is not None:
            density_grad_grad[~density_grad_grad.isfinite()] = 0

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
        )
        if has_density_grad:
            del ctx.density_grad

        return (
            None,  # pipeline
            points_grad,  # _points
            attr_grad,  # _density
            None,  # _point_adjacency
            None,  # _point_adjacency_offsets
            None,  # rays
            None,  # start_point
            None,  # return_contribution
            density_grad_grad,  # _density_grad
            None,  # _gradient_max_slope
            None,  # _interpolation_mode
            None,  # _idw_sigma
            None,  # _idw_sigma_v
        )
