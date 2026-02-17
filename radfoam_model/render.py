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
    ):
        ctx.rays = rays
        ctx.start_point = start_point
        ctx.pipeline = pipeline
        ctx.points = _points
        ctx.density = _density
        ctx.point_adjacency = _point_adjacency
        ctx.point_adjacency_offsets = _point_adjacency_offsets

        results = pipeline.trace_forward(
            _points,
            _density,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            return_contribution=return_contribution,
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

        results = pipeline.trace_backward(
            _points,
            _density,
            _point_adjacency,
            _point_adjacency_offsets,
            rays,
            start_point,
            grad_projection,
            ctx.errbox.ray_error,
        )
        points_grad = results["points_grad"]
        attr_grad = results["attr_grad"]
        ctx.errbox.point_error = results.get("point_error", None)

        points_grad[~points_grad.isfinite()] = 0
        attr_grad[~attr_grad.isfinite()] = 0

        del (
            ctx.rays,
            ctx.start_point,
            ctx.pipeline,
            ctx.points,
            ctx.density,
            ctx.point_adjacency,
            ctx.point_adjacency_offsets,
        )
        return (
            None,  # pipeline
            points_grad,  # _points
            attr_grad,  # _density
            None,  # _point_adjacency
            None,  # _point_adjacency_offsets
            None,  # rays
            None,  # start_point
            None,  # return_contribution
        )
