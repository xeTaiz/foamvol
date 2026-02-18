#include "pipeline_bindings.h"

#include "tracing/pipeline.h"
#include "viewer/viewer.h"

namespace radfoam_bindings {

void validate_scene_data(const Pipeline &pipeline,
                         torch::Tensor points,
                         torch::Tensor attributes,
                         torch::Tensor point_adjacency,
                         torch::Tensor point_adjacency_offsets) {

    if (points.size(-1) != 3) {
        throw std::runtime_error("points had dimension " +
                                 std::to_string(points.size(-1)) +
                                 " along axis -1, expected 3");
    }
    if (dtype_to_scalar_type(points.scalar_type()) != ScalarType::Float32) {
        throw std::runtime_error(
            "points had dtype " +
            std::string(c10::toString(points.scalar_type())) + ", expected " +
            std::string(scalar_to_string(ScalarType::Float32)));
    }
    if (points.device().type() != at::kCUDA) {
        throw std::runtime_error("points must be on CUDA device");
    }
    uint32_t num_points = points.numel() / 3;

    if (attributes.size(-1) != pipeline.attribute_dim()) {
        throw std::runtime_error("attributes had dimension " +
                                 std::to_string(attributes.size(-1)) +
                                 " along axis -1, expected " +
                                 std::to_string(pipeline.attribute_dim()));
    }
    if (attributes.numel() / pipeline.attribute_dim() != num_points) {
        throw std::runtime_error("attributes must have the same number of "
                                 "rows as points");
    }
    if (dtype_to_scalar_type(attributes.scalar_type()) !=
        pipeline.attribute_type()) {
        throw std::runtime_error(
            "attributes had dtype " +
            std::string(c10::toString(attributes.scalar_type())) +
            ", expected " +
            std::string(scalar_to_string(pipeline.attribute_type())));
    }
    if (attributes.device().type() != at::kCUDA) {
        throw std::runtime_error("attributes must be on CUDA device");
    }

    if (point_adjacency_offsets.scalar_type() != at::kUInt32) {
        throw std::runtime_error(
            "point_adjacency_offsets must have uint32 dtype");
    }
    if (point_adjacency_offsets.device().type() != at::kCUDA) {
        throw std::runtime_error(
            "point_adjacency_offsets must be on CUDA device");
    }
    if (point_adjacency_offsets.numel() != num_points + 1) {
        throw std::runtime_error("point_adjacency_offsets must have num_points "
                                 "+ 1 elements");
    }

    if (point_adjacency.scalar_type() != at::kUInt32) {
        throw std::runtime_error("point_adjacency must have uint32 dtype");
    }
    if (point_adjacency.device().type() != at::kCUDA) {
        throw std::runtime_error("point_adjacency must be on CUDA device");
    }
}

void update_scene(Viewer &self,
                  torch::Tensor points_in,
                  torch::Tensor attributes_in,
                  torch::Tensor point_adjacency_in,
                  torch::Tensor point_adjacency_offsets_in,
                  torch::Tensor aabb_tree_in) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor aabb_tree = aabb_tree_in.contiguous();

    validate_scene_data(self.get_pipeline(),
                        points,
                        attributes,
                        point_adjacency,
                        point_adjacency_offsets);

    set_default_stream();

    uint32_t num_points = points.size(0);
    uint32_t num_attrs = attributes.size(0);
    uint32_t num_point_adjacency = point_adjacency.size(0);
    self.update_scene(num_points,
                      num_attrs,
                      num_point_adjacency,
                      points.data_ptr(),
                      attributes.data_ptr(),
                      point_adjacency.data_ptr(),
                      point_adjacency_offsets.data_ptr(),
                      aabb_tree.data_ptr());
}

py::object trace_forward(Pipeline &self,
                         torch::Tensor points_in,
                         torch::Tensor attributes_in,
                         torch::Tensor point_adjacency_in,
                         torch::Tensor point_adjacency_offsets_in,
                         torch::Tensor rays_in,
                         torch::Tensor start_point_in,
                         py::object max_intersections,
                         bool return_contribution) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size "
                                 "as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    TraceSettings settings = default_trace_settings();
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < rays.dim() - 1; i++) {
        output_shape.push_back(rays.size(i));
    }

    // Output: 1 float per ray (projection)
    auto output_projection_shape = output_shape;
    output_projection_shape.push_back(1);
    torch::Tensor output_projection =
        torch::zeros(output_projection_shape,
                     torch::dtype(torch::kFloat32).device(rays.device()));

    auto output_num_intersections_shape = output_shape;
    output_num_intersections_shape.push_back(1);
    torch::Tensor num_intersections =
        torch::empty(output_num_intersections_shape,
                     torch::dtype(scalar_to_type_meta(ScalarType::UInt32))
                         .device(rays.device()));

    torch::Tensor output_contribution;
    if (return_contribution) {
        output_contribution = torch::zeros(
            {num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    set_default_stream();

    self.trace_forward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        reinterpret_cast<const float *>(attributes.data_ptr()),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const radfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        reinterpret_cast<float *>(output_projection.data_ptr()),
        reinterpret_cast<uint32_t *>(num_intersections.data_ptr()),
        return_contribution
            ? reinterpret_cast<float *>(output_contribution.data_ptr())
            : nullptr);

    py::dict output_dict;

    output_dict["projection"] = output_projection;
    if (return_contribution) {
        output_dict["contribution"] = output_contribution;
    }
    output_dict["num_intersections"] = num_intersections;

    return output_dict;
}

py::object trace_backward(Pipeline &self,
                          torch::Tensor points_in,
                          torch::Tensor attributes_in,
                          torch::Tensor point_adjacency_in,
                          torch::Tensor point_adjacency_offsets_in,
                          torch::Tensor rays_in,
                          torch::Tensor start_point_in,
                          torch::Tensor grad_in,
                          std::optional<torch::Tensor> ray_error_in,
                          py::object max_intersections) {
    torch::Tensor points = points_in.contiguous();
    torch::Tensor attributes = attributes_in.contiguous();
    torch::Tensor point_adjacency = point_adjacency_in.contiguous();
    torch::Tensor point_adjacency_offsets =
        point_adjacency_offsets_in.contiguous();
    torch::Tensor rays = rays_in.contiguous();
    torch::Tensor start_point = start_point_in.contiguous();

    validate_scene_data(self,
                        points_in,
                        attributes_in,
                        point_adjacency_in,
                        point_adjacency_offsets_in);

    bool return_error = ray_error_in.has_value();

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;

    if (rays.size(-1) != 6) {
        throw std::runtime_error("rays must have 6 as the last dimension");
    }
    if (rays.scalar_type() != at::kFloat) {
        throw std::runtime_error("rays must have float32 dtype");
    }
    if (rays.device().type() != at::kCUDA) {
        throw std::runtime_error("rays must be on CUDA device");
    }

    if (start_point.numel() != num_rays) {
        throw std::runtime_error("start_point must have the same batch size "
                                 "as rays");
    }
    if (start_point.scalar_type() != at::kUInt32) {
        throw std::runtime_error("start_point must have uint32 dtype");
    }
    if (start_point.device().type() != at::kCUDA) {
        throw std::runtime_error("start_point must be on CUDA device");
    }

    torch::Tensor grad_in_c = grad_in.contiguous();
    if (grad_in_c.size(-1) != 1) {
        throw std::runtime_error("grad_in must have 1 as the last dimension");
    }
    if (grad_in_c.scalar_type() != at::kFloat) {
        throw std::runtime_error("grad_in must have float32 dtype");
    }
    if (grad_in_c.device().type() != at::kCUDA) {
        throw std::runtime_error("grad_in must be on CUDA device");
    }
    if (grad_in_c.numel() != num_rays) {
        throw std::runtime_error("grad_in must have the same batch size "
                                 "as rays");
    }

    torch::Tensor ray_error;
    torch::Tensor point_error;
    if (return_error) {
        ray_error = ray_error_in.value().contiguous();

        if (ray_error.scalar_type() != at::kFloat) {
            throw std::runtime_error("ray_error must have float32 dtype");
        }
        if (ray_error.device().type() != at::kCUDA) {
            throw std::runtime_error("ray_error must be on CUDA device");
        }
        if (ray_error.numel() != num_rays) {
            throw std::runtime_error("ray_error must have the same batch size "
                                     "as rays");
        }

        point_error = torch::zeros(
            {num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    TraceSettings settings = default_trace_settings();
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }

    int64_t num_attr = attributes.size(0);

    std::vector<int64_t> attr_grad_shape = {num_attr, (int64_t)self.attribute_dim()};

    torch::Tensor attr_grad =
        torch::zeros(attr_grad_shape,
                     torch::dtype(torch::kFloat32).device(rays.device()));

    std::vector<int64_t> points_grad_shape = {(int64_t)num_points, 3};

    torch::Tensor points_grad = torch::zeros(
        points_grad_shape, torch::dtype(rays.dtype()).device(rays.device()));

    set_default_stream();

    self.trace_backward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        reinterpret_cast<const float *>(attributes.data_ptr()),
        point_adjacency_size,
        reinterpret_cast<const uint32_t *>(point_adjacency.data_ptr()),
        reinterpret_cast<const uint32_t *>(point_adjacency_offsets.data_ptr()),
        num_rays,
        reinterpret_cast<const radfoam::Ray *>(rays.data_ptr()),
        reinterpret_cast<const uint32_t *>(start_point.data_ptr()),
        reinterpret_cast<const float *>(grad_in_c.data_ptr()),
        return_error ? reinterpret_cast<const float *>(ray_error.data_ptr())
                     : nullptr,
        reinterpret_cast<radfoam::Vec3f *>(points_grad.data_ptr()),
        reinterpret_cast<float *>(attr_grad.data_ptr()),
        return_error ? reinterpret_cast<float *>(point_error.data_ptr())
                     : nullptr);

    py::dict output_dict;

    output_dict["points_grad"] = points_grad;
    output_dict["attr_grad"] = attr_grad;
    if (return_error) {
        output_dict["point_error"] = point_error;
    }

    return output_dict;
}

std::shared_ptr<Pipeline> create_ct_pipeline_binding() {
    return create_ct_pipeline();
}

void run_with_viewer(std::shared_ptr<Pipeline> pipeline,
                     std::function<void(std::shared_ptr<Viewer>)> callback,
                     std::optional<int> total_iterations,
                     std::optional<torch::Tensor> camera_pos,
                     std::optional<torch::Tensor> camera_forward,
                     std::optional<torch::Tensor> camera_up,
                     std::optional<torch::Tensor> orbit_target) {
    py::gil_scoped_release release;

    ViewerOptions options = default_viewer_options();
    if (total_iterations.has_value()) {
        options.total_iterations = total_iterations.value();
    }
    if (camera_pos.has_value()) {
        torch::Tensor camera_pos_cpu =
            camera_pos->contiguous().cpu().to(torch::kFloat);
        options.camera_pos = radfoam::Vec3f(camera_pos_cpu.data_ptr<float>());
    }
    if (camera_forward.has_value()) {
        torch::Tensor camera_forward_cpu =
            camera_forward->contiguous().cpu().to(torch::kFloat);
        options.camera_forward =
            radfoam::Vec3f(camera_forward_cpu.data_ptr<float>());
    }
    if (camera_up.has_value()) {
        torch::Tensor camera_up_cpu =
            camera_up->contiguous().cpu().to(torch::kFloat);
        options.camera_up = radfoam::Vec3f(camera_up_cpu.data_ptr<float>());
    }
    if (orbit_target.has_value()) {
        torch::Tensor orbit_target_cpu =
            orbit_target->contiguous().cpu().to(torch::kFloat);
        options.orbit_target = radfoam::Vec3f(orbit_target_cpu.data_ptr<float>());
    }

    set_default_stream();

    run_with_viewer(std::move(pipeline), std::move(callback), options);
}

void init_pipeline_bindings(py::module &module) {
    py::class_<Pipeline, std::shared_ptr<Pipeline>>(module, "Pipeline")
        .def("trace_forward",
             trace_forward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("max_intersections") = py::none(),
             py::arg("return_contribution") = false)
        .def("trace_backward",
             trace_backward,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("rays"),
             py::arg("start_point"),
             py::arg("grad_in"),
             py::arg("ray_error") = py::none(),
             py::arg("max_intersections") = py::none());

    module.def("create_ct_pipeline", create_ct_pipeline_binding);

    py::class_<Viewer, std::shared_ptr<Viewer>>(module, "Viewer")
        .def("update_scene",
             update_scene,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("aabb_tree"))
        .def("step", &Viewer::step)
        .def("is_closed", &Viewer::is_closed);

    module.def("run_with_viewer",
               run_with_viewer,
               py::arg("pipeline"),
               py::arg("callback"),
               py::arg("total_iterations") = py::none(),
               py::arg("camera_pos") = py::none(),
               py::arg("camera_forward") = py::none(),
               py::arg("camera_up") = py::none(),
               py::arg("orbit_target") = py::none());
}

} // namespace radfoam_bindings
