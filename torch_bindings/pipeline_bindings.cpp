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

void update_tet_topology(Viewer &self,
                         torch::Tensor tets_in,
                         torch::Tensor tet_adjacency_in,
                         torch::Tensor permutation_in,
                         torch::Tensor inv_perm_in,
                         torch::Tensor vert_to_tet_in) {
    torch::Tensor tets = tets_in.contiguous();
    torch::Tensor tet_adj = tet_adjacency_in.contiguous();
    torch::Tensor perm = permutation_in.contiguous();
    torch::Tensor inv_perm = inv_perm_in.contiguous();
    torch::Tensor vert_to_tet = vert_to_tet_in.contiguous();

    if (tets.device().type() != at::kCUDA)
        throw std::runtime_error("tets must be on a CUDA device");
    if (tet_adj.device().type() != at::kCUDA)
        throw std::runtime_error("tet_adjacency must be on a CUDA device");
    if (perm.device().type() != at::kCUDA)
        throw std::runtime_error("permutation must be on a CUDA device");
    if (inv_perm.device().type() != at::kCPU)
        throw std::runtime_error("inv_perm must be on CPU");
    if (vert_to_tet.device().type() != at::kCPU)
        throw std::runtime_error("vert_to_tet must be on CPU");

    uint32_t num_tets = tets.size(0);
    self.update_tet_topology(num_tets,
                             tets.data_ptr(),
                             tet_adj.data_ptr(),
                             perm.data_ptr(),
                             reinterpret_cast<const uint32_t *>(inv_perm.data_ptr()),
                             reinterpret_cast<const uint32_t *>(vert_to_tet.data_ptr()));
}

void update_volume(Viewer &self, torch::Tensor volume_in) {
    if (volume_in.dim() != 3)
        throw std::runtime_error("volume must be a 3D tensor (X, Y, Z)");
    torch::Tensor volume = volume_in.contiguous().to(torch::kFloat32);
    if (volume.device().type() != at::kCUDA)
        throw std::runtime_error("volume must be on a CUDA device");
    self.update_volume((uint32_t)volume.size(0),
                       (uint32_t)volume.size(1),
                       (uint32_t)volume.size(2),
                       volume.data_ptr());
}

py::object trace_forward(Pipeline &self,
                         torch::Tensor points_in,
                         torch::Tensor attributes_in,
                         torch::Tensor point_adjacency_in,
                         torch::Tensor point_adjacency_offsets_in,
                         torch::Tensor rays_in,
                         torch::Tensor start_point_in,
                         py::object max_intersections,
                         bool return_contribution,
                         std::optional<torch::Tensor> density_grad_in,
                         float gradient_max_slope,
                         bool interpolation_mode,
                         float idw_sigma,
                         float idw_sigma_v,
                         bool per_cell_sigma,
                         bool per_neighbor_sigma,
                         std::optional<torch::Tensor> cell_radius_in,
                         bool gaussian_mode,
                         std::optional<torch::Tensor> density_peak_in,
                         std::optional<torch::Tensor> delta_raw_in,
                         std::optional<torch::Tensor> cov_raw_in,
                         bool thin_surface_mode,
                         std::optional<torch::Tensor> density_delta_in,
                         std::optional<torch::Tensor> quaternions_in,
                         std::optional<torch::Tensor> texel_sites_2d_in,
                         std::optional<torch::Tensor> texel_heights_in,
                         int thin_K,
                         float thin_temp,
                         float thin_height_eps,
                         bool thin_surface_relative_delta,
                         float thin_surface_delta_max_frac,
                         // LC64 plan v3 Commit 2A -- independent-side raw
                         // logits (each (N,1)).  Only read when
                         // thin_surface_independent_mode is true; nullptr
                         // for legacy / absolute / relative.
                         std::optional<torch::Tensor> raw_plus_in,
                         std::optional<torch::Tensor> raw_minus_in,
                         bool thin_surface_independent_mode,
                         float thin_surface_activation_scale) {
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

    bool has_density_grad = density_grad_in.has_value();
    torch::Tensor density_grad;
    if (has_density_grad) {
        density_grad = density_grad_in.value().contiguous();
    }

    bool has_cell_radius = cell_radius_in.has_value();
    torch::Tensor cell_radius;
    if (has_cell_radius) {
        cell_radius = cell_radius_in.value().contiguous();
    }

    bool has_density_peak = density_peak_in.has_value();
    torch::Tensor density_peak;
    if (has_density_peak) {
        density_peak = density_peak_in.value().contiguous();
    }

    bool has_delta_raw = delta_raw_in.has_value();
    torch::Tensor delta_raw_t;
    if (has_delta_raw) {
        delta_raw_t = delta_raw_in.value().contiguous();
    }

    bool has_cov_raw = cov_raw_in.has_value();
    torch::Tensor cov_raw_t;
    if (has_cov_raw) {
        cov_raw_t = cov_raw_in.value().contiguous();
    }

    bool has_density_delta = density_delta_in.has_value();
    torch::Tensor density_delta_t;
    if (has_density_delta) density_delta_t = density_delta_in.value().contiguous();

    bool has_quaternions = quaternions_in.has_value();
    torch::Tensor quaternions_t;
    if (has_quaternions) quaternions_t = quaternions_in.value().contiguous();

    bool has_texel_sites = texel_sites_2d_in.has_value();
    torch::Tensor texel_sites_t;
    if (has_texel_sites) texel_sites_t = texel_sites_2d_in.value().contiguous();

    bool has_texel_heights = texel_heights_in.has_value();
    torch::Tensor texel_heights_t;
    if (has_texel_heights) texel_heights_t = texel_heights_in.value().contiguous();

    // LC64 plan v3 Commit 2A -- independent-side raw logits.
    bool has_raw_plus = raw_plus_in.has_value();
    bool has_raw_minus = raw_minus_in.has_value();
    torch::Tensor raw_plus_t;
    torch::Tensor raw_minus_t;
    if (has_raw_plus) raw_plus_t = raw_plus_in.value().contiguous();
    if (has_raw_minus) raw_minus_t = raw_minus_in.value().contiguous();

    // Pre-launch validation (mixed/missing) -- per the acceptance contract
    // these must fail before kernel launch, not after the dispatcher has
    // selected the wrong branch.
    if (thin_surface_independent_mode) {
        if (!thin_surface_mode) {
            throw std::runtime_error(
                "thin_surface_independent_mode=True requires "
                "thin_surface_mode=True (independent mode reuses the "
                "thin-surface geometry: quaternion + K texel sites + "
                "heights + cell_radius).");
        }
        if (!has_raw_plus || !has_raw_minus) {
            throw std::runtime_error(
                "thin_surface_independent_mode=True requires both "
                "raw_plus and raw_minus tensors (each (N,1)). Missing: "
                + std::string(has_raw_plus ? "" : "raw_plus ")
                + std::string(has_raw_minus ? "" : "raw_minus"));
        }
        if (raw_plus_t.scalar_type() != at::kFloat ||
            raw_minus_t.scalar_type() != at::kFloat) {
            throw std::runtime_error(
                "raw_plus / raw_minus must have float32 dtype under "
                "thin_surface_independent_mode.");
        }
        if (raw_plus_t.device().type() != at::kCUDA ||
            raw_minus_t.device().type() != at::kCUDA) {
            throw std::runtime_error(
                "raw_plus / raw_minus must be on a CUDA device under "
                "thin_surface_independent_mode.");
        }
        if (raw_plus_t.size(0) != num_points ||
            raw_minus_t.size(0) != num_points) {
            throw std::runtime_error(
                "raw_plus / raw_minus must have num_points rows under "
                "thin_surface_independent_mode (got " +
                std::to_string(raw_plus_t.size(0)) + " vs " +
                std::to_string(num_points) + ").");
        }
        if (has_density_delta) {
            throw std::runtime_error(
                "thin_surface_independent_mode is mutually exclusive with "
                "density_delta (legacy absolute/relative thin-surface path). "
                "Pass density_delta=None under independent mode.");
        }
    } else {
        if (has_raw_plus || has_raw_minus) {
            throw std::runtime_error(
                "raw_plus / raw_minus must be None when "
                "thin_surface_independent_mode is False (legacy / absolute / "
                "relative thin-surface path). Mixed inputs are rejected "
                "before kernel launch.");
        }
    }

    TraceSettings settings = default_trace_settings();
    if (!max_intersections.is_none()) {
        settings.max_intersections = max_intersections.cast<uint32_t>();
    }
    settings.gradient_max_slope = gradient_max_slope;
    settings.interpolation_mode = interpolation_mode;
    settings.idw_sigma = idw_sigma;
    settings.idw_sigma_v = idw_sigma_v;
    settings.per_cell_sigma = per_cell_sigma;
    settings.per_neighbor_sigma = per_neighbor_sigma;
    settings.gaussian_mode = gaussian_mode;
    settings.thin_surface_mode = thin_surface_mode;
    settings.thin_K = thin_K;
    settings.thin_temp = thin_temp;
    settings.thin_height_eps = thin_height_eps;
    // M5 chest rescue: relative-delta parameterization.  When on, the
    // kernel applies delta = rho * mu_bar * tanh(raw_delta) so the split
    // is bounded by rho * mu_bar and both sides stay nonneg for rho in (0,1].
    // Geometry is unaffected.
    settings.thin_surface_relative_delta = thin_surface_relative_delta;
    settings.thin_surface_delta_max_frac  = thin_surface_delta_max_frac;
    // LC64 plan v3 Commit 2A -- forward-only independent-side dispatch.
    settings.thin_surface_independent_mode = thin_surface_independent_mode;
    settings.thin_surface_activation_scale = thin_surface_activation_scale;

    // Hard cap: the CUDA backward kernel uses a fixed-size stack buffer
    // `float w_arr[8]` (pipeline.cu, ct_thinsurface_backward). An invalid K
    // would overflow it or loop past the texel tensor stride. Reject before
    // any kernel launch. Mirrors the Python guard assert_supported_thin_K.
    if (thin_surface_mode && (thin_K <= 0 || thin_K > 8)) {
        throw std::runtime_error(
            "thin_surface_mode requires 1 <= thin_K <= 8 (got " +
            std::to_string(thin_K) + "); CUDA backward w_arr[8] hard cap.");
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
    torch::Tensor output_hit_count;
    if (return_contribution) {
        output_contribution = torch::zeros(
            {num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
        output_hit_count = torch::zeros(
            {num_points, 1},
            torch::dtype(scalar_to_type_meta(ScalarType::UInt32)).device(rays.device()));
    }

    set_default_stream();

    self.trace_forward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        reinterpret_cast<const float *>(attributes.data_ptr()),
        has_density_grad
            ? reinterpret_cast<const radfoam::Vec3f *>(density_grad.data_ptr())
            : nullptr,
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
            : nullptr,
        return_contribution
            ? reinterpret_cast<uint32_t *>(output_hit_count.data_ptr())
            : nullptr,
        has_cell_radius
            ? reinterpret_cast<const float *>(cell_radius.data_ptr())
            : nullptr,
        has_density_peak
            ? reinterpret_cast<const float *>(density_peak.data_ptr())
            : nullptr,
        has_delta_raw
            ? reinterpret_cast<const float *>(delta_raw_t.data_ptr())
            : nullptr,
        has_cov_raw
            ? reinterpret_cast<const float *>(cov_raw_t.data_ptr())
            : nullptr,
        has_density_delta
            ? reinterpret_cast<const float *>(density_delta_t.data_ptr())
            : nullptr,
        has_quaternions
            ? reinterpret_cast<const float *>(quaternions_t.data_ptr())
            : nullptr,
        has_texel_sites
            ? reinterpret_cast<const float *>(texel_sites_t.data_ptr())
            : nullptr,
        has_texel_heights
            ? reinterpret_cast<const float *>(texel_heights_t.data_ptr())
            : nullptr,
        thin_surface_independent_mode && has_raw_plus
            ? reinterpret_cast<const float *>(raw_plus_t.data_ptr())
            : nullptr,
        thin_surface_independent_mode && has_raw_minus
            ? reinterpret_cast<const float *>(raw_minus_t.data_ptr())
            : nullptr);

    py::dict output_dict;

    output_dict["projection"] = output_projection;
    if (return_contribution) {
        output_dict["contribution"] = output_contribution;
        output_dict["hit_count"] = output_hit_count;
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
                          py::object max_intersections,
                          std::optional<torch::Tensor> density_grad_in,
                          float gradient_max_slope,
                          bool interpolation_mode,
                          float idw_sigma,
                          float idw_sigma_v,
                          bool per_cell_sigma,
                          bool per_neighbor_sigma,
                          std::optional<torch::Tensor> cell_radius_in,
                          bool gaussian_mode,
                          std::optional<torch::Tensor> density_peak_in,
                          std::optional<torch::Tensor> delta_raw_in,
                          std::optional<torch::Tensor> cov_raw_in,
                          bool thin_surface_mode,
                          std::optional<torch::Tensor> density_delta_in,
                          std::optional<torch::Tensor> quaternions_in,
                          std::optional<torch::Tensor> texel_sites_2d_in,
                          std::optional<torch::Tensor> texel_heights_in,
                          int thin_K,
                          float thin_temp,
                          float thin_height_eps,
                          bool thin_surface_relative_delta,
                          float thin_surface_delta_max_frac,
                          // LC64 plan v3 Commit 2A -- independent-side raw
                          // logits consumed by the CUDA-native forward/backward paths.
                          std::optional<torch::Tensor> raw_plus_in,
                          std::optional<torch::Tensor> raw_minus_in,
                          bool thin_surface_independent_mode,
                          float thin_surface_activation_scale) {
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
    bool has_density_grad = density_grad_in.has_value();

    torch::Tensor density_grad;
    if (has_density_grad) {
        density_grad = density_grad_in.value().contiguous();
    }

    bool has_cell_radius = cell_radius_in.has_value();
    torch::Tensor cell_radius;
    if (has_cell_radius) {
        cell_radius = cell_radius_in.value().contiguous();
    }

    bool has_density_peak = density_peak_in.has_value();
    torch::Tensor density_peak;
    if (has_density_peak) {
        density_peak = density_peak_in.value().contiguous();
    }

    bool has_delta_raw = delta_raw_in.has_value();
    torch::Tensor delta_raw_t;
    if (has_delta_raw) {
        delta_raw_t = delta_raw_in.value().contiguous();
    }

    bool has_cov_raw = cov_raw_in.has_value();
    torch::Tensor cov_raw_t;
    if (has_cov_raw) {
        cov_raw_t = cov_raw_in.value().contiguous();
    }

    uint32_t num_points = points.size(0);
    uint32_t point_adjacency_size = point_adjacency.size(0);
    uint32_t num_rays = rays.numel() / 6;

    // LC64 plan v3 Commit 2B -- independent-side raw logits are read
    // by the new ct_independent_backward kernel under the same
    // discriminator as the forward.  Validate before any kernel launch
    // so mixed / missing inputs are caught at the binding, mirroring
    // the trace_forward contract.
    bool has_raw_plus_b  = raw_plus_in.has_value();
    bool has_raw_minus_b = raw_minus_in.has_value();
    torch::Tensor raw_plus_t_b;
    torch::Tensor raw_minus_t_b;
    if (has_raw_plus_b)  raw_plus_t_b  = raw_plus_in.value().contiguous();
    if (has_raw_minus_b) raw_minus_t_b = raw_minus_in.value().contiguous();
    if (thin_surface_independent_mode) {
        if (!thin_surface_mode) {
            throw std::runtime_error(
                "trace_backward: thin_surface_independent_mode=True "
                "requires thin_surface_mode=True.");
        }
        if (!has_raw_plus_b || !has_raw_minus_b) {
            throw std::runtime_error(
                "trace_backward: thin_surface_independent_mode=True "
                "requires both raw_plus and raw_minus tensors (each "
                "(N,1)). Missing: "
                + std::string(has_raw_plus_b ? "" : "raw_plus ")
                + std::string(has_raw_minus_b ? "" : "raw_minus"));
        }
        if (raw_plus_t_b.scalar_type() != at::kFloat ||
            raw_minus_t_b.scalar_type() != at::kFloat) {
            throw std::runtime_error(
                "trace_backward: raw_plus / raw_minus must have "
                "float32 dtype under thin_surface_independent_mode.");
        }
        if (raw_plus_t_b.device().type() != at::kCUDA ||
            raw_minus_t_b.device().type() != at::kCUDA) {
            throw std::runtime_error(
                "trace_backward: raw_plus / raw_minus must be on a "
                "CUDA device under thin_surface_independent_mode.");
        }
        if (raw_plus_t_b.size(0) != num_points ||
            raw_minus_t_b.size(0) != num_points) {
            throw std::runtime_error(
                "trace_backward: raw_plus / raw_minus must have "
                "num_points rows under thin_surface_independent_mode "
                "(got " + std::to_string(raw_plus_t_b.size(0)) +
                " vs " + std::to_string(num_points) + ").");
        }
    } else {
        if (has_raw_plus_b || has_raw_minus_b) {
            throw std::runtime_error(
                "trace_backward: raw_plus / raw_minus must be None "
                "when thin_surface_independent_mode is False (legacy "
                "/ absolute / relative thin-surface path). Mixed "
                "inputs are rejected before kernel launch.");
        }
    }

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
    settings.gradient_max_slope = gradient_max_slope;
    settings.interpolation_mode = interpolation_mode;
    settings.idw_sigma = idw_sigma;
    settings.idw_sigma_v = idw_sigma_v;
    settings.per_cell_sigma = per_cell_sigma;
    settings.per_neighbor_sigma = per_neighbor_sigma;
    settings.gaussian_mode = gaussian_mode;
    settings.thin_surface_mode = thin_surface_mode;
    settings.thin_K = thin_K;
    settings.thin_temp = thin_temp;
    settings.thin_height_eps = thin_height_eps;
    // Mirror the forward branch (see trace_forward).
    settings.thin_surface_relative_delta = thin_surface_relative_delta;
    settings.thin_surface_delta_max_frac  = thin_surface_delta_max_frac;
    settings.thin_surface_independent_mode = thin_surface_independent_mode;
    settings.thin_surface_activation_scale = thin_surface_activation_scale;

    // Hard cap (same as trace_forward): protect the CUDA backward w_arr[8].
    if (thin_surface_mode && (thin_K <= 0 || thin_K > 8)) {
        throw std::runtime_error(
            "thin_surface_mode requires 1 <= thin_K <= 8 (got " +
            std::to_string(thin_K) + "); CUDA backward w_arr[8] hard cap.");
    }

    bool has_density_delta = density_delta_in.has_value();
    torch::Tensor density_delta_t;
    if (has_density_delta) density_delta_t = density_delta_in.value().contiguous();

    bool has_quaternions = quaternions_in.has_value();
    torch::Tensor quaternions_t;
    if (has_quaternions) quaternions_t = quaternions_in.value().contiguous();

    bool has_texel_sites = texel_sites_2d_in.has_value();
    torch::Tensor texel_sites_t;
    if (has_texel_sites) texel_sites_t = texel_sites_2d_in.value().contiguous();

    bool has_texel_heights = texel_heights_in.has_value();
    torch::Tensor texel_heights_t;
    if (has_texel_heights) texel_heights_t = texel_heights_in.value().contiguous();

    int64_t num_attr = attributes.size(0);

    std::vector<int64_t> attr_grad_shape = {num_attr, (int64_t)self.attribute_dim()};

    torch::Tensor attr_grad =
        torch::zeros(attr_grad_shape,
                     torch::dtype(torch::kFloat32).device(rays.device()));

    std::vector<int64_t> points_grad_shape = {(int64_t)num_points, 3};

    torch::Tensor points_grad = torch::zeros(
        points_grad_shape, torch::dtype(rays.dtype()).device(rays.device()));

    torch::Tensor density_grad_grad;
    if (has_density_grad) {
        density_grad_grad = torch::zeros(
            {(int64_t)num_points, 3},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    // Gaussian gradient tensors
    torch::Tensor density_peak_grad_t, delta_raw_grad_t, cov_raw_grad_t;
    if (gaussian_mode && has_density_peak) {
        density_peak_grad_t = torch::zeros(
            {(int64_t)num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
        delta_raw_grad_t = torch::zeros(
            {(int64_t)num_points, 3},
            torch::dtype(torch::kFloat32).device(rays.device()));
        cov_raw_grad_t = torch::zeros(
            {(int64_t)num_points, 6},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    // Thin-surface gradient tensors. Shapes MUST match the corresponding
    // nn.Parameter shapes in CTScene so torch.autograd can accumulate them:
    //   density_delta (N,1), quaternions (N,4), texel_sites_2d (N,K,2),
    //   texel_heights (N,K). The trailing size-1 dim on density_delta is
    //   layout-transparent to the kernel's flat atomicAdd writes.
    torch::Tensor density_delta_grad_t, quaternions_grad_t,
                  texel_sites_grad_t, texel_heights_grad_t;
    if (thin_surface_mode && has_density_delta) {
        density_delta_grad_t = torch::zeros(
            {(int64_t)num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }
    // Geometry gradients are required by both legacy delta and independent
    // side-density modes. Independent mode intentionally has no density_delta.
    if (thin_surface_mode && (has_density_delta || thin_surface_independent_mode)) {
        quaternions_grad_t = torch::zeros(
            {(int64_t)num_points, 4},
            torch::dtype(torch::kFloat32).device(rays.device()));
        texel_sites_grad_t = torch::zeros(
            {(int64_t)num_points, thin_K, 2},
            torch::dtype(torch::kFloat32).device(rays.device()));
        texel_heights_grad_t = torch::zeros(
            {(int64_t)num_points, thin_K},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    // LC64 plan v3 Commit 2B -- independent-side raw logits get
    // (N, 1) gradient tensors that the kernel atomicAdd's into.  The
    // legacy base density gradient (attr_grad) is left as a zero
    // tensor: under independent mode the optimizer does not step the
    // frozen base density, and the Python TraceRays.backward returns
    // None in the corresponding autograd slot.
    torch::Tensor raw_plus_grad_t, raw_minus_grad_t;
    if (thin_surface_independent_mode) {
        raw_plus_grad_t = torch::zeros(
            {(int64_t)num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
        raw_minus_grad_t = torch::zeros(
            {(int64_t)num_points, 1},
            torch::dtype(torch::kFloat32).device(rays.device()));
    }

    set_default_stream();

    // LC64 plan v3 Commit 2B -- C++ ternary does NOT short-circuit,
    // so we cannot write
    //     (cond) ? tensor.data_ptr() : nullptr
    // for an uninitialized tensor (data_ptr() throws).  Build every
    // pointer into a local via explicit if/else so the read is
    // guarded by the same condition that allocated the tensor.
    const float *p_density_delta_in = nullptr;
    const float *p_quaternions_in = nullptr;
    const float *p_texel_sites_2d_in = nullptr;
    const float *p_texel_heights_in = nullptr;
    float *p_density_delta_grad_out = nullptr;
    float *p_quaternions_grad_out = nullptr;
    float *p_texel_sites_2d_grad_out = nullptr;
    float *p_texel_heights_grad_out = nullptr;
    const float *p_raw_plus_in = nullptr;
    const float *p_raw_minus_in = nullptr;
    float *p_raw_plus_grad_out = nullptr;
    float *p_raw_minus_grad_out = nullptr;
    float *p_density_peak_grad_out = nullptr;
    float *p_delta_raw_grad_out = nullptr;
    float *p_cov_raw_grad_out = nullptr;

    if (thin_surface_mode && has_density_delta) {
        p_density_delta_in = reinterpret_cast<const float *>(density_delta_t.data_ptr());
        p_density_delta_grad_out = reinterpret_cast<float *>(density_delta_grad_t.data_ptr());
    }
    if (thin_surface_mode && has_quaternions) {
        p_quaternions_in = reinterpret_cast<const float *>(quaternions_t.data_ptr());
        p_quaternions_grad_out = reinterpret_cast<float *>(quaternions_grad_t.data_ptr());
    }
    if (thin_surface_mode && has_texel_sites) {
        p_texel_sites_2d_in = reinterpret_cast<const float *>(texel_sites_t.data_ptr());
        p_texel_sites_2d_grad_out = reinterpret_cast<float *>(texel_sites_grad_t.data_ptr());
    }
    if (thin_surface_mode && has_texel_heights) {
        p_texel_heights_in = reinterpret_cast<const float *>(texel_heights_t.data_ptr());
        p_texel_heights_grad_out = reinterpret_cast<float *>(texel_heights_grad_t.data_ptr());
    }
    if (thin_surface_independent_mode && has_raw_plus_b) {
        p_raw_plus_in = reinterpret_cast<const float *>(raw_plus_t_b.data_ptr());
    }
    if (thin_surface_independent_mode && has_raw_minus_b) {
        p_raw_minus_in = reinterpret_cast<const float *>(raw_minus_t_b.data_ptr());
    }
    if (thin_surface_independent_mode) {
        p_raw_plus_grad_out = reinterpret_cast<float *>(raw_plus_grad_t.data_ptr());
        p_raw_minus_grad_out = reinterpret_cast<float *>(raw_minus_grad_t.data_ptr());
    }
    if (gaussian_mode && has_density_peak) {
        p_density_peak_grad_out = reinterpret_cast<float *>(density_peak_grad_t.data_ptr());
    }
    if (gaussian_mode && has_delta_raw) {
        p_delta_raw_grad_out = reinterpret_cast<float *>(delta_raw_grad_t.data_ptr());
    }
    if (gaussian_mode && has_cov_raw) {
        p_cov_raw_grad_out = reinterpret_cast<float *>(cov_raw_grad_t.data_ptr());
    }

    self.trace_backward(
        settings,
        num_points,
        reinterpret_cast<const radfoam::Vec3f *>(points.data_ptr()),
        reinterpret_cast<const float *>(attributes.data_ptr()),
        has_density_grad
            ? reinterpret_cast<const radfoam::Vec3f *>(density_grad.data_ptr())
            : nullptr,
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
        has_density_grad
            ? reinterpret_cast<radfoam::Vec3f *>(density_grad_grad.data_ptr())
            : nullptr,
        return_error ? reinterpret_cast<float *>(point_error.data_ptr())
                     : nullptr,
        has_cell_radius
            ? reinterpret_cast<const float *>(cell_radius.data_ptr())
            : nullptr,
        has_density_peak
            ? reinterpret_cast<const float *>(density_peak.data_ptr())
            : nullptr,
        has_delta_raw
            ? reinterpret_cast<const float *>(delta_raw_t.data_ptr())
            : nullptr,
        has_cov_raw
            ? reinterpret_cast<const float *>(cov_raw_t.data_ptr())
            : nullptr,
        p_density_peak_grad_out,
        p_delta_raw_grad_out,
        p_cov_raw_grad_out,
        p_density_delta_in,
        p_quaternions_in,
        p_texel_sites_2d_in,
        p_texel_heights_in,
        p_density_delta_grad_out,
        p_quaternions_grad_out,
        p_texel_sites_2d_grad_out,
        p_texel_heights_grad_out,
        // LC64 plan v3 Commit 2B -- independent-side raw logits +
        // per-cell raw gradients.  The forward pointers and the
        // output pointers are both non-null only when
        // thin_surface_independent_mode is True; otherwise both are
        // nullptr and the legacy / absolute / relative branch is
        // taken.  Raw gradient tensors are allocated above and
        // atomicAdd'd by ct_independent_backward.
        p_raw_plus_in,
        p_raw_minus_in,
        p_raw_plus_grad_out,
        p_raw_minus_grad_out);

    py::dict output_dict;

    output_dict["points_grad"] = points_grad;
    output_dict["attr_grad"] = attr_grad;
    if (has_density_grad) {
        output_dict["density_grad_grad"] = density_grad_grad;
    }
    if (gaussian_mode && has_density_peak) {
        output_dict["density_peak_grad"] = density_peak_grad_t;
        output_dict["delta_raw_grad"] = delta_raw_grad_t;
        output_dict["cov_raw_grad"] = cov_raw_grad_t;
    }
    if (thin_surface_mode && has_density_delta) {
        output_dict["density_delta_grad"] = density_delta_grad_t;
    }
    if (thin_surface_mode && (has_density_delta || thin_surface_independent_mode)) {
        output_dict["quaternions_grad"] = quaternions_grad_t;
        output_dict["texel_sites_2d_grad"] = texel_sites_grad_t;
        output_dict["texel_heights_grad"] = texel_heights_grad_t;
    }
    if (thin_surface_independent_mode) {
        output_dict["raw_plus_grad"] = raw_plus_grad_t;
        output_dict["raw_minus_grad"] = raw_minus_grad_t;
    }
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
             py::arg("return_contribution") = false,
             py::arg("density_grad") = py::none(),
             py::arg("gradient_max_slope") = 5.0f,
             py::arg("interpolation_mode") = false,
             py::arg("idw_sigma") = 0.01f,
             py::arg("idw_sigma_v") = 0.1f,
             py::arg("per_cell_sigma") = false,
             py::arg("per_neighbor_sigma") = false,
             py::arg("cell_radius") = py::none(),
             py::arg("gaussian_mode") = false,
             py::arg("density_peak") = py::none(),
             py::arg("delta_raw") = py::none(),
             py::arg("cov_raw") = py::none(),
             py::arg("thin_surface_mode") = false,
             py::arg("density_delta") = py::none(),
             py::arg("quaternions") = py::none(),
             py::arg("texel_sites_2d") = py::none(),
             py::arg("texel_heights") = py::none(),
             py::arg("thin_K") = 4,
             py::arg("thin_temp") = 10.0f,
             py::arg("thin_height_eps") = 1e-4f,
             // M5 relative-delta parameterization.  Off by default; opt in.
             py::arg("thin_surface_relative_delta") = false,
             py::arg("thin_surface_delta_max_frac") = 0.5f,
             // LC64 plan v3 Commit 2A -- independent-side raw logits.
             // Off by default; when True, raw_plus / raw_minus must be
             // provided (each (N,1)) and activation_scale multiplies the
             // activated side attenuation.  Forward only; backward
             // raises NotImplementedError until Commit 2B.
             py::arg("raw_plus") = py::none(),
             py::arg("raw_minus") = py::none(),
             py::arg("thin_surface_independent_mode") = false,
             py::arg("thin_surface_activation_scale") = 1.0f)
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
             py::arg("max_intersections") = py::none(),
             py::arg("density_grad") = py::none(),
             py::arg("gradient_max_slope") = 5.0f,
             py::arg("interpolation_mode") = false,
             py::arg("idw_sigma") = 0.01f,
             py::arg("idw_sigma_v") = 0.1f,
             py::arg("per_cell_sigma") = false,
             py::arg("per_neighbor_sigma") = false,
             py::arg("cell_radius") = py::none(),
             py::arg("gaussian_mode") = false,
             py::arg("density_peak") = py::none(),
             py::arg("delta_raw") = py::none(),
             py::arg("cov_raw") = py::none(),
             py::arg("thin_surface_mode") = false,
             py::arg("density_delta") = py::none(),
             py::arg("quaternions") = py::none(),
             py::arg("texel_sites_2d") = py::none(),
             py::arg("texel_heights") = py::none(),
             py::arg("thin_K") = 4,
             py::arg("thin_temp") = 10.0f,
             py::arg("thin_height_eps") = 1e-4f,
             // M5 relative-delta parameterization.  Off by default; opt in.
             py::arg("thin_surface_relative_delta") = false,
             py::arg("thin_surface_delta_max_frac") = 0.5f,
             // LC64 plan v3 Commit 2A -- independent-side raw logits.
             // Backward under this mode raises NotImplementedError until
             // Commit 2B (see trace_forward above for the forward
             // semantics).
             py::arg("raw_plus") = py::none(),
             py::arg("raw_minus") = py::none(),
             py::arg("thin_surface_independent_mode") = false,
             py::arg("thin_surface_activation_scale") = 1.0f);

    module.def("create_ct_pipeline", create_ct_pipeline_binding);

    py::class_<Viewer, std::shared_ptr<Viewer>>(module, "Viewer")
        .def("update_scene",
             update_scene,
             py::arg("points"),
             py::arg("attributes"),
             py::arg("point_adjacency"),
             py::arg("point_adjacency_offsets"),
             py::arg("aabb_tree"))
        .def("update_volume",
             update_volume,
             py::arg("volume"))
        .def("update_tet_topology",
             update_tet_topology,
             py::arg("tets"),
             py::arg("tet_adjacency"),
             py::arg("permutation"),
             py::arg("inv_perm"),
             py::arg("vert_to_tet"))
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
