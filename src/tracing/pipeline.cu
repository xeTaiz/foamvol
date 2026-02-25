#include "../aabb_tree/aabb_tree.h"
#include "../delaunay/triangulation_ops.h"
#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "pipeline.h"

#include "../utils/common_kernels.cuh"
#include "tracing_utils.cuh"

namespace radfoam {

template <int block_size>
__global__ void ct_forward(TraceSettings settings,
                           const Vec3f *__restrict__ points,
                           const float *__restrict__ density,
                           const Vec3f *__restrict__ density_grad,
                           const uint32_t *__restrict__ point_adjacency,
                           const uint32_t *__restrict__ point_adjacency_offsets,
                           const Vec4h *__restrict__ adjacent_diff,
                           const Ray *__restrict__ rays,
                           uint32_t num_rays,
                           const uint32_t *__restrict__ start_point_index,
                           float *__restrict__ ray_projection,
                           uint32_t *__restrict__ num_intersections,
                           float *__restrict__ point_contribution) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float projection = 0.0f;

    constexpr float sp_beta = 10.0f;

    float max_slope = settings.gradient_max_slope;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float raw = density[point_idx];
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // softplus activation on raw scalar density
        float mu_base = (sp_beta * raw > 20.0f) ? raw
                        : logf(1.0f + expf(sp_beta * raw)) / sp_beta;

        float mu;
        if (density_grad) {
            float t_mid = (t_0 + t_1) * 0.5f;
            Vec3f x_mid = ray.origin + t_mid * ray.direction;
            Vec3f g = density_grad[point_idx];
            Vec3f slope = max_slope * Vec3f(tanhf(g[0]), tanhf(g[1]), tanhf(g[2]));
            mu = fmaxf(0.0f, mu_base + slope.dot(x_mid - current_point));
        } else {
            mu = mu_base;
        }

        projection += mu * delta_t;

        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, delta_t);
        }

        return true; // no early termination for CT
    };

    uint32_t start_point = start_point_index[thread_idx];

    uint32_t n = trace<block_size, 4>(ray,
                                      points,
                                      point_adjacency,
                                      point_adjacency_offsets,
                                      adjacent_diff,
                                      start_point,
                                      settings.max_intersections,
                                      functor);

    ray_projection[thread_idx] = projection;

    if (num_intersections)
        num_intersections[thread_idx] = n;
}

template <int block_size>
__global__ void ct_backward(TraceSettings settings,
                            const Vec3f *__restrict__ points,
                            const float *__restrict__ density,
                            const Vec3f *__restrict__ density_grad_in,
                            const uint32_t *__restrict__ point_adjacency,
                            const uint32_t *__restrict__ point_adjacency_offsets,
                            const Vec4h *__restrict__ adjacent_diff,
                            const Ray *__restrict__ rays,
                            uint32_t num_rays,
                            const uint32_t *__restrict__ start_point_index,
                            const float *__restrict__ ray_projection_grad,
                            const float *__restrict__ ray_error,
                            Vec3f *__restrict__ points_grad,
                            float *__restrict__ density_scalar_grad,
                            Vec3f *__restrict__ density_grad_grad,
                            float *__restrict__ point_error) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float dL_dprojection = ray_projection_grad[thread_idx];

    float error;
    if (ray_error) {
        error = ray_error[thread_idx];
    }

    uint32_t prev_point_idx = UINT32_MAX;
    Vec3f prev_point = Vec3f::Zero();
    Vec3f prev_point_grad = Vec3f::Zero();

    Vec3f current_point_grad = Vec3f::Zero();
    Vec3f next_point_grad = Vec3f::Zero();

    bool grad_active = (density_grad_in != nullptr);
    constexpr float sp_beta = 10.0f;
    float max_slope = settings.gradient_max_slope;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float raw = density[point_idx];
        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        Vec3f x_mid_offset;

        if (point_error) {
            float weight = delta_t;
            atomicAdd(point_error + point_idx, weight * error);
        }

        // softplus activation on raw scalar density
        float mu_base = (sp_beta * raw > 20.0f) ? raw
                        : logf(1.0f + expf(sp_beta * raw)) / sp_beta;

        float mu;
        if (grad_active) {
            float t_mid = (t_0 + t_1) * 0.5f;
            Vec3f x_mid = ray.origin + t_mid * ray.direction;
            x_mid_offset = x_mid - current_point;
            Vec3f g = density_grad_in[point_idx];
            Vec3f slope = max_slope * Vec3f(tanhf(g[0]), tanhf(g[1]), tanhf(g[2]));
            mu = fmaxf(0.0f, mu_base + slope.dot(x_mid_offset));
        } else {
            mu = mu_base;
        }

        // indicator for ReLU clamp
        float indicator = (mu > 0.0f) ? 1.0f : 0.0f;
        float dL_dmu = dL_dprojection * delta_t * indicator;

        // dL/d(raw) through softplus
        float d_softplus = 1.0f / (1.0f + expf(-sp_beta * raw));
        atomicAdd(density_scalar_grad + point_idx, dL_dmu * d_softplus);

        // dL/d(g_param) — only when gradients active
        if (grad_active && density_grad_grad) {
            Vec3f g = density_grad_in[point_idx];
            Vec3f sech2(1.0f - tanhf(g[0]) * tanhf(g[0]),
                        1.0f - tanhf(g[1]) * tanhf(g[1]),
                        1.0f - tanhf(g[2]) * tanhf(g[2]));
            Vec3f dL_dg = dL_dmu * max_slope * sech2.cwiseProduct(x_mid_offset);
            atomic_add_vec(density_grad_grad + point_idx, dL_dg);
        }

        // dL/d(delta_t) = dL/dprojection * mu
        float dL_ddelta_t = dL_dprojection * mu;

        float dL_dt0 = -dL_ddelta_t;
        float dL_dt1 = dL_ddelta_t;

        Vec3f dt0_dprev_point;
        if (prev_point_idx != UINT32_MAX) {
            dt0_dprev_point =
                cell_intersection_grad(prev_point, current_point, ray);
        } else {
            dt0_dprev_point = Vec3f::Zero();
        }

        Vec3f dt1_dcurrent_point =
            cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcurrent_point =
            cell_intersection_grad(current_point, prev_point, ray);

        Vec3f dt1_dnext_point =
            cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad += dL_dt0 * dt0_dprev_point;
        current_point_grad +=
            dL_dt0 * dt0_dcurrent_point + dL_dt1 * dt1_dcurrent_point;
        next_point_grad += dL_dt1 * dt1_dnext_point;

        if (prev_point_idx != UINT32_MAX) {
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        }
        prev_point = current_point;
        prev_point_idx = point_idx;
        prev_point_grad = current_point_grad;

        current_point_grad = next_point_grad;
        next_point_grad = Vec3f::Zero();

        return true; // no early termination for CT
    };

    uint32_t start_point = start_point_index[thread_idx];

    trace<block_size, 2>(ray,
                         points,
                         point_adjacency,
                         point_adjacency_offsets,
                         adjacent_diff,
                         start_point,
                         settings.max_intersections,
                         functor);
}

__global__ void prefetch_adjacent_diff_kernel(
    const Vec3f *__restrict__ points,
    uint32_t num_points,
    uint32_t point_adjacency_size,
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    Vec4h *__restrict__ adjacent_diff) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;

    Vec3f p = points[i];
    uint32_t offset_start = point_adjacency_offsets[i];
    uint32_t offset_end = point_adjacency_offsets[i + 1];
    uint32_t num_adjacent = offset_end - offset_start;

    for (uint32_t j = 0; j < num_adjacent; ++j) {
        uint32_t adjacent_idx = point_adjacency[offset_start + j];
        Vec3f q = points[adjacent_idx];
        Vec3f diff = q - p;
        adjacent_diff[offset_start + j] = Vec4h(diff[0], diff[1], diff[2], 0);
    }
}

void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            Vec4h *adjacent_diff,
                            const void *stream) {
    launch_kernel_1d<256>(prefetch_adjacent_diff_kernel,
                          num_points,
                          stream,
                          points,
                          num_points,
                          point_adjacency_size,
                          point_adjacency,
                          point_adjacency_offsets,
                          adjacent_diff);
}

__global__ void ct_visualization(TraceSettings settings,
                                  VisualizationSettings vis_settings,
                                  Camera camera,
                                  CMapTable cmap_table,
                                  TransferFunctionTable tf_table,
                                  const Vec3f *__restrict__ points,
                                  const float *__restrict__ density,
                                  const uint32_t *__restrict__ point_adjacency,
                                  const uint32_t *__restrict__ point_adjacency_offsets,
                                  const Vec4h *__restrict__ adjacent_diff,
                                  uint32_t start_index,
                                  CUsurfObject output_surface) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= camera.width || j >= camera.height)
        return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction /= ray.direction.norm();

    float beta = vis_settings.activation_beta;
    float act_scale = vis_settings.activation_scale;
    float den_scale = vis_settings.density_scale;
    ColorMap cmap = vis_settings.color_map;
    float depth_quantile = vis_settings.depth_quantile;

    Vec3f color = Vec3f::Zero();
    float transmittance = 1.0f;
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    bool use_tf = vis_settings.use_transfer_function;
    float tf_density_min = vis_settings.tf_density_min;
    float tf_density_max = vis_settings.tf_density_max;
    float tf_opacity_scale = vis_settings.tf_opacity_scale;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float raw = density[point_idx];
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // Softplus activation with numerical stability
        float mu;
        if (beta * raw > 20.0f) {
            mu = act_scale * raw;
        } else {
            mu = act_scale * logf(1.0f + expf(beta * raw)) / beta;
        }

        Vec3f rgb;
        float alpha;

        if (use_tf) {
            // Transfer function path
            float range = tf_density_max - tf_density_min;
            float v = (range > 1e-8f)
                ? fmaxf(0.0f, fminf((mu - tf_density_min) / range, 1.0f))
                : 0.0f;
            float tf_opacity;
            sample_transfer_function(v, tf_table, rgb, tf_opacity);
            alpha = 1.0f - expf(-tf_opacity * tf_opacity_scale * delta_t);
        } else {
            // Original colormap path
            float v = fminf(mu * den_scale, 1.0f);
            rgb = colormap(v, cmap, cmap_table);
            alpha = 1.0f - expf(-mu * delta_t);
        }

        float next_transmittance = transmittance * (1.0f - alpha);

        // Depth: find where transmittance crosses the quantile threshold
        if (!depth_quantile_passed && next_transmittance < depth_quantile) {
            depth_quantile_passed = true;
            if (mu > 1e-6f) {
                depth = t_0 + logf(transmittance / depth_quantile) / mu;
            } else {
                depth = t_0;
            }
        }

        color += transmittance * alpha * rgb;
        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold;
    };

    uint32_t n = trace<128, 4>(ray,
                               points,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff,
                               start_index,
                               settings.max_intersections,
                               functor);

    // Output based on visualization mode
    Vec3f out;
    switch (vis_settings.mode) {
    case VolumeDensity:
    case RGB: {
        Vec3f bg = *vis_settings.bg_color;
        if (vis_settings.checker_bg) {
            int ci = i / 16;
            int cj = j / 16;
            if ((ci + cj) % 2 == 0) {
                bg = Vec3f(0.8f, 0.8f, 0.8f);
            } else {
                bg = Vec3f(0.6f, 0.6f, 0.6f);
            }
        }
        out = color + transmittance * bg;
        break;
    }
    case Depth: {
        float val = depth / vis_settings.max_depth;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        out = colormap(val, cmap, cmap_table);
        break;
    }
    case Alpha: {
        float opacity = 1.0f - transmittance;
        out = Vec3f(opacity, opacity, opacity);
        break;
    }
    case Intersections: {
        float val = (n > 1) ? float(n - 1) / float(settings.max_intersections) : 0.0f;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        out = colormap(val, cmap, cmap_table);
        break;
    }
    default:
        out = Vec3f::Zero();
        break;
    }

    uint32_t rgba = make_rgba8(out[0], out[1], out[2], 1.0f);
    surf2Dwrite(rgba, output_surface, i * 4, j);
}

class CUDADensityPipeline : public Pipeline {
  public:
    CUDADensityPipeline() = default;

    virtual ~CUDADensityPipeline() {}

    void trace_forward(const TraceSettings &settings,
                       uint32_t num_points,
                       const Vec3f *points,
                       const float *density,
                       const Vec3f *density_grad,
                       uint32_t point_adjacency_size,
                       const uint32_t *point_adjacency,
                       const uint32_t *point_adjacency_offsets,
                       uint32_t num_rays,
                       const Ray *rays,
                       const uint32_t *start_point_index,
                       float *ray_projection,
                       uint32_t *num_intersections,
                       float *point_contribution) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            ct_forward<block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            density,
            density_grad,
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            ray_projection,
            num_intersections,
            point_contribution);
    }

    void trace_backward(const TraceSettings &settings,
                        uint32_t num_points,
                        const Vec3f *points,
                        const float *density,
                        const Vec3f *density_grad,
                        uint32_t point_adjacency_size,
                        const uint32_t *point_adjacency,
                        const uint32_t *point_adjacency_offsets,
                        uint32_t num_rays,
                        const Ray *rays,
                        const uint32_t *start_point_index,
                        const float *ray_projection_grad,
                        const float *ray_error,
                        Vec3f *points_grad,
                        float *density_scalar_grad,
                        Vec3f *density_grad_grad,
                        float *point_error) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        launch_kernel_1d<block_size>(
            ct_backward<block_size>,
            num_rays,
            nullptr,
            settings,
            points,
            density,
            density_grad,
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            ray_projection_grad,
            ray_error,
            points_grad,
            density_scalar_grad,
            density_grad_grad,
            point_error);
    }

    void trace_visualization(const TraceSettings &settings,
                             const VisualizationSettings &vis_settings,
                             const Camera &camera,
                             CMapTable cmap_table,
                             TransferFunctionTable tf_table,
                             uint32_t num_points,
                             uint32_t num_tets,
                             const void *points,
                             const void *attributes,
                             const void *point_adjacency,
                             const void *point_adjacency_offsets,
                             const void *adjacent_points,
                             uint32_t start_index,
                             uint64_t output_surface,
                             const void *stream = nullptr) override {

        dim3 block(16, 16);
        dim3 grid((camera.width + block.x - 1) / block.x,
                  (camera.height + block.y - 1) / block.y);

        CUstream cu_stream = 0;
        if (stream) {
            cu_stream = *reinterpret_cast<const CUstream *>(stream);
        }

        ct_visualization<<<grid, block, 0, cu_stream>>>(
            settings,
            vis_settings,
            camera,
            cmap_table,
            tf_table,
            reinterpret_cast<const Vec3f *>(points),
            reinterpret_cast<const float *>(attributes),
            reinterpret_cast<const uint32_t *>(point_adjacency),
            reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
            reinterpret_cast<const Vec4h *>(adjacent_points),
            start_index,
            static_cast<CUsurfObject>(output_surface));
    }

    uint32_t attribute_dim() const override {
        return 1;
    }

    ScalarType attribute_type() const override {
        return scalar_code<float>();
    }
};

std::shared_ptr<Pipeline> create_ct_pipeline() {
    return std::make_shared<CUDADensityPipeline>();
}

} // namespace radfoam
