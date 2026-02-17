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

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float mu = density[point_idx];
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

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
                            const uint32_t *__restrict__ point_adjacency,
                            const uint32_t *__restrict__ point_adjacency_offsets,
                            const Vec4h *__restrict__ adjacent_diff,
                            const Ray *__restrict__ rays,
                            uint32_t num_rays,
                            const uint32_t *__restrict__ start_point_index,
                            const float *__restrict__ ray_projection_grad,
                            const float *__restrict__ ray_error,
                            Vec3f *__restrict__ points_grad,
                            float *__restrict__ density_grad,
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

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float mu = density[point_idx];
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        if (point_error) {
            float weight = delta_t;
            atomicAdd(point_error + point_idx, weight * error);
        }

        // dL/dmu_i = dL/dprojection * delta_t
        float dL_dmu = dL_dprojection * delta_t;
        atomicAdd(density_grad + point_idx, dL_dmu);

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

class CUDADensityPipeline : public Pipeline {
  public:
    CUDADensityPipeline() = default;

    virtual ~CUDADensityPipeline() {}

    void trace_forward(const TraceSettings &settings,
                       uint32_t num_points,
                       const Vec3f *points,
                       const float *density,
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
                        uint32_t point_adjacency_size,
                        const uint32_t *point_adjacency,
                        const uint32_t *point_adjacency_offsets,
                        uint32_t num_rays,
                        const Ray *rays,
                        const uint32_t *start_point_index,
                        const float *ray_projection_grad,
                        const float *ray_error,
                        Vec3f *points_grad,
                        float *density_grad,
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
            point_adjacency,
            point_adjacency_offsets,
            adjacent_diff.begin(),
            rays,
            num_rays,
            start_point_index,
            ray_projection_grad,
            ray_error,
            points_grad,
            density_grad,
            point_error);
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
