#include "../aabb_tree/aabb_tree.h"
#include "../delaunay/triangulation_ops.h"
#include "../utils/cuda_array.h"
#include "../utils/cuda_helpers.h"
#include "../utils/geometry.h"
#include "pipeline.h"

#include <stdexcept>

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
                           float *__restrict__ point_contribution,
                           uint32_t *__restrict__ point_hit_count) {

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
        if (point_hit_count) {
            atomicAdd(point_hit_count + point_idx, 1u);
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

template <int block_size>
__global__ void ct_gaussian_forward(TraceSettings settings,
                                     const Vec3f *__restrict__ points,
                                     const float *__restrict__ density,
                                     const float *__restrict__ density_peak,
                                     const float *__restrict__ delta_raw,
                                     const float *__restrict__ cov_raw,
                                     const float *__restrict__ cell_radius,
                                     const uint32_t *__restrict__ point_adjacency,
                                     const uint32_t *__restrict__ point_adjacency_offsets,
                                     const Vec4h *__restrict__ adjacent_diff,
                                     const Ray *__restrict__ rays,
                                     uint32_t num_rays,
                                     const uint32_t *__restrict__ start_point_index,
                                     float *__restrict__ ray_projection,
                                     uint32_t *__restrict__ num_intersections,
                                     float *__restrict__ point_contribution,
                                     uint32_t *__restrict__ point_hit_count) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float projection = 0.0f;

    constexpr float sp_beta = 10.0f;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // Base density (softplus activation)
        float raw_b = density[point_idx];
        float mu_base = (sp_beta * raw_b > 20.0f) ? raw_b
                        : logf(1.0f + expf(sp_beta * raw_b)) / sp_beta;
        float projection_base = mu_base * delta_t;

        // Gaussian peak
        float raw_p = density_peak[point_idx];
        float mu_peak = (sp_beta * raw_p > 20.0f) ? raw_p
                        : logf(1.0f + expf(sp_beta * raw_p)) / sp_beta;

        // Center offset: c_off = current_point + cell_r * tanh(delta_raw)
        float cell_r = cell_radius[point_idx];
        const float *dr = delta_raw + point_idx * 3;
        Vec3f c_off = current_point + cell_r * Vec3f(tanhf(dr[0]), tanhf(dr[1]), tanhf(dr[2]));
        Vec3f c_vec = ray.origin - c_off;  // vector from Gaussian center to ray origin

        // Cholesky: L (lower triangular, diagonal via softplus)
        const float *Lr = cov_raw + point_idx * 6;
        float L00 = (sp_beta * Lr[0] > 20.0f) ? Lr[0] : logf(1.0f + expf(sp_beta * Lr[0])) / sp_beta;
        float L10 = Lr[1];
        float L11 = (sp_beta * Lr[2] > 20.0f) ? Lr[2] : logf(1.0f + expf(sp_beta * Lr[2])) / sp_beta;
        float L20 = Lr[3];
        float L21 = Lr[4];
        float L22 = (sp_beta * Lr[5] > 20.0f) ? Lr[5] : logf(1.0f + expf(sp_beta * Lr[5])) / sp_beta;

        // Forward substitution: y = L^{-1} d, z = L^{-1} c_vec
        float y0 = ray.direction[0] / L00;
        float y1 = (ray.direction[1] - L10 * y0) / L11;
        float y2 = (ray.direction[2] - L20 * y0 - L21 * y1) / L22;

        float z0 = c_vec[0] / L00;
        float z1 = (c_vec[1] - L10 * z0) / L11;
        float z2 = (c_vec[2] - L20 * z0 - L21 * z1) / L22;

        float A = y0 * y0 + y1 * y1 + y2 * y2;
        float B = 2.0f * (z0 * y0 + z1 * y1 + z2 * y2);
        float C_val = z0 * z0 + z1 * z1 + z2 * z2;

        A = fmaxf(A, 1e-8f);

        float t_peak = -B / (2.0f * A);
        float d_eff_sq = fmaxf(C_val - B * B / (4.0f * A), 0.0f);

        float sqrt_half_A = sqrtf(0.5f * A);
        float arg_hi = (t_1 - t_peak) * sqrt_half_A;
        float arg_lo = (t_0 - t_peak) * sqrt_half_A;
        float erf_hi = erff(arg_hi);
        float erf_lo = erff(arg_lo);
        float erf_diff = erf_hi - erf_lo;

        float envelope = mu_peak * expf(-0.5f * d_eff_sq);
        float scale = sqrtf(M_PIf / (2.0f * A));
        float projection_gauss = envelope * scale * erf_diff;

        projection += projection_base + projection_gauss;

        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, delta_t);
        }
        if (point_hit_count) {
            atomicAdd(point_hit_count + point_idx, 1u);
        }

        return true;
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
__global__ void ct_gaussian_backward(TraceSettings settings,
                                      const Vec3f *__restrict__ points,
                                      const float *__restrict__ density,
                                      const float *__restrict__ density_peak,
                                      const float *__restrict__ delta_raw,
                                      const float *__restrict__ cov_raw,
                                      const float *__restrict__ cell_radius,
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
                                      float *__restrict__ density_peak_grad,
                                      float *__restrict__ delta_raw_grad,
                                      float *__restrict__ cov_raw_grad,
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

    constexpr float sp_beta = 10.0f;
    constexpr float two_over_sqrt_pi = 2.0f / 1.7724538509f; // 2/sqrt(pi)

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        if (point_error) {
            float weight = delta_t;
            atomicAdd(point_error + point_idx, weight * error);
        }

        // ===== Recompute forward quantities =====
        float raw_b = density[point_idx];
        float mu_base = (sp_beta * raw_b > 20.0f) ? raw_b
                        : logf(1.0f + expf(sp_beta * raw_b)) / sp_beta;

        float raw_p = density_peak[point_idx];
        float mu_peak = (sp_beta * raw_p > 20.0f) ? raw_p
                        : logf(1.0f + expf(sp_beta * raw_p)) / sp_beta;

        float cell_r = cell_radius[point_idx];
        const float *dr = delta_raw + point_idx * 3;
        float tanh_dr0 = tanhf(dr[0]), tanh_dr1 = tanhf(dr[1]), tanh_dr2 = tanhf(dr[2]);
        Vec3f c_off = current_point + cell_r * Vec3f(tanh_dr0, tanh_dr1, tanh_dr2);
        Vec3f c_vec = ray.origin - c_off;

        const float *Lr = cov_raw + point_idx * 6;
        float L00 = (sp_beta * Lr[0] > 20.0f) ? Lr[0] : logf(1.0f + expf(sp_beta * Lr[0])) / sp_beta;
        float L10 = Lr[1];
        float L11 = (sp_beta * Lr[2] > 20.0f) ? Lr[2] : logf(1.0f + expf(sp_beta * Lr[2])) / sp_beta;
        float L20 = Lr[3];
        float L21 = Lr[4];
        float L22 = (sp_beta * Lr[5] > 20.0f) ? Lr[5] : logf(1.0f + expf(sp_beta * Lr[5])) / sp_beta;

        float y0 = ray.direction[0] / L00;
        float y1 = (ray.direction[1] - L10 * y0) / L11;
        float y2 = (ray.direction[2] - L20 * y0 - L21 * y1) / L22;

        float z0 = c_vec[0] / L00;
        float z1 = (c_vec[1] - L10 * z0) / L11;
        float z2 = (c_vec[2] - L20 * z0 - L21 * z1) / L22;

        float A = y0 * y0 + y1 * y1 + y2 * y2;
        float B = 2.0f * (z0 * y0 + z1 * y1 + z2 * y2);
        float C_val = z0 * z0 + z1 * z1 + z2 * z2;
        A = fmaxf(A, 1e-8f);

        float t_peak = -B / (2.0f * A);
        float d_eff_sq = fmaxf(C_val - B * B / (4.0f * A), 0.0f);

        float sqrt_half_A = sqrtf(0.5f * A);
        float arg_hi = (t_1 - t_peak) * sqrt_half_A;
        float arg_lo = (t_0 - t_peak) * sqrt_half_A;
        float erf_hi = erff(arg_hi);
        float erf_lo = erff(arg_lo);
        float erf_diff = erf_hi - erf_lo;

        float exp_deff = expf(-0.5f * d_eff_sq);
        float envelope = mu_peak * exp_deff;
        float scale = sqrtf(M_PIf / (2.0f * A));
        float I_gauss = envelope * scale * erf_diff;

        // ===== Base density backward =====
        float d_softplus_base = 1.0f / (1.0f + expf(-sp_beta * raw_b));
        atomicAdd(density_scalar_grad + point_idx, dL_dprojection * delta_t * d_softplus_base);

        // ===== Gaussian backward =====
        // I_gauss = envelope * scale * erf_diff
        float dL_d_envelope = dL_dprojection * scale * erf_diff;
        float dL_d_scale = dL_dprojection * envelope * erf_diff;
        float dL_d_erf_diff = dL_dprojection * envelope * scale;

        // erf_diff = erf(arg_hi) - erf(arg_lo)
        float derf_hi = two_over_sqrt_pi * expf(-arg_hi * arg_hi);
        float derf_lo = two_over_sqrt_pi * expf(-arg_lo * arg_lo);
        float dL_d_arg_hi = dL_d_erf_diff * derf_hi;
        float dL_d_arg_lo = -dL_d_erf_diff * derf_lo;

        // arg = (t - t_peak) * sqrt_half_A
        float dL_d_t_peak = -sqrt_half_A * (dL_d_arg_hi + dL_d_arg_lo);
        float dL_d_sha = dL_d_arg_hi * (t_1 - t_peak) + dL_d_arg_lo * (t_0 - t_peak);

        // scale = sqrt(pi) / (2 * sqrt_half_A)  =>  d(scale)/d(sha) = -scale/sha
        dL_d_sha += dL_d_scale * (-scale / fmaxf(sqrt_half_A, 1e-12f));

        // sqrt_half_A = sqrt(A/2)  =>  d(sha)/dA = 1/(4*sha)
        float dL_dA = dL_d_sha / fmaxf(4.0f * sqrt_half_A, 1e-12f);

        // t_peak = -B/(2A)
        float inv2A = 1.0f / (2.0f * A);
        float dL_dB = dL_d_t_peak * (-inv2A);
        dL_dA += dL_d_t_peak * B * inv2A / A;  // B/(2A^2)

        // envelope = mu_peak * exp(-d_eff_sq/2)
        float dL_d_mu_peak = dL_d_envelope * exp_deff;
        float dL_d_d_eff_sq = -0.5f * dL_d_envelope * envelope;

        // d_eff_sq = C_val - B^2/(4A)
        float dL_dC = dL_d_d_eff_sq;
        dL_dB += dL_d_d_eff_sq * (-B * inv2A);  // -B/(2A)
        dL_dA += dL_d_d_eff_sq * B * B / (4.0f * A * A);

        // A = y.y, B = 2(z.y), C = z.z
        float dL_dy0 = dL_dA * 2.0f * y0 + dL_dB * 2.0f * z0;
        float dL_dy1 = dL_dA * 2.0f * y1 + dL_dB * 2.0f * z1;
        float dL_dy2 = dL_dA * 2.0f * y2 + dL_dB * 2.0f * z2;

        float dL_dz0 = dL_dB * 2.0f * y0 + dL_dC * 2.0f * z0;
        float dL_dz1 = dL_dB * 2.0f * y1 + dL_dC * 2.0f * z1;
        float dL_dz2 = dL_dB * 2.0f * y2 + dL_dC * 2.0f * z2;

        // Initialize L gradient accumulators
        float dL_dL00 = 0.0f, dL_dL10 = 0.0f, dL_dL11 = 0.0f;
        float dL_dL20 = 0.0f, dL_dL21 = 0.0f, dL_dL22 = 0.0f;

        // Backprop through y = L^{-1} d  (reverse order)
        // y2 = (d2 - L20*y0 - L21*y1) / L22
        dL_dL22 += dL_dy2 * (-y2 / L22);
        dL_dL20 += dL_dy2 * (-y0 / L22);
        dL_dL21 += dL_dy2 * (-y1 / L22);
        dL_dy0 += dL_dy2 * (-L20 / L22);
        dL_dy1 += dL_dy2 * (-L21 / L22);
        // y1 = (d1 - L10*y0) / L11
        dL_dL11 += dL_dy1 * (-y1 / L11);
        dL_dL10 += dL_dy1 * (-y0 / L11);
        dL_dy0 += dL_dy1 * (-L10 / L11);
        // y0 = d0 / L00
        dL_dL00 += dL_dy0 * (-y0 / L00);

        // Backprop through z = L^{-1} c_vec  (reverse order)
        // z2 = (c2 - L20*z0 - L21*z1) / L22
        dL_dL22 += dL_dz2 * (-z2 / L22);
        dL_dL20 += dL_dz2 * (-z0 / L22);
        dL_dL21 += dL_dz2 * (-z1 / L22);
        float dL_dc2 = dL_dz2 / L22;
        dL_dz0 += dL_dz2 * (-L20 / L22);
        dL_dz1 += dL_dz2 * (-L21 / L22);
        // z1 = (c1 - L10*z0) / L11
        dL_dL11 += dL_dz1 * (-z1 / L11);
        dL_dL10 += dL_dz1 * (-z0 / L11);
        float dL_dc1 = dL_dz1 / L11;
        dL_dz0 += dL_dz1 * (-L10 / L11);
        // z0 = c0 / L00
        dL_dL00 += dL_dz0 * (-z0 / L00);
        float dL_dc0 = dL_dz0 / L00;

        // ===== Write gradients =====

        // raw_peak gradient (through softplus)
        float d_softplus_peak = 1.0f / (1.0f + expf(-sp_beta * raw_p));
        atomicAdd(density_peak_grad + point_idx, dL_d_mu_peak * d_softplus_peak);

        // cov_raw gradient (L_raw): diagonal entries chain through softplus
        float d_sp_L00 = 1.0f / (1.0f + expf(-sp_beta * Lr[0]));
        float d_sp_L11 = 1.0f / (1.0f + expf(-sp_beta * Lr[2]));
        float d_sp_L22 = 1.0f / (1.0f + expf(-sp_beta * Lr[5]));
        atomicAdd(cov_raw_grad + point_idx * 6 + 0, dL_dL00 * d_sp_L00);
        atomicAdd(cov_raw_grad + point_idx * 6 + 1, dL_dL10);
        atomicAdd(cov_raw_grad + point_idx * 6 + 2, dL_dL11 * d_sp_L11);
        atomicAdd(cov_raw_grad + point_idx * 6 + 3, dL_dL20);
        atomicAdd(cov_raw_grad + point_idx * 6 + 4, dL_dL21);
        atomicAdd(cov_raw_grad + point_idx * 6 + 5, dL_dL22 * d_sp_L22);

        // delta_raw gradient: c_off = current_point + cell_r * tanh(dr)
        // c_vec = origin - c_off  =>  dL/dc_off = -dL/dc
        // dL/d(dr[i]) = -dL/dc[i] * cell_r * sech^2(dr[i])
        Vec3f dL_dc(dL_dc0, dL_dc1, dL_dc2);
        float sech2_0 = 1.0f - tanh_dr0 * tanh_dr0;
        float sech2_1 = 1.0f - tanh_dr1 * tanh_dr1;
        float sech2_2 = 1.0f - tanh_dr2 * tanh_dr2;
        atomicAdd(delta_raw_grad + point_idx * 3 + 0, -dL_dc0 * cell_r * sech2_0);
        atomicAdd(delta_raw_grad + point_idx * 3 + 1, -dL_dc1 * cell_r * sech2_1);
        atomicAdd(delta_raw_grad + point_idx * 3 + 2, -dL_dc2 * cell_r * sech2_2);

        // Position gradient from Gaussian: dL/d(current_point) += -dL/dc
        // (c_off = current_point + ..., c_vec = origin - c_off)
        current_point_grad += Vec3f(-dL_dc0, -dL_dc1, -dL_dc2);

        // ===== Cell intersection position gradients (same as existing) =====
        float mu = mu_base;  // use base density for intersection grads
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

        return true;
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

// ============================================================================
// Device helpers for thin-surface kernels
// ============================================================================

__device__ __forceinline__ void quat_to_frame(
    const float *q, Vec3f &n, Vec3f &tang, Vec3f &bita)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float inv_norm = rsqrtf(w*w + x*x + y*y + z*z + 1e-12f);
    w *= inv_norm; x *= inv_norm; y *= inv_norm; z *= inv_norm;

    n[0]    = 1.0f - 2.0f*(y*y + z*z);
    n[1]    = 2.0f*(x*y + w*z);
    n[2]    = 2.0f*(x*z - w*y);

    tang[0] = 2.0f*(x*y - w*z);
    tang[1] = 1.0f - 2.0f*(x*x + z*z);
    tang[2] = 2.0f*(y*z + w*x);

    bita[0] = 2.0f*(x*z + w*y);
    bita[1] = 2.0f*(y*z - w*x);
    bita[2] = 1.0f - 2.0f*(x*x + y*y);
}

// Adjoint of quat_to_frame. Accumulates into dL_dq[4].
__device__ __forceinline__ void quat_to_frame_bwd(
    const float *q,
    const Vec3f &dL_dn, const Vec3f &dL_dtang, const Vec3f &dL_dbita,
    float *dL_dq)
{
    float w = q[0], x = q[1], y = q[2], z = q[3];
    float norm_sq = w*w + x*x + y*y + z*z + 1e-12f;
    float inv_norm = rsqrtf(norm_sq);
    float w_ = w*inv_norm, x_ = x*inv_norm, y_ = y*inv_norm, z_ = z*inv_norm;

    // Gradients w.r.t. normalized (w_, x_, y_, z_)
    float dw = 0.0f, dx = 0.0f, dy = 0.0f, dz = 0.0f;

    // n[0]=1-2(y²+z²), n[1]=2(xy+wz), n[2]=2(xz-wy)
    dy += dL_dn[0] * (-4.0f*y_);
    dz += dL_dn[0] * (-4.0f*z_);
    dw += dL_dn[1] * 2.0f*z_;
    dx += dL_dn[1] * 2.0f*y_;
    dy += dL_dn[1] * 2.0f*x_;
    dz += dL_dn[1] * 2.0f*w_;
    dw += dL_dn[2] * (-2.0f*y_);
    dx += dL_dn[2] * 2.0f*z_;
    dy += dL_dn[2] * (-2.0f*w_);
    dz += dL_dn[2] * 2.0f*x_;

    // t[0]=2(xy-wz), t[1]=1-2(x²+z²), t[2]=2(yz+wx)
    dw += dL_dtang[0] * (-2.0f*z_);
    dx += dL_dtang[0] * 2.0f*y_;
    dy += dL_dtang[0] * 2.0f*x_;
    dz += dL_dtang[0] * (-2.0f*w_);
    dx += dL_dtang[1] * (-4.0f*x_);
    dz += dL_dtang[1] * (-4.0f*z_);
    dw += dL_dtang[2] * 2.0f*x_;
    dx += dL_dtang[2] * 2.0f*w_;
    dy += dL_dtang[2] * 2.0f*z_;
    dz += dL_dtang[2] * 2.0f*y_;

    // b[0]=2(xz+wy), b[1]=2(yz-wx), b[2]=1-2(x²+y²)
    dw += dL_dbita[0] * 2.0f*y_;
    dx += dL_dbita[0] * 2.0f*z_;
    dy += dL_dbita[0] * 2.0f*w_;
    dz += dL_dbita[0] * 2.0f*x_;
    dw += dL_dbita[1] * (-2.0f*x_);
    dx += dL_dbita[1] * (-2.0f*w_);
    dy += dL_dbita[1] * 2.0f*z_;
    dz += dL_dbita[1] * 2.0f*y_;
    dx += dL_dbita[2] * (-4.0f*x_);
    dy += dL_dbita[2] * (-4.0f*y_);

    // Chain through normalization: q_ = q * inv_norm
    float dot = w_*dw + x_*dx + y_*dy + z_*dz;
    dL_dq[0] += (dw - dot*w_) * inv_norm;
    dL_dq[1] += (dx - dot*x_) * inv_norm;
    dL_dq[2] += (dy - dot*y_) * inv_norm;
    dL_dq[3] += (dz - dot*z_) * inv_norm;
}

// ============================================================================
// ct_thinsurface_forward
// ============================================================================

template <int block_size>
__global__ void ct_thinsurface_forward(
    TraceSettings settings,
    const Vec3f *__restrict__ points,
    const float *__restrict__ density,        // raw density_base (N,)
    const float *__restrict__ density_delta,  // raw delta (N,)
    const float *__restrict__ quaternions,    // (N, 4) [w,x,y,z]
    const float *__restrict__ texel_sites_2d, // (N, K, 2)
    const float *__restrict__ texel_heights,  // (N, K)
    const float *__restrict__ cell_radius,    // (N,)
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    const Vec4h *__restrict__ adjacent_diff,
    const Ray *__restrict__ rays,
    uint32_t num_rays,
    const uint32_t *__restrict__ start_point_index,
    float *__restrict__ ray_projection,
    uint32_t *__restrict__ num_intersections,
    float *__restrict__ point_contribution,
    uint32_t *__restrict__ point_hit_count)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float projection = 0.0f;
    constexpr float sp_beta = 10.0f;
    const int K = settings.thin_K;
    const float thin_temp = settings.thin_temp;
    const float thin_height_eps = settings.thin_height_eps;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // Densities
        float raw_base  = density[point_idx];
        float mu_bar    = (sp_beta * raw_base > 20.0f) ? raw_base
                          : logf(1.0f + expf(sp_beta * raw_base)) / sp_beta;
        // raw_delta is the learnable parameter. Two parameterizations:
        //   absolute (legacy):  delta_val = raw_delta
        //   relative (M5 rescue): delta_val = rho * mu_bar * tanh(raw_delta)
        // so that |delta_val| <= rho * mu_bar and both mu_p, mu_n >= 0 for
        // rho in (0, 1].  At raw_delta=0 both collapse to delta_val=0, so
        // activation continuity at init is preserved by construction.
        float raw_delta = density_delta[point_idx];
        float delta_val;
        if (settings.thin_surface_relative_delta) {
            float rho = settings.thin_surface_delta_max_frac;
            delta_val = rho * mu_bar * tanhf(raw_delta);
        } else {
            delta_val = raw_delta;
        }
        float mu_p      = fmaxf(mu_bar + delta_val, 0.0f);
        float mu_n      = fmaxf(mu_bar - delta_val, 0.0f);

        // Tangent frame from quaternion
        const float *q = quaternions + point_idx * 4;
        Vec3f n, tang, bita;
        quat_to_frame(q, n, tang, bita);

        float dp = n.dot(ray.direction);

        // Grazing-ray fallback: surface crosses at near-infinite t
        if (fabsf(dp) < 1e-3f) {
            projection += mu_bar * delta_t;
            if (point_contribution) atomicAdd(point_contribution + point_idx, delta_t);
            if (point_hit_count)    atomicAdd(point_hit_count + point_idx, 1u);
            return true;
        }

        float r = cell_radius[point_idx];

        // Fixed-point step 1: query point on flat plane
        float t_flat = (current_point - ray.origin).dot(n) / dp;
        float t_q0 = (dp < 0.0f) ? fmaxf(t_0, t_flat) : t_0;
        Vec3f x0 = ray.origin + t_q0 * ray.direction;

        // Fixed-point step 2: soft-Voronoi height eval at x0
        float h_sum = 0.0f, w_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float *s2d = texel_sites_2d + (point_idx * K + k) * 2;
            Vec3f site3 = current_point + r * (s2d[0] * tang + s2d[1] * bita);
            float d2 = (x0 - site3).squaredNorm() / (r * r + 1e-20f);
            float w = expf(-thin_temp * d2);
            h_sum += w * (r * texel_heights[point_idx * K + k]);
            w_sum += w;
        }
        float w_sum_safe = fmaxf(w_sum, 1e-20f);
        float h_eval = h_sum / w_sum_safe;

        // Fixed-point step 3: surface intersection with height offset
        float t_surf = ((current_point - ray.origin).dot(n) + h_eval) / dp;

        // Two-sided partition
        float mu_near = (dp > 0.0f) ? mu_n : mu_p;
        float mu_far  = (dp > 0.0f) ? mu_p : mu_n;

        float t_s = fminf(fmaxf(t_surf, t_0), t_1);
        bool crossing = (t_surf > t_0 + thin_height_eps) &&
                        (t_surf < t_1 - thin_height_eps);

        float contrib;
        if (crossing) {
            contrib = mu_near * (t_s - t_0) + mu_far * (t_1 - t_s);
        } else {
            bool plus_side = (dp > 0.0f) ? (t_surf <= t_0) : (t_surf >= t_1);
            contrib = (plus_side ? mu_p : mu_n) * delta_t;
        }
        projection += contrib;

        if (point_contribution) atomicAdd(point_contribution + point_idx, delta_t);
        if (point_hit_count)    atomicAdd(point_hit_count + point_idx, 1u);

        return true;
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

// ============================================================================
// ct_independent_forward
// ----------------------------------------------------------------------------
// LC64 plan v3 Commit 2A -- CUDA-native independent-side FORWARD only.
// Renders exactly two raw side tensors per cell:
//
//     mu_plus  = activation_scale * softplus(raw_plus,  beta=10)
//     mu_minus = activation_scale * softplus(raw_minus, beta=10)
//
// The thin-surface geometry (quaternion + K texel sites + heights) and the
// crossing / non-crossing / dp-sign semantics are REUSED UNCHANGED from
// ct_thinsurface_forward: the upstream kernel computes the same two-sided
// partition over the cell along the surface normal, and we only swap the
// "mu_p / mu_n" constants for the independently-evaluated softplus values.
// Legacy base `density` is not read (it is frozen in this mode at the
// optimizer level; the binding rejects mixed inputs so the kernel never
// sees a stale density pointer).
//
// Backward is NOT implemented in Commit 2A; the trace_backward binding
// rejects independent-mode calls before any kernel launch.
// ============================================================================

template <int block_size>
__global__ void ct_independent_forward(
    TraceSettings settings,
    const Vec3f *__restrict__ points,
    const float *__restrict__ raw_plus,        // (N,1) raw logit, softplus’d to mu_plus
    const float *__restrict__ raw_minus,       // (N,1) raw logit, softplus’d to mu_minus
    const float *__restrict__ quaternions,     // (N, 4) [w,x,y,z]
    const float *__restrict__ texel_sites_2d,  // (N, K, 2)
    const float *__restrict__ texel_heights,   // (N, K)
    const float *__restrict__ cell_radius,     // (N,)
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    const Vec4h *__restrict__ adjacent_diff,
    const Ray *__restrict__ rays,
    uint32_t num_rays,
    const uint32_t *__restrict__ start_point_index,
    float *__restrict__ ray_projection,
    uint32_t *__restrict__ num_intersections,
    float *__restrict__ point_contribution,
    uint32_t *__restrict__ point_hit_count)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float projection = 0.0f;
    constexpr float sp_beta = 10.0f;
    const int K = settings.thin_K;
    const float thin_temp = settings.thin_temp;
    const float thin_height_eps = settings.thin_height_eps;
    const float activation_scale = settings.thin_surface_activation_scale;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        // Independent-side per-side attenuation.
        // activation_scale is applied AFTER softplus so the legacy
        // identity mu=softplus(raw) is preserved when the scene sets
        // activation_scale=1.0 (the default).
        float raw_p = raw_plus[point_idx];
        float mu_p = activation_scale * (
            (sp_beta * raw_p > 20.0f) ? raw_p
            : logf(1.0f + expf(sp_beta * raw_p)) / sp_beta);
        float raw_m = raw_minus[point_idx];
        float mu_n = activation_scale * (
            (sp_beta * raw_m > 20.0f) ? raw_m
            : logf(1.0f + expf(sp_beta * raw_m)) / sp_beta);
        // mu_p / mu_n are already nonneg by construction of softplus;
        // the legacy branches fmax(...,0) here is a no-op for the
        // independent path but kept documented for readers.

        // Tangent frame from quaternion (unchanged from thinsurface_forward).
        const float *q = quaternions + point_idx * 4;
        Vec3f n, tang, bita;
        quat_to_frame(q, n, tang, bita);

        float dp = n.dot(ray.direction);

        // Grazing-ray fallback: when the surface is nearly parallel to
        // the ray the crossing t is ill-conditioned; the legacy branch
        // uses mu_bar * delta_t, but under independent mode there is no
        // mu_bar -- use the side-averaged mean (which equals mu_bar when
        // raw_plus == raw_minus, so the zero-split invariant below
        // collapses to the scalar baseline).
        if (fabsf(dp) < 1e-3f) {
            float mu_bar = 0.5f * (mu_p + mu_n);
            projection += mu_bar * delta_t;
            if (point_contribution) atomicAdd(point_contribution + point_idx, delta_t);
            if (point_hit_count)    atomicAdd(point_hit_count + point_idx, 1u);
            return true;
        }

        float r = cell_radius[point_idx];

        // Fixed-point step 1: query point on flat plane (unchanged).
        float t_flat = (current_point - ray.origin).dot(n) / dp;
        float t_q0 = (dp < 0.0f) ? fmaxf(t_0, t_flat) : t_0;
        Vec3f x0 = ray.origin + t_q0 * ray.direction;

        // Fixed-point step 2: soft-Voronoi height eval (unchanged).
        float h_sum = 0.0f, w_sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            const float *s2d = texel_sites_2d + (point_idx * K + k) * 2;
            Vec3f site3 = current_point + r * (s2d[0] * tang + s2d[1] * bita);
            float d2 = (x0 - site3).squaredNorm() / (r * r + 1e-20f);
            float w = expf(-thin_temp * d2);
            h_sum += w * (r * texel_heights[point_idx * K + k]);
            w_sum += w;
        }
        float w_sum_safe = fmaxf(w_sum, 1e-20f);
        float h_eval = h_sum / w_sum_safe;

        // Fixed-point step 3: surface intersection (unchanged).
        float t_surf = ((current_point - ray.origin).dot(n) + h_eval) / dp;

        // Two-sided partition: near / far sides are DERIVED from the
        // physical side constants, not a symmetric delta.  The dp-sign
        // mapping is identical to the legacy branches:
        //   dp > 0: ray origin is on the -n side; near=-n=mu_minus, far=+n=mu_plus
        //   dp < 0: ray origin is on the +n side; near=+n=mu_plus,   far=-n=mu_minus
        float mu_near = (dp > 0.0f) ? mu_n : mu_p;
        float mu_far  = (dp > 0.0f) ? mu_p : mu_n;

        float t_s = fminf(fmaxf(t_surf, t_0), t_1);
        bool crossing = (t_surf > t_0 + thin_height_eps) &&
                        (t_surf < t_1 - thin_height_eps);

        float contrib;
        if (crossing) {
            contrib = mu_near * (t_s - t_0) + mu_far * (t_1 - t_s);
        } else {
            bool plus_side = (dp > 0.0f) ? (t_surf <= t_0) : (t_surf >= t_1);
            contrib = (plus_side ? mu_p : mu_n) * delta_t;
        }
        projection += contrib;

        if (point_contribution) atomicAdd(point_contribution + point_idx, delta_t);
        if (point_hit_count)    atomicAdd(point_hit_count + point_idx, 1u);

        return true;
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

// ============================================================================
// ct_thinsurface_backward
// ============================================================================

template <int block_size>
__global__ void ct_thinsurface_backward(
    TraceSettings settings,
    const Vec3f *__restrict__ points,
    const float *__restrict__ density,        // raw density_base (N,)
    const float *__restrict__ density_delta,  // raw delta (N,)
    const float *__restrict__ quaternions,    // (N, 4) [w,x,y,z]
    const float *__restrict__ texel_sites_2d, // (N, K, 2)
    const float *__restrict__ texel_heights,  // (N, K)
    const float *__restrict__ cell_radius,    // (N,)
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    const Vec4h *__restrict__ adjacent_diff,
    const Ray *__restrict__ rays,
    uint32_t num_rays,
    const uint32_t *__restrict__ start_point_index,
    const float *__restrict__ ray_projection_grad,
    const float *__restrict__ ray_error,
    Vec3f *__restrict__ points_grad,
    float *__restrict__ density_base_grad,
    float *__restrict__ density_delta_grad,
    float *__restrict__ quaternions_grad,
    float *__restrict__ texel_sites_2d_grad,
    float *__restrict__ texel_heights_grad,
    float *__restrict__ point_error)
{
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float dL_dprojection = ray_projection_grad[thread_idx];
    float error = 0.0f;
    if (ray_error) error = ray_error[thread_idx];

    constexpr float sp_beta = 10.0f;
    const int K = settings.thin_K;
    const float thin_temp = settings.thin_temp;
    const float thin_height_eps = settings.thin_height_eps;

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
        float delta_t = fmaxf(t_1 - t_0, 0.0f);

        if (point_error) {
            atomicAdd(point_error + point_idx, delta_t * error);
        }

        // ---- Recompute forward quantities ----
        float raw_base  = density[point_idx];
        float mu_bar    = (sp_beta * raw_base > 20.0f) ? raw_base
                          : logf(1.0f + expf(sp_beta * raw_base)) / sp_beta;
        float d_softplus = 1.0f / (1.0f + expf(-sp_beta * raw_base));

        float raw_delta = density_delta[point_idx];
        // Mirror the forward-branch dispatch (see ct_thinsurface_forward):
        //   absolute: delta_val = raw_delta,                 d(raw_delta)/d(delta_val) = 1
        //   relative: delta_val = rho * mu_bar * tanh(raw), d/d(raw) chain via sech^2
        // t_raw holds tanh(raw_delta) for both branches so the mu_p / mu_n
        // adjoint below stays a single code path; the absolute branch sets
        // t_raw = 0 so the extra dL/dmu_bar additive collapses to 0 too.
        float t_raw;
        float delta_val;
        if (settings.thin_surface_relative_delta) {
            float rho = settings.thin_surface_delta_max_frac;
            t_raw     = tanhf(raw_delta);
            delta_val = rho * mu_bar * t_raw;
        } else {
            t_raw     = 0.0f;
            delta_val = raw_delta;
        }
        float mu_p      = fmaxf(mu_bar + delta_val, 0.0f);
        float mu_n      = fmaxf(mu_bar - delta_val, 0.0f);
        float ind_p     = (mu_p > 0.0f) ? 1.0f : 0.0f;
        float ind_n     = (mu_n > 0.0f) ? 1.0f : 0.0f;

        const float *q = quaternions + point_idx * 4;
        Vec3f n, tang, bita;
        quat_to_frame(q, n, tang, bita);

        float dp = n.dot(ray.direction);

        // ---- Grazing-ray fallback ----
        if (fabsf(dp) < 1e-3f) {
            // contrib = mu_bar * delta_t
            float dL_dmu_bar = dL_dprojection * delta_t;
            atomicAdd(density_base_grad + point_idx, dL_dmu_bar * d_softplus);
            // dL/d(density_delta) = 0

            float dL_ddelta_t = dL_dprojection * mu_bar;
            float dL_dt0 = -dL_ddelta_t, dL_dt1 = dL_ddelta_t;
            // Propagate through bisector intersections (same as ct_backward)
            Vec3f dt0_dprev;
            if (prev_point_idx != UINT32_MAX)
                dt0_dprev = cell_intersection_grad(prev_point, current_point, ray);
            else
                dt0_dprev = Vec3f::Zero();
            Vec3f dt1_dcur  = cell_intersection_grad(current_point, next_point, ray);
            Vec3f dt0_dcur  = cell_intersection_grad(current_point, prev_point, ray);
            Vec3f dt1_dnext = cell_intersection_grad(next_point, current_point, ray);

            prev_point_grad += dL_dt0 * dt0_dprev;
            current_point_grad += dL_dt0 * dt0_dcur + dL_dt1 * dt1_dcur;
            next_point_grad += dL_dt1 * dt1_dnext;

            if (prev_point_idx != UINT32_MAX)
                atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
            prev_point = current_point;
            prev_point_idx = point_idx;
            prev_point_grad = current_point_grad;
            current_point_grad = next_point_grad;
            next_point_grad = Vec3f::Zero();
            return true;
        }

        float r = cell_radius[point_idx];

        // ---- Recompute fixed-point quantities ----
        float t_flat = (current_point - ray.origin).dot(n) / dp;
        bool q0_from_tflat = (dp < 0.0f) && (t_flat >= t_0);
        float t_q0 = q0_from_tflat ? t_flat : t_0;
        Vec3f x0 = ray.origin + t_q0 * ray.direction;

        // soft-Voronoi weights
        float h_sum = 0.0f, w_sum = 0.0f;
        float w_arr[8]; // max K=8
        for (int k = 0; k < K; ++k) {
            const float *s2d = texel_sites_2d + (point_idx * K + k) * 2;
            Vec3f site3 = current_point + r * (s2d[0] * tang + s2d[1] * bita);
            float d2 = (x0 - site3).squaredNorm() / (r * r + 1e-20f);
            float w = expf(-thin_temp * d2);
            w_arr[k] = w;
            h_sum += w * (r * texel_heights[point_idx * K + k]);
            w_sum += w;
        }
        float w_sum_safe = fmaxf(w_sum, 1e-20f);
        float h_eval = h_sum / w_sum_safe;

        float t_surf = ((current_point - ray.origin).dot(n) + h_eval) / dp;

        float mu_near = (dp > 0.0f) ? mu_n : mu_p;
        float mu_far  = (dp > 0.0f) ? mu_p : mu_n;
        float t_s = fminf(fmaxf(t_surf, t_0), t_1);
        bool crossing = (t_surf > t_0 + thin_height_eps) &&
                        (t_surf < t_1 - thin_height_eps);

        // ---- Backward pass ----

        // Chord-length gradients
        float dL_dmu_near = 0.0f, dL_dmu_far = 0.0f;
        float dL_dt_s = 0.0f;
        // Chord-endpoint (t_0, t_1) gradients feed the point-position grads via
        // cell_intersection_grad. These are INDEPENDENT of t_s: in the crossing
        // branch t_surf depends on cp/n/h_eval but NOT on t_0/t_1, so the only
        // t_0/t_1 dependence of contrib is the explicit chord lengths
        // (t_s - t_0),(t_1 - t_s) [crossing] or (t_1 - t_0) [non-crossing].
        // Previously the crossing branch set dL_ddelta_t=0 and reused
        // dL_dt0=-dL_ddelta_t, dL_dt1=dL_ddelta_t as the SOLE endpoint grads,
        // dropping the chord-endpoint (point-position) gradient entirely --
        // at delta=0 this gave scalar point-grad ~mu_bar but thin point-grad 0,
        // a confirmed correctness defect (see specs/SPLIT-CELL-EXECUTION-LOG.md).
        float dL_dt0_chord = 0.0f, dL_dt1_chord = 0.0f;
        // dL_ddelta_t now carries ONLY the t_q0->t_0 contribution (set in the
        // t_surf/x0 backward block below when t_q0 == t_0). It must NOT flow to
        // t_1 (t_q0 does not depend on t_1); the old tail leaked it to t_1.
        float dL_ddelta_t = 0.0f;

        if (crossing) {
            dL_dmu_near = dL_dprojection * (t_s - t_0);
            dL_dmu_far  = dL_dprojection * (t_1 - t_s);
            dL_dt_s     = dL_dprojection * (mu_near - mu_far);
            // d(contrib)/d(t_0) = -mu_near ; d(contrib)/d(t_1) = +mu_far.
            // At delta=0 mu_near=mu_far=mu_bar -> -mu_bar/+mu_bar, matching the
            // scalar contrib = mu_bar*(t_1 - t_0) endpoint gradient.
            dL_dt0_chord = -dL_dprojection * mu_near;
            dL_dt1_chord =  dL_dprojection * mu_far;
        } else {
            bool plus_side = (dp > 0.0f) ? (t_surf <= t_0) : (t_surf >= t_1);
            float mu_eff = plus_side ? mu_p : mu_n;
            // contrib = mu_eff * (t_1 - t_0); d(contrib)/d(mu_eff) = delta_t.
            // Route to mu_near/mu_far respecting the dp-sign physical mapping:
            //   dp>0: mu_near=mu_n, mu_far=mu_p
            //   dp<0: mu_near=mu_p, mu_far=mu_n
            // (The old code assigned purely by plus_side, which was correct
            // only for dp>0 and inverted the mu_p/mu_n adjoint for dp<0.)
            if (plus_side) {            // mu_eff = mu_p
                if (dp > 0.0f) dL_dmu_far  = dL_dprojection * delta_t;  // mu_p=mu_far
                else           dL_dmu_near = dL_dprojection * delta_t;  // mu_p=mu_near
            } else {                    // mu_eff = mu_n
                if (dp > 0.0f) dL_dmu_near = dL_dprojection * delta_t;  // mu_n=mu_near
                else           dL_dmu_far  = dL_dprojection * delta_t;  // mu_n=mu_far
            }
            dL_dt0_chord = -dL_dprojection * mu_eff;
            dL_dt1_chord =  dL_dprojection * mu_eff;
        }

        // Unscramble mu_near/mu_far -> mu_p/mu_n
        float dL_dmu_p, dL_dmu_n;
        if (dp > 0.0f) {
            dL_dmu_n = dL_dmu_near;
            dL_dmu_p = dL_dmu_far;
        } else {
            dL_dmu_p = dL_dmu_near;
            dL_dmu_n = dL_dmu_far;
        }

        // mu_p = max(mu_bar + delta_val, 0), mu_n = max(mu_bar - delta_val, 0)
        float dL_dmu_bar = ind_p * dL_dmu_p + ind_n * dL_dmu_n;
        float dL_ddelta  = ind_p * dL_dmu_p - ind_n * dL_dmu_n;
        // Final mu_bar adjoint = contribution from explicit mu_p / mu_n via
        // mu_bar + delta (== 1 from ind_* mask) PLUS any contribution from
        // delta depending on mu_bar (only in the relative parameterization:
        // delta = rho * mu_bar * tanh(raw), so d(delta)/d(mu_bar) = rho * tanh(raw)).
        if (settings.thin_surface_relative_delta) {
            float rho = settings.thin_surface_delta_max_frac;
            dL_dmu_bar += dL_ddelta * rho * t_raw;
            // Chain to the raw learnable parameter:
            //   delta = rho * mu_bar * tanh(raw)  ->
            //   d(delta)/d(raw) = rho * mu_bar * (1 - tanh^2(raw)) = rho * mu_bar * sech^2(raw)
            float sech2 = 1.0f - t_raw * t_raw;
            atomicAdd(density_delta_grad + point_idx,
                      dL_ddelta * rho * mu_bar * sech2);
        } else {
            // Absolute (legacy): delta = raw_delta, so d(delta)/d(raw_delta)=1
            // and there is no extra mu_bar contribution from delta.
            atomicAdd(density_delta_grad + point_idx, dL_ddelta);
        }
        atomicAdd(density_base_grad + point_idx, dL_dmu_bar * d_softplus);

        // ---- dL/d(t_surf) -> dL/d(h_eval) + dL/d(current_point) + dL/dn ----
        // t_surf = (cp · n - origin · n + h_eval) / dp
        // dt_surf/dh_eval = 1/dp
        // dt_surf/d(cp) = n/dp
        // dt_surf/dn = (cp - origin)/dp - t_surf * d/dp  =  (cp - origin - t_surf*d) / dp

        Vec3f dL_dn = Vec3f::Zero();
        Vec3f dL_dtang = Vec3f::Zero();
        Vec3f dL_dbita = Vec3f::Zero();
        Vec3f dL_dcurrent_point = Vec3f::Zero();

        if (fabsf(dL_dt_s) > 0.0f) {
            float dL_dt_surf = dL_dt_s; // t_s = clamp(t_surf, t_0, t_1)
            // Only when crossing is true, t_s = t_surf, so the clamp is transparent

            float dL_dh_eval = dL_dt_surf / dp;
            dL_dcurrent_point += dL_dt_surf * n / dp;
            dL_dn += dL_dt_surf * (current_point - ray.origin - t_surf * ray.direction) / dp;

            // ---- dL/d(h_eval) -> soft-Voronoi backward ----
            // h_eval = h_sum / w_sum_safe
            float dL_dh_sum   = dL_dh_eval / w_sum_safe;
            float dL_dw_total = -dL_dh_eval * h_eval / w_sum_safe;  // chain via w_sum

            Vec3f dL_dx0 = Vec3f::Zero();

            for (int k = 0; k < K; ++k) {
                float w = w_arr[k];
                float h_k = texel_heights[point_idx * K + k];
                const float *s2d = texel_sites_2d + (point_idx * K + k) * 2;
                Vec3f site3 = current_point + r * (s2d[0] * tang + s2d[1] * bita);

                // dL/d(w_k) = dL/d(h_sum) * r * h_k + dL/dw_total
                float dL_dw = dL_dh_sum * (r * h_k) + dL_dw_total;

                // dL/d(texel_heights[k]) via h_sum contribution
                float dL_dhk = dL_dh_sum * w * r;
                atomicAdd(texel_heights_grad + point_idx * K + k, dL_dhk);

                // dL/d(d2_k) via w_k = exp(-temp * d2_k)
                float dL_dd2 = dL_dw * (-thin_temp) * w;

                Vec3f diff = x0 - site3;
                float inv_r2 = 1.0f / (r * r + 1e-20f);

                // dL/d(x0) += dL/d(d2_k) * 2 * diff / r²
                dL_dx0 += dL_dd2 * 2.0f * diff * inv_r2;

                // dL/d(site3) = -dL/d(x0 - site3) part
                Vec3f dL_dsite3 = dL_dd2 * (-2.0f) * diff * inv_r2;

                // site3 = cp + r * (s2d[0]*tang + s2d[1]*bita)
                dL_dcurrent_point += dL_dsite3;
                dL_dtang += dL_dsite3 * (r * s2d[0]);
                dL_dbita += dL_dsite3 * (r * s2d[1]);

                // dL/d(texel_sites_2d[k, 0]) and [k, 1]
                float dL_ds0 = dL_dsite3.dot(r * tang);
                float dL_ds1 = dL_dsite3.dot(r * bita);
                atomicAdd(texel_sites_2d_grad + (point_idx * K + k) * 2 + 0, dL_ds0);
                atomicAdd(texel_sites_2d_grad + (point_idx * K + k) * 2 + 1, dL_ds1);
            }

            // ---- dL/d(x0) -> dL/d(t_q0) -> dL/d(current_point, n) ----
            float dL_dt_q0 = dL_dx0.dot(ray.direction);

            if (q0_from_tflat) {
                // t_q0 = t_flat = (cp - origin).n / dp
                // dt_flat/d(cp) = n/dp
                // dt_flat/dn = (cp - origin)/dp - t_flat * d/dp
                dL_dcurrent_point += dL_dt_q0 * n / dp;
                dL_dn += dL_dt_q0 * (current_point - ray.origin - t_flat * ray.direction) / dp;
                // Note: dL_dt0 contribution from t_q0 = max(t_0, t_flat) when t_flat < t_0
                // is zero since we're in the t_flat >= t_0 branch
            } else {
                // t_q0 = t_0; gradient flows to t_0 via bisector below
                // We accumulate into dL_ddelta_t path: dL/d(t_0) += -dL_dt_q0
                // (since t_0 contributes to delta_t and now also to t_q0)
                // But we handle t_0 grad below via the bisector chain,
                // so we need to record this extra contribution.
                // Use dL_ddelta_t to carry the cell-boundary t_0 signal:
                dL_ddelta_t += -dL_dt_q0; // feeds into dL_dt0 = dL_dt0_chord - dL_ddelta_t
            }
        }

        // ---- Quaternion backward ----
        if (dL_dn.squaredNorm() > 0.0f || dL_dtang.squaredNorm() > 0.0f ||
            dL_dbita.squaredNorm() > 0.0f) {
            float local_qg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            quat_to_frame_bwd(q, dL_dn, dL_dtang, dL_dbita, local_qg);
            float *qg = quaternions_grad + point_idx * 4;
            atomicAdd(qg + 0, local_qg[0]);
            atomicAdd(qg + 1, local_qg[1]);
            atomicAdd(qg + 2, local_qg[2]);
            atomicAdd(qg + 3, local_qg[3]);
        }

        // ---- Cell-boundary position gradients ----
        // chord-endpoint contribution (always present) + t_q0 carrier (only
        // flows to t_0, when t_q0 == t_0; dL_ddelta_t is 0 in non-crossing and
        // in crossing-at-delta=0 since the t_surf/x0 block is skipped there).
        float dL_dt0 = dL_dt0_chord - dL_ddelta_t;
        float dL_dt1 = dL_dt1_chord;

        Vec3f dt0_dprev;
        if (prev_point_idx != UINT32_MAX)
            dt0_dprev = cell_intersection_grad(prev_point, current_point, ray);
        else
            dt0_dprev = Vec3f::Zero();
        Vec3f dt1_dcur  = cell_intersection_grad(current_point, next_point, ray);
        Vec3f dt0_dcur  = cell_intersection_grad(current_point, prev_point, ray);
        Vec3f dt1_dnext = cell_intersection_grad(next_point, current_point, ray);

        prev_point_grad += dL_dt0 * dt0_dprev;
        dL_dcurrent_point += dL_dt0 * dt0_dcur + dL_dt1 * dt1_dcur;
        next_point_grad    += dL_dt1 * dt1_dnext;

        current_point_grad += dL_dcurrent_point;

        if (prev_point_idx != UINT32_MAX)
            atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
        prev_point = current_point;
        prev_point_idx = point_idx;
        prev_point_grad = current_point_grad;
        current_point_grad = next_point_grad;
        next_point_grad = Vec3f::Zero();

        return true;
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

__global__ void precompute_activated_density(
    const float *__restrict__ density,
    float *__restrict__ activated,
    float *__restrict__ dsigmoid_out,
    uint32_t num_points) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;
    constexpr float sp_beta = 10.0f;
    float raw = density[i];
    float e = expf(sp_beta * raw);
    float mu = (sp_beta * raw > 20.0f) ? raw : logf(1.0f + e) / sp_beta;
    activated[i] = mu;
    if (dsigmoid_out) {
        dsigmoid_out[i] = (sp_beta * raw > 20.0f) ? 1.0f : e / (1.0f + e);
    }
}

__global__ void precompute_activated_density_vis(
    const float *__restrict__ density,
    float *__restrict__ activated,
    uint32_t num_points,
    float beta,
    float scale) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points)
        return;
    float raw = density[i];
    activated[i] = (beta * raw > 20.0f)
                       ? scale * raw
                       : scale * logf(1.0f + expf(beta * raw)) / beta;
}

template <int block_size>
__global__ void ct_interp_forward(TraceSettings settings,
                                   const Vec3f *__restrict__ points,
                                   const float *__restrict__ activated,
                                   const uint32_t *__restrict__ point_adjacency,
                                   const uint32_t *__restrict__ point_adjacency_offsets,
                                   const Vec4h *__restrict__ adjacent_diff,
                                   const float *__restrict__ cell_radius,
                                   const Ray *__restrict__ rays,
                                   uint32_t num_rays,
                                   const uint32_t *__restrict__ start_point_index,
                                   float *__restrict__ ray_projection,
                                   uint32_t *__restrict__ num_intersections,
                                   float *__restrict__ point_contribution,
                                   uint32_t *__restrict__ point_hit_count) {

    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= num_rays)
        return;

    Ray ray = rays[thread_idx];
    ray.direction /= ray.direction.norm();

    float projection = 0.0f;

    float sigma_sq_base = settings.idw_sigma * settings.idw_sigma;
    float sigma_v_sq = settings.idw_sigma_v * settings.idw_sigma_v;
    bool adaptive = settings.per_cell_sigma && cell_radius;
    bool per_nb = settings.per_neighbor_sigma && cell_radius;
    constexpr float eps = 1e-7f;
    constexpr float w_floor = 1e-6f;
    constexpr float volume_extent = 1.05f;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float t_mid = (t_0 + t_1) * 0.5f;
        Vec3f x_mid = ray.origin + t_mid * ray.direction;

        // Skip interpolation outside the reconstruction volume
        if (fabsf(x_mid[0]) > volume_extent || fabsf(x_mid[1]) > volume_extent || fabsf(x_mid[2]) > volume_extent) {
            return true;
        }

        // Compute self sigma_sq (Mode A: containing cell's radius)
        float self_sigma_sq;
        if (adaptive) {
            float self_r = cell_radius[point_idx];
            self_sigma_sq = sigma_sq_base * self_r * self_r;
        } else {
            self_sigma_sq = sigma_sq_base;
        }

        float mu_ref = activated[point_idx];
        Vec3f diff_self = x_mid - current_point;

        // Self contribution (Gaussian kernel, bilateral diff = 0)
        float d_sq_self = diff_self.squaredNorm();
        float w_self = expf(-d_sq_self / self_sigma_sq);

        float w_sum = w_self + w_floor;
        float mu_weighted = (w_self + w_floor) * mu_ref;

        // Neighbor contributions via adjacent_diff offsets
        uint32_t adj_begin = point_adjacency_offsets[point_idx];
        uint32_t adj_end = point_adjacency_offsets[point_idx + 1];

        for (uint32_t j = adj_begin; j < adj_end; ++j) {
            uint32_t nb = point_adjacency[j];
            float mu_nb = activated[nb];

            // Use precomputed half-precision offset instead of random global read
            Vec4h adj_h = adjacent_diff[j];
            Vec3f offset(__half2float(adj_h[0]),
                         __half2float(adj_h[1]),
                         __half2float(adj_h[2]));
            Vec3f diff_nb = diff_self - offset;

            // Per-neighbor or per-cell sigma
            float nb_sigma_sq;
            if (per_nb) {
                float nb_r = __half2float(adj_h[3]);
                nb_sigma_sq = sigma_sq_base * nb_r * nb_r;
            } else {
                nb_sigma_sq = self_sigma_sq;
            }

            // Fused Gaussian spatial + Gaussian bilateral in single exp
            float d_sq_nb = diff_nb.squaredNorm();
            float dmu = mu_nb - mu_ref;
            float w_nb = expf(-d_sq_nb / nb_sigma_sq - dmu * dmu / sigma_v_sq);

            w_sum += w_nb + w_floor;
            mu_weighted += (w_nb + w_floor) * mu_nb;
        }

        float mu = fmaxf(0.0f, mu_weighted / fmaxf(w_sum, eps));
        projection += mu * delta_t;

        if (point_contribution) {
            atomicAdd(point_contribution + point_idx, delta_t);
        }
        if (point_hit_count) {
            atomicAdd(point_hit_count + point_idx, 1u);
        }

        return true;
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
__global__ void ct_interp_backward(TraceSettings settings,
                                    const Vec3f *__restrict__ points,
                                    const float *__restrict__ activated,
                                    const float *__restrict__ dsigmoid,
                                    const uint32_t *__restrict__ point_adjacency,
                                    const uint32_t *__restrict__ point_adjacency_offsets,
                                    const Vec4h *__restrict__ adjacent_diff,
                                    const float *__restrict__ cell_radius,
                                    const Ray *__restrict__ rays,
                                    uint32_t num_rays,
                                    const uint32_t *__restrict__ start_point_index,
                                    const float *__restrict__ ray_projection_grad,
                                    const float *__restrict__ ray_error,
                                    Vec3f *__restrict__ points_grad,
                                    float *__restrict__ density_scalar_grad,
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

    float sigma_sq_base = settings.idw_sigma * settings.idw_sigma;
    float sigma_v_sq = settings.idw_sigma_v * settings.idw_sigma_v;
    bool adaptive = settings.per_cell_sigma && cell_radius;
    bool per_nb = settings.per_neighbor_sigma && cell_radius;
    constexpr float eps = 1e-7f;
    constexpr float w_floor = 1e-6f;
    constexpr float volume_extent = 1.05f;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f &next_point) {
        float delta_t = fmaxf(t_1 - t_0, 0.0f);
        float t_mid = (t_0 + t_1) * 0.5f;
        Vec3f x_mid = ray.origin + t_mid * ray.direction;

        // Skip outside reconstruction volume (must match forward)
        if (fabsf(x_mid[0]) > volume_extent || fabsf(x_mid[1]) > volume_extent || fabsf(x_mid[2]) > volume_extent) {
            if (prev_point_idx != UINT32_MAX) {
                atomic_add_vec(points_grad + prev_point_idx, prev_point_grad);
            }
            prev_point = current_point;
            prev_point_idx = point_idx;
            prev_point_grad = current_point_grad;
            current_point_grad = next_point_grad;
            next_point_grad = Vec3f::Zero();
            return true;
        }

        // Compute self sigma_sq (Mode A: containing cell's radius)
        float self_sigma_sq;
        if (adaptive) {
            float self_r = cell_radius[point_idx];
            self_sigma_sq = sigma_sq_base * self_r * self_r;
        } else {
            self_sigma_sq = sigma_sq_base;
        }

        float mu_ref = activated[point_idx];

        if (point_error) {
            float weight = delta_t;
            atomicAdd(point_error + point_idx, weight * error);
        }

        Vec3f diff_self = x_mid - current_point;

        // Self contribution (Gaussian kernel, bilateral diff = 0)
        float d_sq_self = diff_self.squaredNorm();
        float w_self = expf(-d_sq_self / self_sigma_sq);

        float w_sum = w_self + w_floor;
        float mu_weighted = (w_self + w_floor) * mu_ref;

        uint32_t adj_begin = point_adjacency_offsets[point_idx];
        uint32_t adj_end = point_adjacency_offsets[point_idx + 1];

        // --- Pass 1: accumulate w_sum and mu_weighted (2 running floats) ---
        for (uint32_t j = adj_begin; j < adj_end; ++j) {
            uint32_t nb = point_adjacency[j];
            float mu_nb = activated[nb];

            Vec4h adj_h = adjacent_diff[j];
            Vec3f offset(__half2float(adj_h[0]),
                         __half2float(adj_h[1]),
                         __half2float(adj_h[2]));
            Vec3f diff_nb = diff_self - offset;

            float nb_sigma_sq;
            if (per_nb) {
                float nb_r = __half2float(adj_h[3]);
                nb_sigma_sq = sigma_sq_base * nb_r * nb_r;
            } else {
                nb_sigma_sq = self_sigma_sq;
            }

            float d_sq_nb = diff_nb.squaredNorm();
            float dmu = mu_nb - mu_ref;
            float w_nb = expf(-d_sq_nb / nb_sigma_sq - dmu * dmu / sigma_v_sq);

            w_sum += w_nb + w_floor;
            mu_weighted += (w_nb + w_floor) * mu_nb;
        }

        float W = fmaxf(w_sum, eps);
        float mu = fmaxf(0.0f, mu_weighted / W);

        // indicator for ReLU clamp
        float indicator = (mu > 0.0f) ? 1.0f : 0.0f;
        float dL_dmu = dL_dprojection * delta_t * indicator;

        // --- Density gradient for self ---
        float alpha_self = w_self / W;
        atomicAdd(density_scalar_grad + point_idx,
                  dL_dmu * alpha_self * dsigmoid[point_idx]);

        // --- Position gradient for self (Gaussian: no 1/d singularity) ---
        {
            Vec3f pos_grad_self =
                dL_dmu * (w_self * 2.0f / (self_sigma_sq * W)) * (mu_ref - mu) * diff_self;
            current_point_grad += pos_grad_self;
        }

        // --- Pass 2: recompute weights, apply density + position gradients ---
        for (uint32_t j = adj_begin; j < adj_end; ++j) {
            uint32_t nb = point_adjacency[j];
            float mu_nb = activated[nb];

            Vec4h adj_h = adjacent_diff[j];
            Vec3f offset(__half2float(adj_h[0]),
                         __half2float(adj_h[1]),
                         __half2float(adj_h[2]));
            Vec3f diff_nb = diff_self - offset;

            float nb_sigma_sq;
            if (per_nb) {
                float nb_r = __half2float(adj_h[3]);
                nb_sigma_sq = sigma_sq_base * nb_r * nb_r;
            } else {
                nb_sigma_sq = self_sigma_sq;
            }

            float d_sq_nb = diff_nb.squaredNorm();
            float dmu = mu_nb - mu_ref;
            float w_nb = expf(-d_sq_nb / nb_sigma_sq - dmu * dmu / sigma_v_sq);

            // Density gradient for neighbor
            float alpha_k = w_nb / W;
            atomicAdd(density_scalar_grad + nb,
                      dL_dmu * alpha_k * dsigmoid[nb]);

            // Position gradient for neighbor (Gaussian kernel)
            Vec3f pos_grad_nb =
                dL_dmu * (w_nb * 2.0f / (nb_sigma_sq * W)) * (mu_nb - mu) * diff_nb;
            atomic_add_vec(points_grad + nb, pos_grad_nb);
        }

        // --- Cell intersection position gradients ---
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

        return true;
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
    const float *__restrict__ cell_radius,
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
        float nb_radius = cell_radius ? cell_radius[adjacent_idx] : 0.0f;
        adjacent_diff[offset_start + j] = Vec4h(diff[0], diff[1], diff[2], nb_radius);
    }
}

void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            const float *cell_radius,
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
                          cell_radius,
                          adjacent_diff);
}

__device__ __forceinline__ Vec3f rotate_yx(const Vec3f &v, float a, float b) {
    float ca = cosf(a), sa = sinf(a);
    float cb = cosf(b), sb = sinf(b);
    float x1 = v[0];
    float y1 = cb * v[1] - sb * v[2];
    float z1 = sb * v[1] + cb * v[2];
    return Vec3f(ca * x1 + sa * z1, y1, -sa * x1 + ca * z1);
}

__device__ __forceinline__ float compute_ao(
        uint32_t cell_idx,
        const Vec3f &x_sample,
        const Vec3f *__restrict__ points,
        const float *__restrict__ activated,
        const uint32_t *__restrict__ point_adjacency,
        const uint32_t *__restrict__ point_adjacency_offsets,
        const Vec4h *__restrict__ adjacent_diff,
        const float *__restrict__ ao_directions,
        uint32_t num_dirs,
        float max_dist,
        bool use_tf,
        float tf_density_min,
        float tf_density_max,
        float tf_opacity_scale,
        TransferFunctionTable tf_table) {
    uint32_t h = wang_hash(cell_idx);
    float ang_a = (h & 0xFFFFu) * (6.2831853f / 65536.0f);
    float ang_b = ((h >> 16) & 0xFFFFu) * (6.2831853f / 65536.0f);

    float vis_sum = 0.0f;
    for (uint32_t k = 0; k < num_dirs; ++k) {
        Vec3f base(ao_directions[3 * k + 0],
                   ao_directions[3 * k + 1],
                   ao_directions[3 * k + 2]);
        Vec3f d = rotate_yx(base, ang_a, ang_b);

        Ray ao_ray;
        ao_ray.origin = x_sample;
        ao_ray.direction = d;

        float tau = 0.0f;
        auto ao_functor = [&](uint32_t pi, float t0, float t1,
                              const Vec3f & /*cp*/, const Vec3f & /*np*/) {
            if (t0 >= max_dist)
                return false;
            float dt = fminf(t1, max_dist) - t0;
            if (dt <= 0.0f)
                return false;
            float mu = activated[pi];
            float opac;
            if (use_tf) {
                float range = tf_density_max - tf_density_min;
                float v = (range > 1e-8f)
                              ? fmaxf(0.0f, fminf((mu - tf_density_min) / range, 1.0f))
                              : 0.0f;
                Vec3f rgb_tmp;
                float tf_opacity;
                sample_transfer_function(v, tf_table, rgb_tmp, tf_opacity);
                opac = tf_opacity * tf_opacity_scale;
            } else {
                opac = mu;
            }
            tau += opac * dt;
            if (tau > 4.6f)
                return false;
            return t1 < max_dist;
        };

        trace<128, 4>(ao_ray, points, point_adjacency,
                      point_adjacency_offsets, adjacent_diff,
                      cell_idx, 64u, ao_functor);

        vis_sum += expf(-tau);
    }
    return vis_sum / float(num_dirs);
}

__global__ void ct_visualization(TraceSettings settings,
                                  VisualizationSettings vis_settings,
                                  Camera camera,
                                  CMapTable cmap_table,
                                  TransferFunctionTable tf_table,
                                  const Vec3f *__restrict__ points,
                                  const float *__restrict__ activated,
                                  const uint32_t *__restrict__ point_adjacency,
                                  const uint32_t *__restrict__ point_adjacency_offsets,
                                  const Vec4h *__restrict__ adjacent_diff,
                                  const float *__restrict__ ao_directions,
                                  uint32_t start_index,
                                  CUsurfObject output_surface) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= camera.width || j >= camera.height)
        return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction /= ray.direction.norm();

    // Slab-test against the slicing AABB; clip ray integration to [t_enter, t_exit]
    const Vec3f s_min = *vis_settings.slice_min;
    const Vec3f s_max = *vis_settings.slice_max;
    float aabb_t_enter = -1e38f, aabb_t_exit = 1e38f;
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        float d = ray.direction[ax];
        float inv_d = 1.0f / (fabsf(d) > 1e-20f ? d : copysignf(1e-20f, d));
        float t1 = (s_min[ax] - ray.origin[ax]) * inv_d;
        float t2 = (s_max[ax] - ray.origin[ax]) * inv_d;
        aabb_t_enter = fmaxf(aabb_t_enter, fminf(t1, t2));
        aabb_t_exit  = fminf(aabb_t_exit,  fmaxf(t1, t2));
    }
    const float t_enter = fmaxf(aabb_t_enter, 0.0f);
    const float t_exit  = aabb_t_exit;

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
        // Clip segment to the slicing AABB
        float tc_0 = fmaxf(t_0, t_enter);
        float tc_1 = fminf(t_1, t_exit);
        if (tc_0 >= tc_1)
            return t_1 < t_exit;  // skip; stop once past cube exit
        float delta_t = tc_1 - tc_0;

        float mu;
        if (vis_settings.interpolation_mode == InterpolationMode::IDW) {
            constexpr float w_floor = 1e-6f;
            constexpr float eps = 1e-7f;
            // Adaptive neighborhood thresholds: skip work when exp(-(r/sigma)^2) < 0.1
            // tau_const_sq = -ln(0.1) ~ 2.3026  -> skip all neighbors when r > 1.517*sigma
            // tau_2hop_sq  = tau_const_sq / 4    -> skip 2-hop when r > 0.758*sigma
            constexpr float TAU_CONST_SQ = 2.0f; // 2.302585f;
            constexpr float TAU_2HOP_SQ  = TAU_CONST_SQ * 0.25f;

            float t_mid = 0.5f * (tc_0 + tc_1);
            Vec3f x_mid = ray.origin + t_mid * ray.direction;

            float sigma_sq = vis_settings.idw_sigma * vis_settings.idw_sigma;
            float sigma_v_sq = vis_settings.idw_sigma_v * vis_settings.idw_sigma_v;

            float mu_ref = activated[point_idx];
            Vec3f diff_self = x_mid - current_point;

            uint32_t a0 = point_adjacency_offsets[point_idx];
            uint32_t a1 = point_adjacency_offsets[point_idx + 1];

            // Pre-pass: compute cell radius squared as max squared distance to any neighbor
            float r_sq = 0.f;
            for (uint32_t j = a0; j < a1; ++j) {
                Vec4h adj_h = adjacent_diff[j];
                float ox = __half2float(adj_h[0]);
                float oy = __half2float(adj_h[1]);
                float oz = __half2float(adj_h[2]);
                r_sq = fmaxf(r_sq, ox * ox + oy * oy + oz * oz);
            }

            if (r_sq > TAU_CONST_SQ * sigma_sq) {
                // Cell is large relative to sigma: weight at 1-hop < 0.1, stay constant
                mu = mu_ref;
            } else {
                float w_self = expf(-diff_self.squaredNorm() / sigma_sq);
                float w_sum = w_self + w_floor;
                float mu_w = (w_self + w_floor) * mu_ref;

                // 1-hop loop — record neighbor indices for 2-hop dedup
                uint32_t one_hop_ids[64];
                int n_one_hop = 0;
                bool skip_2hop = false;

                for (uint32_t j = a0; j < a1; ++j) {
                    uint32_t nb = point_adjacency[j];
                    float mu_nb = activated[nb];
                    Vec4h adj_h = adjacent_diff[j];
                    Vec3f offset(__half2float(adj_h[0]),
                                 __half2float(adj_h[1]),
                                 __half2float(adj_h[2]));
                    Vec3f diff_nb = diff_self - offset;
                    float dmu = mu_nb - mu_ref;
                    float w_nb = expf(-diff_nb.squaredNorm() / sigma_sq
                                      - dmu * dmu / sigma_v_sq);
                    w_sum += w_nb + w_floor;
                    mu_w += (w_nb + w_floor) * mu_nb;

                    if (n_one_hop < 64) {
                        one_hop_ids[n_one_hop++] = nb;
                    } else {
                        skip_2hop = true;
                    }
                }

                // 2-hop loop: only when cell is small enough relative to sigma
                if (r_sq <= TAU_2HOP_SQ * sigma_sq && !skip_2hop) {
                    for (int h = 0; h < n_one_hop; ++h) {
                        uint32_t nb1 = one_hop_ids[h];
                        uint32_t b0 = point_adjacency_offsets[nb1];
                        uint32_t b1 = point_adjacency_offsets[nb1 + 1];
                        for (uint32_t k = b0; k < b1; ++k) {
                            uint32_t nb2 = point_adjacency[k];
                            // Skip self and all 1-hop neighbors (strict dedup)
                            if (nb2 == point_idx) continue;
                            bool is_dup = false;
                            for (int d = 0; d < n_one_hop; ++d) {
                                if (one_hop_ids[d] == nb2) { is_dup = true; break; }
                            }
                            if (is_dup) continue;

                            float mu_nb2 = activated[nb2];
                            Vec3f p_nb2 = points[nb2];
                            Vec3f diff2 = x_mid - p_nb2;
                            float dmu2 = mu_nb2 - mu_ref;
                            float w2 = expf(-diff2.squaredNorm() / sigma_sq
                                           - dmu2 * dmu2 / sigma_v_sq);
                            w_sum += w2 + w_floor;
                            mu_w += (w2 + w_floor) * mu_nb2;
                        }
                    }
                }

                mu = fmaxf(0.0f, mu_w / fmaxf(w_sum, eps));
            }
        } else {
            mu = activated[point_idx];
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

            if (vis_settings.phong_enabled) {
                // Gradient via cell adjacency: least-squares style sum of
                // one-sided directional derivatives weighted by 1/dist²
                Vec3f grad = Vec3f::Zero();
                float mu_self = activated[point_idx];
                uint32_t ga0 = point_adjacency_offsets[point_idx];
                uint32_t ga1 = point_adjacency_offsets[point_idx + 1];
                for (uint32_t j = ga0; j < ga1; ++j) {
                    Vec4h adj_h = adjacent_diff[j];
                    Vec3f offset(__half2float(adj_h[0]),
                                 __half2float(adj_h[1]),
                                 __half2float(adj_h[2]));
                    float r2 = offset.squaredNorm();
                    if (r2 > 1e-12f) {
                        float dmu = activated[point_adjacency[j]] - mu_self;
                        grad += dmu * offset / r2;
                    }
                }

                float gn = grad.norm();
                float lighting = vis_settings.phong_ambient;
                if (gn > 1e-6f) {
                    // Fixed world-space light direction (upper-right-front)
                    const Vec3f light_dir =
                        Vec3f(1.0f, 1.0f, 1.0f) / sqrtf(3.0f);
                    Vec3f N = -grad / gn;
                    float NdotL = fmaxf(N.dot(light_dir), 0.0f);
                    // Blinn-Phong: H = (L + V).normalized()
                    Vec3f V = -ray.direction;
                    Vec3f H = (light_dir + V).normalized();
                    float NdotH = fmaxf(N.dot(H), 0.0f);
                    float spec = (NdotL > 0.0f)
                                     ? powf(NdotH, vis_settings.phong_shininess)
                                     : 0.0f;
                    lighting += vis_settings.phong_diffuse * NdotL
                              + vis_settings.phong_specular * spec;
                }
                rgb = rgb * lighting;
            }
        } else {
            // Original colormap path
            float v = fminf(mu * den_scale, 1.0f);
            rgb = colormap(v, cmap, cmap_table);
            alpha = 1.0f - expf(-mu * delta_t);
        }

        if (vis_settings.ao_enabled && alpha > 1e-2f && ao_directions != nullptr) {
            float t_mid = 0.5f * (tc_0 + tc_1);
            Vec3f x_sample = ray.origin + t_mid * ray.direction;
            float ao = compute_ao(point_idx, x_sample,
                                  points, activated, point_adjacency,
                                  point_adjacency_offsets, adjacent_diff,
                                  ao_directions, vis_settings.ao_num_dirs,
                                  vis_settings.ao_max_distance,
                                  use_tf, tf_density_min, tf_density_max,
                                  tf_opacity_scale, tf_table);
            rgb = rgb * (1.0f - vis_settings.ao_strength + vis_settings.ao_strength * ao);
        }

        float next_transmittance = transmittance * (1.0f - alpha);

        // Depth: find where transmittance crosses the quantile threshold
        if (!depth_quantile_passed && next_transmittance < depth_quantile) {
            depth_quantile_passed = true;
            if (mu > 1e-6f) {
                depth = tc_0 + logf(transmittance / depth_quantile) / mu;
            } else {
                depth = tc_0;
            }
        }

        color += transmittance * alpha * rgb;
        transmittance = next_transmittance;

        return transmittance > settings.weight_threshold && t_1 < t_exit;
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

// ---------------------------------------------------------------------------
// Linear-IDW visualization kernel
// Density at each segment midpoint is the normalized sum of 1/(d²+eps²)
// weights over the containing cell and all its Voronoi neighbours.
// eps² scales with cell radius to prevent the self-weight from dominating.
// Optional bilateral term (sigma_v > 0) edge-preserves across density jumps.
// ---------------------------------------------------------------------------

__global__ void ct_visualization_linear_idw(
        TraceSettings settings,
        VisualizationSettings vis_settings,
        Camera camera,
        CMapTable cmap_table,
        TransferFunctionTable tf_table,
        const Vec3f *__restrict__ points,
        const float *__restrict__ activated,
        const uint32_t *__restrict__ point_adjacency,
        const uint32_t *__restrict__ point_adjacency_offsets,
        const Vec4h *__restrict__ adjacent_diff,
        const float *__restrict__ cell_radius,
        const float *__restrict__ ao_directions,
        uint32_t start_index,
        CUsurfObject output_surface) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= camera.width || j >= camera.height)
        return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction /= ray.direction.norm();

    // Slab-test against the slicing AABB
    const Vec3f s_min = *vis_settings.slice_min;
    const Vec3f s_max = *vis_settings.slice_max;
    float aabb_t_enter = -1e38f, aabb_t_exit = 1e38f;
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        float dir = ray.direction[ax];
        float inv_d = 1.0f / (fabsf(dir) > 1e-20f ? dir : copysignf(1e-20f, dir));
        float t1 = (s_min[ax] - ray.origin[ax]) * inv_d;
        float t2 = (s_max[ax] - ray.origin[ax]) * inv_d;
        aabb_t_enter = fmaxf(aabb_t_enter, fminf(t1, t2));
        aabb_t_exit  = fminf(aabb_t_exit,  fmaxf(t1, t2));
    }
    const float t_enter = fmaxf(aabb_t_enter, 0.0f);
    const float t_exit  = aabb_t_exit;

    float den_scale    = vis_settings.density_scale;
    ColorMap cmap      = vis_settings.color_map;
    float depth_quantile = vis_settings.depth_quantile;

    Vec3f color = Vec3f::Zero();
    float transmittance = 1.0f;
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    bool use_tf         = vis_settings.use_transfer_function;
    float tf_density_min = vis_settings.tf_density_min;
    float tf_density_max = vis_settings.tf_density_max;
    float tf_opacity_scale = vis_settings.tf_opacity_scale;

    bool  use_bilateral = vis_settings.idw_sigma_v > 0.0f;
    float sigma_v_sq    = vis_settings.idw_sigma_v * vis_settings.idw_sigma_v;
    bool  use_preint    = vis_settings.use_preintegrated_tf
                          && vis_settings.preint_tf.data != nullptr;

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f & /*next_point*/) {
        float tc_0 = fmaxf(t_0, t_enter);
        float tc_1 = fminf(t_1, t_exit);
        if (tc_0 >= tc_1)
            return t_1 < t_exit;
        float delta_t = tc_1 - tc_0;
        float t_mid   = 0.5f * (tc_0 + tc_1);
        Vec3f x_mid   = ray.origin + t_mid * ray.direction;

        // Evaluate linear IDW at midpoint (always needed for colour/alpha)
        float mu = eval_linear_idw(x_mid, point_idx, points, activated,
                                   point_adjacency, point_adjacency_offsets,
                                   adjacent_diff, cell_radius,
                                   use_bilateral, sigma_v_sq);

        Vec3f rgb;
        float alpha;

        if (use_tf) {
            if (use_preint) {
                // Pre-integrated path: evaluate μ at segment entry and exit too
                Vec3f x_entry = ray.origin + tc_0 * ray.direction;
                Vec3f x_exit  = ray.origin + tc_1 * ray.direction;
                float mu_in  = eval_linear_idw(x_entry, point_idx, points, activated,
                                               point_adjacency, point_adjacency_offsets,
                                               adjacent_diff, cell_radius,
                                               use_bilateral, sigma_v_sq);
                float mu_out = eval_linear_idw(x_exit,  point_idx, points, activated,
                                               point_adjacency, point_adjacency_offsets,
                                               adjacent_diff, cell_radius,
                                               use_bilateral, sigma_v_sq);
                sample_preintegrated_tf(mu_in, mu_out, delta_t,
                                        tf_density_min, tf_density_max,
                                        vis_settings.preint_tf, rgb, alpha);
            } else {
                float range = tf_density_max - tf_density_min;
                float v = (range > 1e-8f)
                    ? fmaxf(0.0f, fminf((mu - tf_density_min) / range, 1.0f))
                    : 0.0f;
                float tf_opacity;
                sample_transfer_function(v, tf_table, rgb, tf_opacity);
                alpha = 1.0f - expf(-tf_opacity * tf_opacity_scale * delta_t);
            }

            if (vis_settings.phong_enabled) {
                Vec3f grad = Vec3f::Zero();
                float mu_self = activated[point_idx];
                uint32_t ga0 = point_adjacency_offsets[point_idx];
                uint32_t ga1 = point_adjacency_offsets[point_idx + 1];
                for (uint32_t k = ga0; k < ga1; ++k) {
                    Vec4h adj_h = adjacent_diff[k];
                    Vec3f offset(__half2float(adj_h[0]),
                                 __half2float(adj_h[1]),
                                 __half2float(adj_h[2]));
                    float r2 = offset.squaredNorm();
                    if (r2 > 1e-12f) {
                        float dmu = activated[point_adjacency[k]] - mu_self;
                        grad += dmu * offset / r2;
                    }
                }
                float gn = grad.norm();
                float lighting = vis_settings.phong_ambient;
                if (gn > 1e-6f) {
                    const Vec3f light_dir = Vec3f(1.0f, 1.0f, 1.0f) / sqrtf(3.0f);
                    Vec3f N = -grad / gn;
                    float NdotL = fmaxf(N.dot(light_dir), 0.0f);
                    Vec3f V = -ray.direction;
                    Vec3f H = (light_dir + V).normalized();
                    float NdotH = fmaxf(N.dot(H), 0.0f);
                    float spec = (NdotL > 0.0f)
                                     ? powf(NdotH, vis_settings.phong_shininess)
                                     : 0.0f;
                    lighting += vis_settings.phong_diffuse * NdotL
                              + vis_settings.phong_specular * spec;
                }
                rgb = rgb * lighting;
            }
        } else {
            float v = fminf(mu * den_scale, 1.0f);
            rgb = colormap(v, cmap, cmap_table);
            alpha = 1.0f - expf(-mu * delta_t);
        }

        if (vis_settings.ao_enabled && alpha > 1e-2f && ao_directions != nullptr) {
            Vec3f x_sample = ray.origin + t_mid * ray.direction;
            float ao = compute_ao(point_idx, x_sample,
                                  points, activated, point_adjacency,
                                  point_adjacency_offsets, adjacent_diff,
                                  ao_directions, vis_settings.ao_num_dirs,
                                  vis_settings.ao_max_distance,
                                  use_tf, tf_density_min, tf_density_max,
                                  tf_opacity_scale, tf_table);
            rgb = rgb * (1.0f - vis_settings.ao_strength + vis_settings.ao_strength * ao);
        }

        float next_transmittance = transmittance * (1.0f - alpha);
        if (!depth_quantile_passed && next_transmittance < depth_quantile) {
            depth_quantile_passed = true;
            if (mu > 1e-6f) {
                depth = tc_0 + logf(transmittance / depth_quantile) / mu;
            } else {
                depth = tc_0;
            }
        }
        // Pre-integrated TF returns premultiplied rgb (rgb already contains alpha).
        // All other paths return plain rgb and need the alpha factor here.
        if (use_preint && use_tf)
            color += transmittance * rgb;
        else
            color += transmittance * alpha * rgb;
        transmittance = next_transmittance;
        return transmittance > settings.weight_threshold && t_1 < t_exit;
    };

    uint32_t n = trace<128, 4>(ray,
                               points,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff,
                               start_index,
                               settings.max_intersections,
                               functor);

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

// ---------------------------------------------------------------------------
// Sibson natural-neighbor interpolation render kernel
// ---------------------------------------------------------------------------

__global__ void ct_visualization_sibson(
        TraceSettings settings,
        VisualizationSettings vis_settings,
        Camera camera,
        CMapTable cmap_table,
        TransferFunctionTable tf_table,
        const Vec3f *__restrict__ points,
        const float *__restrict__ activated,
        const uint32_t *__restrict__ point_adjacency,
        const uint32_t *__restrict__ point_adjacency_offsets,
        const Vec4h *__restrict__ adjacent_diff,
        const float *__restrict__ cell_radius,
        const float *__restrict__ ao_directions,
        uint32_t start_index,
        CUsurfObject output_surface) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= camera.width || j >= camera.height)
        return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction /= ray.direction.norm();

    const Vec3f s_min = *vis_settings.slice_min;
    const Vec3f s_max = *vis_settings.slice_max;
    float aabb_t_enter = -1e38f, aabb_t_exit = 1e38f;
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        float dir = ray.direction[ax];
        float inv_d = 1.0f / (fabsf(dir) > 1e-20f ? dir : copysignf(1e-20f, dir));
        float t1 = (s_min[ax] - ray.origin[ax]) * inv_d;
        float t2 = (s_max[ax] - ray.origin[ax]) * inv_d;
        aabb_t_enter = fmaxf(aabb_t_enter, fminf(t1, t2));
        aabb_t_exit  = fminf(aabb_t_exit,  fmaxf(t1, t2));
    }
    const float t_enter = fmaxf(aabb_t_enter, 0.0f);
    const float t_exit  = aabb_t_exit;

    float den_scale      = vis_settings.density_scale;
    ColorMap cmap        = vis_settings.color_map;
    float depth_quantile = vis_settings.depth_quantile;

    Vec3f color = Vec3f::Zero();
    float transmittance = 1.0f;
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    bool  use_tf          = vis_settings.use_transfer_function;
    float tf_density_min  = vis_settings.tf_density_min;
    float tf_density_max  = vis_settings.tf_density_max;
    float tf_opacity_scale = vis_settings.tf_opacity_scale;

    bool  use_bilateral   = vis_settings.sibson_sigma_v > 0.0f;
    float sigma_v_sq      = vis_settings.sibson_sigma_v * vis_settings.sibson_sigma_v;
    bool  use_preint      = vis_settings.use_preintegrated_tf
                            && vis_settings.preint_tf.data != nullptr;
    uint32_t k_samples    = vis_settings.sibson_k_samples;
    float radius_scale    = vis_settings.sibson_radius_scale;

    // Per-pixel seed component mixed with point_idx per-segment inside the functor.
    uint32_t pixel_seed = wang_hash((uint32_t)j * (uint32_t)camera.width + (uint32_t)i);

    auto functor = [&](uint32_t point_idx,
                       float t_0,
                       float t_1,
                       const Vec3f &current_point,
                       const Vec3f & /*next_point*/) {
        float tc_0 = fmaxf(t_0, t_enter);
        float tc_1 = fminf(t_1, t_exit);
        if (tc_0 >= tc_1)
            return t_1 < t_exit;
        float delta_t = tc_1 - tc_0;
        float t_mid   = 0.5f * (tc_0 + tc_1);
        Vec3f x_mid   = ray.origin + t_mid * ray.direction;

        // Seed shared across all three eval positions in this segment so
        // (mu_in, mu_out) are correlated → pre-integrated TF stays smooth.
        uint32_t seg_seed = pixel_seed ^ wang_hash(point_idx);

        float mu = eval_sibson(x_mid, point_idx, points, activated,
                               point_adjacency, point_adjacency_offsets,
                               adjacent_diff, cell_radius,
                               seg_seed, k_samples, radius_scale,
                               use_bilateral, sigma_v_sq);

        Vec3f rgb;
        float alpha;

        if (use_tf) {
            if (use_preint) {
                Vec3f x_entry = ray.origin + tc_0 * ray.direction;
                Vec3f x_exit  = ray.origin + tc_1 * ray.direction;
                float mu_in  = eval_sibson(x_entry, point_idx, points, activated,
                                           point_adjacency, point_adjacency_offsets,
                                           adjacent_diff, cell_radius,
                                           seg_seed, k_samples, radius_scale,
                                           use_bilateral, sigma_v_sq);
                float mu_out = eval_sibson(x_exit, point_idx, points, activated,
                                           point_adjacency, point_adjacency_offsets,
                                           adjacent_diff, cell_radius,
                                           seg_seed, k_samples, radius_scale,
                                           use_bilateral, sigma_v_sq);
                sample_preintegrated_tf(mu_in, mu_out, delta_t,
                                        tf_density_min, tf_density_max,
                                        vis_settings.preint_tf, rgb, alpha);
            } else {
                float range = tf_density_max - tf_density_min;
                float v = (range > 1e-8f)
                    ? fmaxf(0.0f, fminf((mu - tf_density_min) / range, 1.0f))
                    : 0.0f;
                float tf_opacity;
                sample_transfer_function(v, tf_table, rgb, tf_opacity);
                alpha = 1.0f - expf(-tf_opacity * tf_opacity_scale * delta_t);
            }

            if (vis_settings.phong_enabled) {
                Vec3f grad = Vec3f::Zero();
                float mu_self = activated[point_idx];
                uint32_t ga0 = point_adjacency_offsets[point_idx];
                uint32_t ga1 = point_adjacency_offsets[point_idx + 1];
                for (uint32_t k = ga0; k < ga1; ++k) {
                    Vec4h adj_h = adjacent_diff[k];
                    Vec3f offset(__half2float(adj_h[0]),
                                 __half2float(adj_h[1]),
                                 __half2float(adj_h[2]));
                    float r2 = offset.squaredNorm();
                    if (r2 > 1e-12f) {
                        float dmu = activated[point_adjacency[k]] - mu_self;
                        grad += dmu * offset / r2;
                    }
                }
                float gn = grad.norm();
                float lighting = vis_settings.phong_ambient;
                if (gn > 1e-6f) {
                    const Vec3f light_dir = Vec3f(1.0f, 1.0f, 1.0f) / sqrtf(3.0f);
                    Vec3f N = -grad / gn;
                    float NdotL = fmaxf(N.dot(light_dir), 0.0f);
                    Vec3f V = -ray.direction;
                    Vec3f H = (light_dir + V).normalized();
                    float NdotH = fmaxf(N.dot(H), 0.0f);
                    float spec = (NdotL > 0.0f)
                                     ? powf(NdotH, vis_settings.phong_shininess)
                                     : 0.0f;
                    lighting += vis_settings.phong_diffuse * NdotL
                              + vis_settings.phong_specular * spec;
                }
                rgb = rgb * lighting;
            }
        } else {
            float v = fminf(mu * den_scale, 1.0f);
            rgb = colormap(v, cmap, cmap_table);
            alpha = 1.0f - expf(-mu * delta_t);
        }

        if (vis_settings.ao_enabled && alpha > 1e-2f && ao_directions != nullptr) {
            Vec3f x_sample = ray.origin + t_mid * ray.direction;
            float ao = compute_ao(point_idx, x_sample,
                                  points, activated, point_adjacency,
                                  point_adjacency_offsets, adjacent_diff,
                                  ao_directions, vis_settings.ao_num_dirs,
                                  vis_settings.ao_max_distance,
                                  use_tf, tf_density_min, tf_density_max,
                                  tf_opacity_scale, tf_table);
            rgb = rgb * (1.0f - vis_settings.ao_strength + vis_settings.ao_strength * ao);
        }

        float next_transmittance = transmittance * (1.0f - alpha);
        if (!depth_quantile_passed && next_transmittance < depth_quantile) {
            depth_quantile_passed = true;
            if (mu > 1e-6f) {
                depth = tc_0 + logf(transmittance / depth_quantile) / mu;
            } else {
                depth = tc_0;
            }
        }
        // Pre-integrated TF returns premultiplied rgb (rgb already contains alpha).
        // All other paths return plain rgb and need the alpha factor here.
        if (use_preint && use_tf)
            color += transmittance * rgb;
        else
            color += transmittance * alpha * rgb;
        transmittance = next_transmittance;
        return transmittance > settings.weight_threshold && t_1 < t_exit;
    };

    uint32_t n = trace<128, 4>(ray,
                               points,
                               point_adjacency,
                               point_adjacency_offsets,
                               adjacent_diff,
                               start_index,
                               settings.max_intersections,
                               functor);

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

// ---------------------------------------------------------------------------
// Volume DVR kernel — trilinear-sampled voxel grid, same TF/Phong/slice as foam
// ---------------------------------------------------------------------------

__global__ void volume_visualization(TraceSettings trace_settings,
                                     VisualizationSettings vis_settings,
                                     int num_steps,
                                     float voxel_eps,
                                     Camera camera,
                                     int x_offset,
                                     CMapTable cmap_table,
                                     TransferFunctionTable tf_table,
                                     cudaTextureObject_t vol_tex,
                                     CUsurfObject output_surface) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= camera.width || j >= camera.height)
        return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction = ray.direction.normalized();

    // Slab test against slice AABB (same as foam)
    const Vec3f s_min = *vis_settings.slice_min;
    const Vec3f s_max = *vis_settings.slice_max;
    float t_enter = -1e38f, t_exit = 1e38f;
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        float inv_d = 1.0f / ray.direction[ax];
        float t1 = (s_min[ax] - ray.origin[ax]) * inv_d;
        float t2 = (s_max[ax] - ray.origin[ax]) * inv_d;
        t_enter = fmaxf(t_enter, fminf(t1, t2));
        t_exit  = fminf(t_exit,  fmaxf(t1, t2));
    }
    t_enter = fmaxf(t_enter, 0.0f);

    Vec3f color = Vec3f::Zero();
    float transmittance = 1.0f;
    float depth = 0.0f;
    bool depth_quantile_passed = false;
    uint32_t steps_taken = 0;

    bool use_tf = vis_settings.use_transfer_function;
    float tf_density_min = vis_settings.tf_density_min;
    float tf_density_max = vis_settings.tf_density_max;
    float tf_opacity_scale = vis_settings.tf_opacity_scale;
    float den_scale = vis_settings.density_scale;
    float depth_quantile = vis_settings.depth_quantile;
    ColorMap cmap = vis_settings.color_map;

    if (t_exit > t_enter) {
        float dt = (t_exit - t_enter) / (float)num_steps;

        for (int k = 0; k < num_steps && transmittance > trace_settings.weight_threshold; ++k) {
            float t = t_enter + (k + 0.5f) * dt;
            Vec3f x = ray.origin + t * ray.direction;

            // Map world [-1,1]^3 → texture [0,1]:
            //   volume shape (X, Y, Z) stored C-contiguous →
            //   cudaArray extent (width=Z, height=Y, depth=X)
            //   tex3D(u, v, w): u→Z, v→Y, w→X
            float u = (x.z() + 1.0f) * 0.5f;
            float v = (x.y() + 1.0f) * 0.5f;
            float w = (x.x() + 1.0f) * 0.5f;

            float mu = tex3D<float>(vol_tex, u, v, w) * vis_settings.activation_scale;
            steps_taken = (uint32_t)(k + 1);

            Vec3f rgb;
            float alpha;

            if (use_tf) {
                float range = tf_density_max - tf_density_min;
                float val = (range > 1e-8f)
                    ? fmaxf(0.0f, fminf((mu - tf_density_min) / range, 1.0f))
                    : 0.0f;
                float tf_opacity;
                sample_transfer_function(val, tf_table, rgb, tf_opacity);
                alpha = 1.0f - expf(-tf_opacity * tf_opacity_scale * dt);

                if (vis_settings.phong_enabled) {
                    // Central-difference gradient in world space.
                    // u→Z, v→Y, w→X: each finite-diff axis maps to a world axis.
                    float gz = tex3D<float>(vol_tex, u + voxel_eps, v, w)
                             - tex3D<float>(vol_tex, u - voxel_eps, v, w);
                    float gy = tex3D<float>(vol_tex, u, v + voxel_eps, w)
                             - tex3D<float>(vol_tex, u, v - voxel_eps, w);
                    float gx = tex3D<float>(vol_tex, u, v, w + voxel_eps)
                             - tex3D<float>(vol_tex, u, v, w - voxel_eps);
                    Vec3f grad(gx, gy, gz);

                    float gn = grad.norm();
                    float lighting = vis_settings.phong_ambient;
                    if (gn > 1e-6f) {
                        const Vec3f light_dir =
                            Vec3f(1.0f, 1.0f, 1.0f) / sqrtf(3.0f);
                        Vec3f N = -grad / gn;
                        float NdotL = fmaxf(N.dot(light_dir), 0.0f);
                        Vec3f V = -ray.direction;
                        Vec3f H = (light_dir + V).normalized();
                        float NdotH = fmaxf(N.dot(H), 0.0f);
                        float spec = (NdotL > 0.0f)
                            ? powf(NdotH, vis_settings.phong_shininess)
                            : 0.0f;
                        lighting += vis_settings.phong_diffuse * NdotL
                                  + vis_settings.phong_specular * spec;
                    }
                    rgb = rgb * lighting;
                }
            } else {
                float val = fminf(mu * den_scale, 1.0f);
                rgb = colormap(val, cmap, cmap_table);
                alpha = 1.0f - expf(-mu * dt);
            }

            float next_transmittance = transmittance * (1.0f - alpha);

            if (!depth_quantile_passed && next_transmittance < depth_quantile) {
                depth_quantile_passed = true;
                depth = (mu > 1e-6f)
                    ? t + logf(transmittance / depth_quantile) / mu
                    : t;
            }

            color += transmittance * alpha * rgb;
            transmittance = next_transmittance;
        }
    }

    // Mode switch — mirrors ct_visualization output section
    Vec3f out;
    switch (vis_settings.mode) {
    case VolumeDensity:
    case RGB: {
        Vec3f bg = *vis_settings.bg_color;
        if (vis_settings.checker_bg) {
            int ci = (i + x_offset) / 16;
            int cj = j / 16;
            bg = ((ci + cj) % 2 == 0) ? Vec3f(0.8f, 0.8f, 0.8f)
                                       : Vec3f(0.6f, 0.6f, 0.6f);
        }
        out = color + transmittance * bg;
        break;
    }
    case Depth: {
        float val = depth / vis_settings.max_depth;
        out = colormap(fminf(fmaxf(val, 0.0f), 1.0f), cmap, cmap_table);
        break;
    }
    case Alpha: {
        float opacity = 1.0f - transmittance;
        out = Vec3f(opacity, opacity, opacity);
        break;
    }
    case Intersections: {
        float val = (steps_taken > 1) ? (float)(steps_taken - 1) / (float)num_steps : 0.0f;
        out = colormap(fminf(fmaxf(val, 0.0f), 1.0f), cmap, cmap_table);
        break;
    }
    default:
        out = Vec3f::Zero();
        break;
    }

    uint32_t rgba = make_rgba8(out[0], out[1], out[2], 1.0f);
    surf2Dwrite(rgba, output_surface, (i + x_offset) * 4, j);
}

void launch_volume_visualization(const TraceSettings &trace_settings,
                                 const VisualizationSettings &vis_settings,
                                 int num_steps,
                                 float voxel_eps,
                                 const Camera &camera,
                                 int x_offset,
                                 CMapTable cmap_table,
                                 TransferFunctionTable tf_table,
                                 uint64_t vol_tex_handle,
                                 uint64_t output_surface_handle,
                                 const void *stream) {
    CUstream cu_stream = stream ? *reinterpret_cast<const CUstream *>(stream) : 0;

    dim3 block(16, 16);
    dim3 grid((camera.width + block.x - 1) / block.x,
              (camera.height + block.y - 1) / block.y);

    volume_visualization<<<grid, block, 0, cu_stream>>>(
        trace_settings,
        vis_settings,
        num_steps,
        voxel_eps,
        camera,
        x_offset,
        cmap_table,
        tf_table,
        static_cast<cudaTextureObject_t>(vol_tex_handle),
        static_cast<CUsurfObject>(output_surface_handle));
}

// ---------------------------------------------------------------------------
// Per-cell radius from adjacent_diff (sqrt of max squared edge length)
// ---------------------------------------------------------------------------
__global__ void compute_cell_radius_kernel(
    uint32_t num_points,
    const Vec4h *__restrict__ adjacent_diff,
    const uint32_t *__restrict__ adj_offsets,
    float *__restrict__ cell_radius) {

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;
    uint32_t a0 = adj_offsets[i], a1 = adj_offsets[i + 1];
    float r_sq = 1e-12f;
    for (uint32_t j = a0; j < a1; ++j) {
        Vec4h h = adjacent_diff[j];
        float ox = __half2float(h[0]);
        float oy = __half2float(h[1]);
        float oz = __half2float(h[2]);
        r_sq = fmaxf(r_sq, ox * ox + oy * oy + oz * oz);
    }
    cell_radius[i] = sqrtf(r_sq);
}

void compute_cell_radius(uint32_t num_points,
                         const Vec4h *adjacent_diff,
                         const uint32_t *adj_offsets,
                         float *cell_radius,
                         const void *stream) {
    CUstream cu_stream = static_cast<CUstream>(const_cast<void *>(stream));
    launch_kernel_1d<128>(compute_cell_radius_kernel, num_points, cu_stream,
                          num_points, adjacent_diff, adj_offsets, cell_radius);
}

// ---------------------------------------------------------------------------
// One iteration of bilateral Jacobi smoothing on the CSR Voronoi graph
// ---------------------------------------------------------------------------
// mu_out[i] = (mu_in[i] + alpha * sum_j w_ij * mu_in[j]) / (1 + alpha * sum_j w_ij)
// w_ij = exp(-d_ij^2 / (sigma_s_scale * cell_radius[i])^2
//          - (mu_in[j] - mu_in[i])^2 / sigma_v^2)
__global__ void smooth_density_graph_step(
    uint32_t num_points,
    const float *__restrict__ mu_in,
    float *__restrict__ mu_out,
    const uint32_t *__restrict__ adj,
    const uint32_t *__restrict__ adj_off,
    const Vec4h *__restrict__ adjacent_diff,
    const float *__restrict__ cell_radius,
    float alpha,
    float sigma_v_sq,
    float sigma_s_scale) {

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float mu_i = mu_in[i];
    float sigma_s = sigma_s_scale * fmaxf(cell_radius[i], 1e-8f);
    float sigma_s_sq = sigma_s * sigma_s;

    uint32_t a0 = adj_off[i], a1 = adj_off[i + 1];
    float w_sum = 0.f, w_mu = 0.f;
    for (uint32_t j = a0; j < a1; ++j) {
        Vec4h h = adjacent_diff[j];
        float ox = __half2float(h[0]);
        float oy = __half2float(h[1]);
        float oz = __half2float(h[2]);
        float d_sq = ox * ox + oy * oy + oz * oz;
        float mu_j = mu_in[adj[j]];
        float dmu = mu_j - mu_i;
        float w = expf(-d_sq / sigma_s_sq - dmu * dmu / sigma_v_sq);
        w_sum += w;
        w_mu += w * mu_j;
    }
    mu_out[i] = (mu_i + alpha * w_mu) / (1.f + alpha * w_sum);
}

// ---------------------------------------------------------------------------
// Per-vertex normal precompute for smooth Phong in bary mode.
// One thread per tet. Accumulates volume-weighted analytic tet gradient
// (∇μ_tet = Σ μ_k n_k / det) into vertex_normal[di[k]] via atomicAdd.
// Buffer must be pre-zeroed. Don't normalize — interpolate raw, then normalize
// at sample time (magnitude carries density-gradient information).
// ---------------------------------------------------------------------------
__global__ void compute_vertex_normals_kernel(
    uint32_t num_tets,
    const uint32_t *__restrict__ tets,
    const uint32_t *__restrict__ perm,
    const Vec3f *__restrict__ points,
    const float *__restrict__ activated,
    Vec3f *__restrict__ vertex_normal)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tets) return;

    uint32_t di[4];
    Vec3f v[4];
    for (int k = 0; k < 4; k++) {
        di[k] = perm[tets[tid * 4 + k]];
        v[k]  = points[di[k]];
    }

    Vec3f e1 = v[1] - v[0], e2 = v[2] - v[0], e3 = v[3] - v[0];
    Vec3f n1 = e2.cross(e3), n2 = e3.cross(e1), n3 = e1.cross(e2);
    Vec3f n0 = -(n1 + n2 + n3);
    float det = e1.dot(n1);
    if (fabsf(det) < 1e-12f) return;

    float inv_det = 1.0f / det;
    float mu[4];
    mu[0] = activated[di[0]]; mu[1] = activated[di[1]];
    mu[2] = activated[di[2]]; mu[3] = activated[di[3]];
    Vec3f ns[4];
    ns[0] = n0; ns[1] = n1; ns[2] = n2; ns[3] = n3;

    Vec3f g_tet = Vec3f::Zero();
    for (int k = 0; k < 4; k++) g_tet += mu[k] * ns[k];
    g_tet *= inv_det;

    float vol    = fabsf(det) / 6.0f;
    Vec3f contrib = g_tet * vol;

    float *vn = reinterpret_cast<float *>(vertex_normal);
    for (int k = 0; k < 4; k++) {
        atomicAdd(&vn[di[k] * 3 + 0], contrib[0]);
        atomicAdd(&vn[di[k] * 3 + 1], contrib[1]);
        atomicAdd(&vn[di[k] * 3 + 2], contrib[2]);
    }
}

// ---------------------------------------------------------------------------
// Barycentric tet visualization kernel
// ---------------------------------------------------------------------------
__global__ void ct_bary_visualization(TraceSettings settings,
                                       VisualizationSettings vis_settings,
                                       Camera camera,
                                       CMapTable cmap_table,
                                       TransferFunctionTable tf_table,
                                       const Vec3f *__restrict__ points,
                                       const float *__restrict__ activated,
                                       const uint32_t *__restrict__ point_adjacency,
                                       const uint32_t *__restrict__ point_adjacency_offsets,
                                       const Vec4h *__restrict__ adjacent_diff,
                                       const float *__restrict__ ao_directions,
                                       const uint32_t *__restrict__ tets,
                                       const uint32_t *__restrict__ tet_adj,
                                       const uint32_t *__restrict__ perm,
                                       const Vec3f *__restrict__ vertex_normal,
                                       uint32_t start_tet,
                                       CUsurfObject output_surface) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= camera.width || j >= camera.height) return;

    Ray ray = cast_ray(camera, i, j);
    ray.direction /= ray.direction.norm();

    const Vec3f s_min = *vis_settings.slice_min;
    const Vec3f s_max = *vis_settings.slice_max;
    float aabb_t_enter = -1e38f, aabb_t_exit = 1e38f;
#pragma unroll
    for (int ax = 0; ax < 3; ++ax) {
        float d = ray.direction[ax];
        float inv_d = 1.0f / (fabsf(d) > 1e-20f ? d : copysignf(1e-20f, d));
        float t1 = (s_min[ax] - ray.origin[ax]) * inv_d;
        float t2 = (s_max[ax] - ray.origin[ax]) * inv_d;
        aabb_t_enter = fmaxf(aabb_t_enter, fminf(t1, t2));
        aabb_t_exit  = fminf(aabb_t_exit,  fmaxf(t1, t2));
    }
    const float t_enter = fmaxf(aabb_t_enter, 0.0f);
    const float t_exit  = aabb_t_exit;

    if (t_enter >= t_exit) {
        Vec3f bg = *vis_settings.bg_color;
        surf2Dwrite(make_rgba8(bg[0], bg[1], bg[2], 1.0f),
                    output_surface, i * 4, j);
        return;
    }

    float den_scale      = vis_settings.density_scale;
    ColorMap cmap        = vis_settings.color_map;
    float depth_quantile = vis_settings.depth_quantile;
    bool use_tf          = vis_settings.use_transfer_function;
    float tf_density_min  = vis_settings.tf_density_min;
    float tf_density_max  = vis_settings.tf_density_max;
    float tf_opacity_scale = vis_settings.tf_opacity_scale;
    float sigma_v_intra  = vis_settings.sigma_v_intra;
    float sigv_sq        = sigma_v_intra * sigma_v_intra + 1e-12f;

    Vec3f color = Vec3f::Zero();
    float transmittance = 1.0f;
    float depth = 0.0f;
    bool depth_quantile_passed = false;

    bool sliver_failed = false;
    uint32_t n = trace_tet(
        ray, points, tets, tet_adj, perm,
        start_tet, settings.max_intersections,
        t_enter, t_exit,
        [&](const Vec3f v[4], const uint32_t di[4],
            const float L[4], const float dL[4], float t_0, float t_1) -> bool {

            float delta_t = t_1 - t_0;
            float mu4[4];
            for (int k = 0; k < 4; k++) mu4[k] = activated[di[k]];

            if (use_tf) {
                // Per-vertex TF lookup → interpolate colour in colour-space.
                // Ensures C⁰ rendered colour at tet faces: shared face vertices have the
                // same global density → same TF result → Lc-blended colour is C⁰ at faces.
                // (Interpolating mu then mapping through TF would give C⁰ density but
                // visible colour seams with steep TFs; colour-space interpolation avoids this.)
                float range = tf_density_max - tf_density_min;
                Vec3f rgb_v[4];
                float op_v[4];
                for (int k = 0; k < 4; k++) {
                    float vv = (range > 1e-8f)
                        ? fmaxf(0.0f, fminf((mu4[k] - tf_density_min) / range, 1.0f))
                        : 0.0f;
                    sample_transfer_function(vv, tf_table, rgb_v[k], op_v[k]);
                }

                float t_mid = 0.5f * (t_0 + t_1);
                float Lm[4];
                for (int k = 0; k < 4; k++) Lm[k] = L[k] + dL[k] * (t_mid - t_0);
                float Lc[4], lc_sum = 0.f;
                for (int k = 0; k < 4; k++) { Lc[k] = fmaxf(Lm[k], 0.f); lc_sum += Lc[k]; }
                if (lc_sum > 1e-10f) { for (int k = 0; k < 4; k++) Lc[k] /= lc_sum; }
                else { for (int k = 0; k < 4; k++) Lc[k] = 0.25f; }

                Vec3f rgb = Vec3f::Zero();
                float opacity_mid = 0.f;
                for (int k = 0; k < 4; k++) {
                    rgb += Lc[k] * rgb_v[k];
                    opacity_mid += Lc[k] * op_v[k];
                }
                float alpha = 1.0f - expf(-opacity_mid * tf_opacity_scale * delta_t);

                if (vis_settings.phong_enabled && vertex_normal != nullptr) {
                    Vec3f N_raw = Vec3f::Zero();
                    for (int k = 0; k < 4; k++) N_raw += Lc[k] * vertex_normal[di[k]];
                    float Nn = N_raw.norm();
                    float lighting = vis_settings.phong_ambient;
                    if (Nn > 1e-6f) {
                        const Vec3f light_dir = Vec3f(1.0f, 1.0f, 1.0f) / sqrtf(3.0f);
                        Vec3f N = -N_raw / Nn;
                        float NdotL = fmaxf(N.dot(light_dir), 0.0f);
                        Vec3f V = -ray.direction;
                        Vec3f H = (light_dir + V).normalized();
                        float NdotH = fmaxf(N.dot(H), 0.0f);
                        float spec = (NdotL > 0.0f)
                            ? powf(NdotH, vis_settings.phong_shininess) : 0.0f;
                        lighting += vis_settings.phong_diffuse * NdotL
                                  + vis_settings.phong_specular * spec;
                    }
                    rgb = rgb * lighting;
                }

                if (vis_settings.ao_enabled && alpha > 1e-2f && ao_directions != nullptr) {
                    int k_max = 0;
                    for (int k = 1; k < 4; k++) if (Lc[k] > Lc[k_max]) k_max = k;
                    Vec3f x_sample = ray.origin + t_mid * ray.direction;
                    float ao = compute_ao(di[k_max], x_sample,
                                          points, activated,
                                          point_adjacency, point_adjacency_offsets,
                                          adjacent_diff, ao_directions,
                                          vis_settings.ao_num_dirs,
                                          vis_settings.ao_max_distance,
                                          use_tf, tf_density_min, tf_density_max,
                                          tf_opacity_scale, tf_table);
                    rgb = rgb * (1.0f - vis_settings.ao_strength
                                 + vis_settings.ao_strength * ao);
                }

                float next_T = transmittance * (1.0f - alpha);
                if (!depth_quantile_passed && next_T < depth_quantile) {
                    depth_quantile_passed = true;
                    float eff_op = opacity_mid * tf_opacity_scale;
                    depth = (eff_op > 1e-6f)
                        ? t_0 + logf(transmittance / depth_quantile) / eff_op
                        : t_0;
                }
                color += transmittance * alpha * rgb;
                transmittance = next_T;
            } else {
                // Colormap mode: single midpoint sample (TF quantization not an issue here).
                float t_mid = 0.5f * (t_0 + t_1);
                float Lm[4];
                for (int k = 0; k < 4; k++) Lm[k] = L[k] + dL[k] * (t_mid - t_0);
                float Lc[4], lc_sum = 0.f;
                for (int k = 0; k < 4; k++) { Lc[k] = fmaxf(Lm[k], 0.f); lc_sum += Lc[k]; }
                if (lc_sum > 1e-10f) { for (int k = 0; k < 4; k++) Lc[k] /= lc_sum; }
                else { for (int k = 0; k < 4; k++) Lc[k] = 0.25f; }

                float mu_mid = 0.f;
                for (int k = 0; k < 4; k++) mu_mid += Lc[k] * mu4[k];
                if (sigma_v_intra > 0.f) {
                    float mu_ref = mu_mid, w_sum = 0.f, w_mu = 0.f;
                    for (int k = 0; k < 4; k++) {
                        float dmu = mu4[k] - mu_ref;
                        float wk = Lc[k] * expf(-dmu * dmu / sigv_sq);
                        w_sum += wk; w_mu += wk * mu4[k];
                    }
                    if (w_sum > 1e-6f) mu_mid = w_mu / w_sum;
                }
                mu_mid = fmaxf(0.f, mu_mid);

                float vv = fminf(mu_mid * den_scale, 1.0f);
                Vec3f rgb = colormap(vv, cmap, cmap_table);
                float alpha = 1.0f - expf(-mu_mid * delta_t);

                if (vis_settings.ao_enabled && alpha > 1e-2f && ao_directions != nullptr) {
                    int k_max = 0;
                    for (int k = 1; k < 4; k++) if (Lc[k] > Lc[k_max]) k_max = k;
                    Vec3f x_sample = ray.origin + t_mid * ray.direction;
                    float ao = compute_ao(di[k_max], x_sample,
                                          points, activated,
                                          point_adjacency, point_adjacency_offsets,
                                          adjacent_diff, ao_directions,
                                          vis_settings.ao_num_dirs,
                                          vis_settings.ao_max_distance,
                                          use_tf, tf_density_min, tf_density_max,
                                          tf_opacity_scale, tf_table);
                    rgb = rgb * (1.0f - vis_settings.ao_strength
                                 + vis_settings.ao_strength * ao);
                }

                float next_T = transmittance * (1.0f - alpha);
                if (!depth_quantile_passed && next_T < depth_quantile) {
                    depth_quantile_passed = true;
                    depth = (mu_mid > 1e-6f)
                        ? t_0 + logf(transmittance / depth_quantile) / mu_mid
                        : t_0;
                }
                color += transmittance * alpha * rgb;
                transmittance = next_T;
            }
            return transmittance > settings.weight_threshold && t_1 < t_exit;
        },
        &sliver_failed);

    Vec3f out;
    switch (vis_settings.mode) {
    case VolumeDensity:
    case RGB: {
        Vec3f bg = *vis_settings.bg_color;
        if (vis_settings.checker_bg) {
            int ci = i / 16, cj = j / 16;
            bg = ((ci + cj) % 2 == 0) ? Vec3f(0.8f, 0.8f, 0.8f)
                                       : Vec3f(0.6f, 0.6f, 0.6f);
        }
        out = color + transmittance * bg;
        break;
    }
    case Depth: {
        float val = fminf(fmaxf(depth / vis_settings.max_depth, 0.0f), 1.0f);
        out = colormap(val, cmap, cmap_table);
        break;
    }
    case Alpha:
        out = Vec3f(1.0f - transmittance, 1.0f - transmittance,
                    1.0f - transmittance);
        break;
    case Intersections: {
        float val = (n > 1)
            ? float(n - 1) / float(settings.max_intersections) : 0.0f;
        val = fminf(fmaxf(val, 0.0f), 1.0f);
        out = colormap(val, cmap, cmap_table);
        break;
    }
    default:
        out = Vec3f::Zero();
        break;
    }

    if (sliver_failed) out = Vec3f(1.0f, 0.0f, 1.0f); // debug: unrecoverable sliver

    surf2Dwrite(make_rgba8(out[0], out[1], out[2], 1.0f),
                output_surface, i * 4, j);
}

class CUDADensityPipeline : public Pipeline {
  public:
    CUDAArray<float> smooth_scratch;
    CUDAArray<float> vertex_normal_buffer; // 3 floats per point (Vec3f), bary+phong only

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
                       float *point_contribution,
                       uint32_t *point_hit_count = nullptr,
                       const float *cell_radius = nullptr,
                       const float *density_peak = nullptr,
                       const float *delta_raw = nullptr,
                       const float *cov_raw = nullptr,
                       const float *density_delta = nullptr,
                       const float *quaternions = nullptr,
                       const float *texel_sites_2d = nullptr,
                       const float *texel_heights = nullptr,
                       // LC64 plan v3 Commit 2A -- independent-side raw
                       // logits.  Only read when
                       // settings.thin_surface_independent_mode is true;
                       // nullptr for legacy / absolute / relative.
                       const float *raw_plus = nullptr,
                       const float *raw_minus = nullptr) override {

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               cell_radius,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        // LC64 plan v3 Commit 2A -- independent mode takes precedence
        // over the legacy thin-surface branch when the discriminator is
        // on.  Both modes share the surface geometry (quaternion +
        // texel sites + heights + cell_radius); only the per-side
        // density parameterization differs.  The Python binding rejects
        // mixed inputs before this point so raw_plus/raw_minus are
        // guaranteed non-null under independent mode.
        if (settings.thin_surface_mode && settings.thin_surface_independent_mode &&
            raw_plus && raw_minus && quaternions &&
            texel_sites_2d && texel_heights && cell_radius) {
            launch_kernel_1d<block_size>(
                ct_independent_forward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                raw_plus,
                raw_minus,
                quaternions,
                texel_sites_2d,
                texel_heights,
                cell_radius,
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                rays,
                num_rays,
                start_point_index,
                ray_projection,
                num_intersections,
                point_contribution,
                point_hit_count);
        } else if (settings.thin_surface_mode && density_delta && quaternions &&
            texel_sites_2d && texel_heights && cell_radius) {
            launch_kernel_1d<block_size>(
                ct_thinsurface_forward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                density,
                density_delta,
                quaternions,
                texel_sites_2d,
                texel_heights,
                cell_radius,
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                rays,
                num_rays,
                start_point_index,
                ray_projection,
                num_intersections,
                point_contribution,
                point_hit_count);
        } else if (settings.gaussian_mode && density_peak && delta_raw && cov_raw && cell_radius) {
            launch_kernel_1d<block_size>(
                ct_gaussian_forward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                density,
                density_peak,
                delta_raw,
                cov_raw,
                cell_radius,
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                rays,
                num_rays,
                start_point_index,
                ray_projection,
                num_intersections,
                point_contribution,
                point_hit_count);
        } else if (settings.interpolation_mode) {
            CUDAArray<float> activated(num_points);
            launch_kernel_1d<256>(precompute_activated_density,
                                  num_points,
                                  nullptr,
                                  density,
                                  activated.begin(),
                                  (float *)nullptr,
                                  num_points);

            launch_kernel_1d<block_size>(
                ct_interp_forward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                activated.begin(),
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                cell_radius,
                rays,
                num_rays,
                start_point_index,
                ray_projection,
                num_intersections,
                point_contribution,
                point_hit_count);
        } else {
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
                point_contribution,
                point_hit_count);
        }
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
                        float *point_error,
                        const float *cell_radius = nullptr,
                        const float *density_peak = nullptr,
                        const float *delta_raw = nullptr,
                        const float *cov_raw = nullptr,
                        float *density_peak_grad = nullptr,
                        float *delta_raw_grad = nullptr,
                        float *cov_raw_grad = nullptr,
                        const float *density_delta = nullptr,
                        const float *quaternions = nullptr,
                        const float *texel_sites_2d = nullptr,
                        const float *texel_heights = nullptr,
                        float *density_delta_grad = nullptr,
                        float *quaternions_grad = nullptr,
                        float *texel_sites_2d_grad = nullptr,
                        float *texel_heights_grad = nullptr,
                        // LC64 plan v3 Commit 2A -- independent-side raw
                        // logits.  Only read under forward; backward is
                        // explicitly NOT implemented in this commit and the
                        // binding raises before reaching this point.
                        const float *raw_plus = nullptr,
                        const float *raw_minus = nullptr) override {

        // LC64 plan v3 Commit 2A -- backward under independent mode is
        // explicitly out of scope.  The Python binding rejects this
        // earlier with a NotImplementedError, but guard again here so a
        // direct C++ caller (e.g. a custom test driver) cannot silently
        // fall through to the legacy backward path and produce a
        // silently-wrong gradient for raw_plus/raw_minus.
        if (settings.thin_surface_independent_mode) {
            throw std::runtime_error(
                "CUDADensityPipeline::trace_backward: independent-side "
                "backward is not implemented in LC64 plan v3 Commit 2A. "
                "It will land in Commit 2B; until then, calling backward "
                "under thin_surface_independent_mode is a hard error.");
        }

        CUDAArray<Vec4h> adjacent_diff(point_adjacency_size + 32);
        prefetch_adjacent_diff(reinterpret_cast<const Vec3f *>(points),
                               num_points,
                               point_adjacency_size,
                               point_adjacency,
                               point_adjacency_offsets,
                               cell_radius,
                               adjacent_diff.begin(),
                               nullptr);

        constexpr uint32_t block_size = 128;
        if (settings.thin_surface_mode && density_delta && quaternions &&
            texel_sites_2d && texel_heights && cell_radius) {
            launch_kernel_1d<block_size>(
                ct_thinsurface_backward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                density,
                density_delta,
                quaternions,
                texel_sites_2d,
                texel_heights,
                cell_radius,
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
                density_delta_grad,
                quaternions_grad,
                texel_sites_2d_grad,
                texel_heights_grad,
                point_error);
        } else if (settings.gaussian_mode && density_peak && delta_raw && cov_raw && cell_radius) {
            launch_kernel_1d<block_size>(
                ct_gaussian_backward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                density,
                density_peak,
                delta_raw,
                cov_raw,
                cell_radius,
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
                density_peak_grad,
                delta_raw_grad,
                cov_raw_grad,
                point_error);
        } else if (settings.interpolation_mode) {
            CUDAArray<float> activated(num_points);
            CUDAArray<float> dsigmoid_buf(num_points);
            launch_kernel_1d<256>(precompute_activated_density,
                                  num_points,
                                  nullptr,
                                  density,
                                  activated.begin(),
                                  dsigmoid_buf.begin(),
                                  num_points);

            launch_kernel_1d<block_size>(
                ct_interp_backward<block_size>,
                num_rays,
                nullptr,
                settings,
                points,
                activated.begin(),
                dsigmoid_buf.begin(),
                point_adjacency,
                point_adjacency_offsets,
                adjacent_diff.begin(),
                cell_radius,
                rays,
                num_rays,
                start_point_index,
                ray_projection_grad,
                ray_error,
                points_grad,
                density_scalar_grad,
                point_error);
        } else {
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
                             float *activated,
                             uint32_t start_index,
                             uint64_t output_surface,
                             const float *ao_directions = nullptr,
                             const uint32_t *tets = nullptr,
                             const uint32_t *tet_adjacency = nullptr,
                             const uint32_t *permutation = nullptr,
                             uint32_t start_tet = 0,
                             const float *cell_radius = nullptr,
                             const void *stream = nullptr) override {

        CUstream cu_stream = 0;
        if (stream) {
            cu_stream = *reinterpret_cast<const CUstream *>(stream);
        }

        // Step 1: activate softplus
        constexpr uint32_t block_size_1d = 128;
        launch_kernel_1d<block_size_1d>(
            precompute_activated_density_vis,
            num_points,
            stream,
            reinterpret_cast<const float *>(attributes),
            activated,
            num_points,
            vis_settings.activation_beta,
            vis_settings.activation_scale);

        // Step 2: bilateral Jacobi pre-smoothing (universal, all modes)
        if (vis_settings.smooth_T > 0 && cell_radius != nullptr) {
            smooth_scratch.resize(num_points);
            float *in_buf  = activated;
            float *out_buf = smooth_scratch.begin();
            float sigma_v_sq = vis_settings.smooth_sigma_v * vis_settings.smooth_sigma_v;
            for (int t = 0; t < vis_settings.smooth_T; t++) {
                launch_kernel_1d<block_size_1d>(
                    smooth_density_graph_step,
                    num_points,
                    stream,
                    num_points,
                    in_buf,
                    out_buf,
                    reinterpret_cast<const uint32_t *>(point_adjacency),
                    reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
                    reinterpret_cast<const Vec4h *>(adjacent_points),
                    cell_radius,
                    vis_settings.smooth_alpha,
                    sigma_v_sq,
                    vis_settings.smooth_sigma_s_scale);
                float *tmp = in_buf; in_buf = out_buf; out_buf = tmp;
            }
            // Ensure result is in activated[]
            if (in_buf != activated) {
                cuda_check(cuMemcpyDtoDAsync((CUdeviceptr)activated,
                                             (CUdeviceptr)in_buf,
                                             num_points * sizeof(float),
                                             cu_stream));
            }
        }

        dim3 block(16, 16);
        dim3 grid((camera.width + block.x - 1) / block.x,
                  (camera.height + block.y - 1) / block.y);

        // Step 3: dispatch renderer based on interpolation mode
        if (vis_settings.interpolation_mode == InterpolationMode::Sibson
            && cell_radius != nullptr) {
            ct_visualization_sibson<<<grid, block, 0, cu_stream>>>(
                settings,
                vis_settings,
                camera,
                cmap_table,
                tf_table,
                reinterpret_cast<const Vec3f *>(points),
                activated,
                reinterpret_cast<const uint32_t *>(point_adjacency),
                reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
                reinterpret_cast<const Vec4h *>(adjacent_points),
                cell_radius,
                ao_directions,
                start_index,
                static_cast<CUsurfObject>(output_surface));
        } else if (vis_settings.interpolation_mode == InterpolationMode::LinearIDW) {
            ct_visualization_linear_idw<<<grid, block, 0, cu_stream>>>(
                settings,
                vis_settings,
                camera,
                cmap_table,
                tf_table,
                reinterpret_cast<const Vec3f *>(points),
                activated,
                reinterpret_cast<const uint32_t *>(point_adjacency),
                reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
                reinterpret_cast<const Vec4h *>(adjacent_points),
                cell_radius,
                ao_directions,
                start_index,
                static_cast<CUsurfObject>(output_surface));
        } else if (vis_settings.interpolation_mode == InterpolationMode::BarycentricTet
            && tets != nullptr && tet_adjacency != nullptr && permutation != nullptr) {

            // Precompute per-vertex normals for smooth Phong (only when phong is active).
            Vec3f *vn_ptr = nullptr;
            if (vis_settings.phong_enabled && num_tets > 0) {
                vertex_normal_buffer.resize(num_points * 3);
                cuda_check(cuMemsetD32Async(
                    (CUdeviceptr)vertex_normal_buffer.begin(), 0,
                    num_points * 3, cu_stream));
                constexpr uint32_t vn_block = 256;
                compute_vertex_normals_kernel<<<
                    (num_tets + vn_block - 1) / vn_block, vn_block, 0, cu_stream>>>(
                    num_tets,
                    tets,
                    permutation,
                    reinterpret_cast<const Vec3f *>(points),
                    activated,
                    reinterpret_cast<Vec3f *>(vertex_normal_buffer.begin()));
                vn_ptr = reinterpret_cast<Vec3f *>(vertex_normal_buffer.begin());
            }

            ct_bary_visualization<<<grid, block, 0, cu_stream>>>(
                settings,
                vis_settings,
                camera,
                cmap_table,
                tf_table,
                reinterpret_cast<const Vec3f *>(points),
                activated,
                reinterpret_cast<const uint32_t *>(point_adjacency),
                reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
                reinterpret_cast<const Vec4h *>(adjacent_points),
                ao_directions,
                tets,
                tet_adjacency,
                permutation,
                vn_ptr,
                start_tet,
                static_cast<CUsurfObject>(output_surface));
        } else {
            ct_visualization<<<grid, block, 0, cu_stream>>>(
                settings,
                vis_settings,
                camera,
                cmap_table,
                tf_table,
                reinterpret_cast<const Vec3f *>(points),
                activated,
                reinterpret_cast<const uint32_t *>(point_adjacency),
                reinterpret_cast<const uint32_t *>(point_adjacency_offsets),
                reinterpret_cast<const Vec4h *>(adjacent_points),
                ao_directions,
                start_index,
                static_cast<CUsurfObject>(output_surface));
        }
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
