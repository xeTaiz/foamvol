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
                                     float *__restrict__ point_contribution) {

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
                                   float *__restrict__ point_contribution) {

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
                       float *point_contribution,
                       const float *cell_radius = nullptr,
                       const float *density_peak = nullptr,
                       const float *delta_raw = nullptr,
                       const float *cov_raw = nullptr) override {

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
        if (settings.gaussian_mode && density_peak && delta_raw && cov_raw && cell_radius) {
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
                point_contribution);
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
                point_contribution);
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
                point_contribution);
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
                        float *cov_raw_grad = nullptr) override {

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
        if (settings.gaussian_mode && density_peak && delta_raw && cov_raw && cell_radius) {
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
