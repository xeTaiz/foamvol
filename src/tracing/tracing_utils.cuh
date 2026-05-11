#pragma once

#include "../utils/geometry.h"
#include "camera.h"

namespace radfoam {

template <int block_size, int chunk_size, typename CellFunctor>
__forceinline__ __device__ uint32_t
trace(const Ray &ray,
      const Vec3f *__restrict__ points,
      const uint32_t *__restrict__ point_adjacency,
      const uint32_t *__restrict__ point_adjacency_offsets,
      const Vec4h *__restrict__ adjacent_points,
      uint32_t start_point,
      uint32_t max_steps,
      CellFunctor cell_functor) {
    float t_0 = 0.0f;
    uint32_t n = 0;

    uint32_t current_point_idx = start_point;
    Vec3f primal_point = points[current_point_idx];

    for (;;) {
        n++;
        if (n > max_steps) {
            break;
        }

        // Outer loop iterates through Voronoi cells
        uint32_t point_adjacency_begin =
            point_adjacency_offsets[current_point_idx];
        uint32_t point_adjacency_end =
            point_adjacency_offsets[current_point_idx + 1];

        uint32_t num_faces = point_adjacency_end - point_adjacency_begin;
        float t_1 = std::numeric_limits<float>::infinity();

        uint32_t next_face = UINT32_MAX;
        Vec3f next_point = Vec3f::Zero();

        half2 chunk[chunk_size * 2];
        for (uint32_t i = 0; i < num_faces; i += chunk_size) {
#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                chunk[2 * j] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[0];
                chunk[2 * j + 1] = reinterpret_cast<const half2 *>(
                    adjacent_points + point_adjacency_begin + i + j)[1];
            }

#pragma unroll
            for (uint32_t j = 0; j < chunk_size; ++j) {
                Vec3f offset(__half2float(chunk[2 * j].x),
                             __half2float(chunk[2 * j].y),
                             __half2float(chunk[2 * j + 1].x));
                Vec3f face_origin = primal_point + offset / 2.0f;
                Vec3f face_normal = offset;
                float dp = face_normal.dot(ray.direction);
                float t = (face_origin - ray.origin).dot(face_normal) / dp;

                if (dp > 0.0f && t < t_1 && (i + j) < num_faces) {
                    t_1 = t;
                    next_face = i + j;
                }
            }
        }

        if (next_face == UINT32_MAX) {
            break;
        }

        uint32_t next_point_idx =
            point_adjacency[point_adjacency_begin + next_face];
        next_point = points[next_point_idx];

        if (t_1 > t_0) {
            if (!cell_functor(
                    current_point_idx, t_0, t_1, primal_point, next_point)) {
                break;
            }
        }
        t_0 = fmaxf(t_0, t_1);
        current_point_idx = next_point_idx;
        primal_point = next_point;
    }

    return n;
}

// ---------------------------------------------------------------------------
// Delaunay tet walker for barycentric interpolation
// ---------------------------------------------------------------------------
// Functor signature: (v[4] vertex positions, lam_mid[4] barycentrics at segment
// midpoint, t_0, t_1) -> bool (true = continue marching).
// The 4 vertex world positions are passed so the functor can look up densities.
// tet_adj is a flat (num_tets * 4) array; entry [t*4 + k] is the packed
// adjacency 4*neighbour_tet + face_slot_in_neighbour, or UINT32_MAX for boundary.
// permutation maps triangulation-internal vertex indices to original point indices
// (i.e. points[permutation[v]] gives the world position of tet vertex v).

template <typename TetFunctor>
__forceinline__ __device__ uint32_t
trace_tet(const Ray &ray,
          const Vec3f *__restrict__ points,
          const uint32_t *__restrict__ tets,        // (T,4) flat uint32
          const uint32_t *__restrict__ tet_adj,     // (T,4) flat uint32 packed
          const uint32_t *__restrict__ permutation, // new→orig
          uint32_t start_tet,
          uint32_t max_steps,
          float t_enter,
          float t_exit,
          TetFunctor functor,
          bool *failed_out = nullptr) {

    uint32_t cur_tet = start_tet;

    // Compute barycentrics of x in tet v[4]. Returns false for degenerate tets.
    // Uses scale-aware threshold (1e-18) — fine for seek convergence.
    auto compute_bary = [&](const Vec3f v[4], const Vec3f &x, float L[4]) -> bool {
        Vec3f e1 = v[1] - v[0];
        Vec3f e2 = v[2] - v[0];
        Vec3f e3 = v[3] - v[0];
        Vec3f n1 = e2.cross(e3);
        Vec3f n2 = e3.cross(e1);
        Vec3f n3 = e1.cross(e2);
        float det = e1.dot(n1);
        float scale_sq = e1.squaredNorm() + e2.squaredNorm() + e3.squaredNorm();
        if (det * det < 1e-18f * scale_sq * scale_sq * scale_sq) return false;
        float inv_det = 1.0f / det;
        Vec3f rhs = x - v[0];
        L[1] = n1.dot(rhs) * inv_det;
        L[2] = n2.dot(rhs) * inv_det;
        L[3] = n3.dot(rhs) * inv_det;
        L[0] = -(n1 + n2 + n3).dot(x - v[1]) * inv_det;
        return true;
    };

    // Worst-barycentric walker shared by initial seek and integration recovery.
    // Walks up to max_iters steps from `start` toward x.
    // Returns the containing tet, or UINT32_MAX on failure.
    auto find_containing_tet = [&](const Vec3f &x,
                                   uint32_t start,
                                   int max_iters) -> uint32_t {
        uint32_t cur = start;
        uint32_t prev = UINT32_MAX;
        for (int s = 0; s < max_iters; s++) {
            Vec3f vv[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                vv[k] = points[permutation[tets[cur * 4 + k]]];
            float L[4];
            if (!compute_bary(vv, x, L)) {
                // Sliver mid-walk: try any non-prev, non-hull face.
                uint32_t hop = UINT32_MAX;
                for (int kk = 0; kk < 4; kk++) {
                    uint32_t packed = tet_adj[cur * 4 + kk];
                    if (packed == UINT32_MAX) continue;
                    uint32_t cand = packed >> 2;
                    if (cand == prev) continue;
                    hop = cand;
                    break;
                }
                if (hop == UINT32_MAX) return UINT32_MAX;
                prev = cur; cur = hop;
                continue;
            }
            int worst = 0;
            for (int k = 1; k < 4; k++) if (L[k] < L[worst]) worst = k;
            if (L[worst] >= -1e-5f) return cur;
            uint32_t packed = tet_adj[cur * 4 + worst];
            if (packed == UINT32_MAX) return UINT32_MAX;
            uint32_t nxt = packed >> 2;
            if (nxt == prev) {
                if (L[worst] > -0.1f) return cur;
                return UINT32_MAX;
            }
            prev = cur; cur = nxt;
        }
        return UINT32_MAX;
    };

    // Initial seek: locate the tet containing the ray entry point.
    {
        Vec3f x_enter = ray.origin + t_enter * ray.direction;
        uint32_t found = find_containing_tet(x_enter, cur_tet, 256);
        if (found == UINT32_MAX) return 0;
        cur_tet = found;
    }

    constexpr float SLIVER_EPS = 1e-4f;
    constexpr float L_TOL = 1e-3f;

    float t_0 = t_enter;
    uint32_t n = 0;

    while (n < max_steps && t_0 < t_exit) {
        Vec3f v[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
            v[k] = points[permutation[tets[cur_tet * 4 + k]]];

        Vec3f e1 = v[1] - v[0];
        Vec3f e2 = v[2] - v[0];
        Vec3f e3 = v[3] - v[0];
        Vec3f n1 = e2.cross(e3);
        Vec3f n2 = e3.cross(e1);
        Vec3f n3 = e1.cross(e2);
        float det = e1.dot(n1);
        float scale_sq = e1.squaredNorm() + e2.squaredNorm() + e3.squaredNorm();

        Vec3f x_now = ray.origin + t_0 * ray.direction;

        // (1) Sliver: inv_det would be unreliable; skip this tet.
        if (det * det < SLIVER_EPS * scale_sq * scale_sq * scale_sq) {
            t_0 = fminf(t_0 + 1e-4f * sqrtf(scale_sq), t_exit);
            x_now = ray.origin + t_0 * ray.direction;
            uint32_t found = find_containing_tet(x_now, cur_tet, 32);
            if (found == UINT32_MAX) { if (failed_out) *failed_out = true; break; }
            cur_tet = found;
            n++;
            continue;
        }

        float inv_det = 1.0f / det;

        float dL[4];
        dL[1] = n1.dot(ray.direction) * inv_det;
        dL[2] = n2.dot(ray.direction) * inv_det;
        dL[3] = n3.dot(ray.direction) * inv_det;
        dL[0] = -(dL[1] + dL[2] + dL[3]);

        Vec3f rhs = x_now - v[0];
        float L[4];
        L[1] = n1.dot(rhs) * inv_det;
        L[2] = n2.dot(rhs) * inv_det;
        L[3] = n3.dot(rhs) * inv_det;
        L[0] = -(n1 + n2 + n3).dot(rhs - e1) * inv_det;

        // (2) Stranded: upstream noise sent us into a tet we're not inside.
        float lmin = L[0];
        for (int k = 1; k < 4; k++) if (L[k] < lmin) lmin = L[k];
        if (lmin < -L_TOL) {
            uint32_t found = find_containing_tet(x_now, cur_tet, 32);
            if (found == UINT32_MAX) { if (failed_out) *failed_out = true; break; }
            cur_tet = found;
            n++;
            continue;
        }

        // Find exit: smallest t strictly > t_0 where some L[k] hits 0.
        float t_1 = t_exit;
        int exit_face = -1;
        for (int k = 0; k < 4; k++) {
            if (dL[k] < 0.f) {
                float tk = t_0 - L[k] / dL[k];
                if (tk > t_0 && tk < t_1) {
                    t_1 = tk;
                    exit_face = k;
                }
            }
        }

        float dt = t_1 - t_0;
        if (dt > 1e-10f) {
            uint32_t di[4];
            for (int k = 0; k < 4; k++) di[k] = permutation[tets[cur_tet * 4 + k]];
            if (!functor(v, di, L, dL, t_0, t_1)) break;
        }

        n++;
        t_0 = t_1;

        if (exit_face < 0) break;
        uint32_t packed = tet_adj[cur_tet * 4 + exit_face];
        if (packed == UINT32_MAX) break; // boundary
        cur_tet = packed >> 2;
    }

    return n;
}

__forceinline__ __device__ Vec3f cell_intersection_grad(
    const Vec3f &primal_point, const Vec3f &opposite_point, const Ray &ray) {
    Vec3f face_origin = (primal_point + opposite_point) / 2.0f;
    Vec3f face_normal = (opposite_point - primal_point);

    float num = (face_origin - ray.origin).dot(face_normal);
    float dp = face_normal.dot(ray.direction);

    Vec3f grad = num * ray.direction + dp * (ray.origin - primal_point);
    grad /= dp * dp;

    return grad;
}

inline RADFOAM_HD uint32_t make_rgba8(float r, float g, float b, float a) {
    r = std::max(0.0f, std::min(1.0f, r));
    g = std::max(0.0f, std::min(1.0f, g));
    b = std::max(0.0f, std::min(1.0f, b));
    a = std::max(0.0f, std::min(1.0f, a));
    int ri = static_cast<int>(r * 255.0f);
    int gi = static_cast<int>(g * 255.0f);
    int bi = static_cast<int>(b * 255.0f);
    int ai = static_cast<int>(a * 255.0f);
    return (ai << 24) | (bi << 16) | (gi << 8) | ri;
}

// Evaluate linear-IDW (w = 1/(d²+eps²)) over the CSR neighbor set of the
// containing cell.  eps² is scaled to 5% of the cell radius to prevent the
// self-weight from dominating when the query lies near the Voronoi site.
__device__ inline float eval_linear_idw(
    const Vec3f &x_query,
    uint32_t point_idx,
    const Vec3f *__restrict__ points,
    const float *__restrict__ activated,
    const uint32_t *__restrict__ point_adjacency,
    const uint32_t *__restrict__ point_adjacency_offsets,
    const Vec4h *__restrict__ adjacent_diff,
    const float *__restrict__ cell_radius,
    bool use_bilateral,
    float sigma_v_sq)
{
    float r_self  = cell_radius ? cell_radius[point_idx] : 1e-3f;
    float eps_frac = 0.05f * r_self;
    float eps_sq   = eps_frac * eps_frac;

    float mu_ref  = activated[point_idx];
    Vec3f diff_self = x_query - points[point_idx];
    float w_self  = 1.0f / (diff_self.squaredNorm() + eps_sq);
    float w_sum   = w_self;
    float mu_w    = w_self * mu_ref;

    uint32_t a0 = point_adjacency_offsets[point_idx];
    uint32_t a1 = point_adjacency_offsets[point_idx + 1];
    for (uint32_t j = a0; j < a1; ++j) {
        uint32_t nb = point_adjacency[j];
        float mu_nb = activated[nb];
        Vec4h adj_h = adjacent_diff[j];
        Vec3f offset(__half2float(adj_h[0]),
                     __half2float(adj_h[1]),
                     __half2float(adj_h[2]));
        Vec3f diff_nb = diff_self - offset;
        float w_nb = 1.0f / (diff_nb.squaredNorm() + eps_sq);
        if (use_bilateral) {
            float dmu = mu_nb - mu_ref;
            w_nb *= expf(-dmu * dmu / sigma_v_sq);
        }
        w_sum += w_nb;
        mu_w  += w_nb * mu_nb;
    }
    return fmaxf(0.0f, mu_w / fmaxf(w_sum, 1e-7f));
}

// Bilinear lookup into a pre-integrated TF table.
// mu_in, mu_out are clamped to [mu_min, mu_max] before indexing.
// The LUT was computed for a reference segment length of 1.0; for an actual
// segment of length delta_t, opacity is scaled as Beer-Lambert:
//   alpha_scaled = 1 - (1 - alpha_lut)^delta_t
// RGB is scaled proportionally to preserve emitted-radiance-per-unit-opacity.
__device__ inline void sample_preintegrated_tf(
    float mu_in, float mu_out, float delta_t,
    float mu_min, float mu_max,
    const PreIntegratedTFTable &lut,
    Vec3f &rgb_out, float &alpha_out)
{
    float range = mu_max - mu_min;
    float u = (range > 1e-8f) ? (mu_in  - mu_min) / range : 0.0f;
    float v = (range > 1e-8f) ? (mu_out - mu_min) / range : 0.0f;
    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));

    int n  = lut.size;
    float ui = u * (n - 1);
    float vi = v * (n - 1);
    int u0 = (int)ui, u1 = min(u0 + 1, n - 1);
    int v0 = (int)vi, v1 = min(v0 + 1, n - 1);
    float tu = ui - u0, tv = vi - v0;

    const float *d = lut.data;
    // LUT row = mu_in index, col = mu_out index; 4 floats (RGBA) per entry.
    const float *c00 = d + (u0 * n + v0) * 4;
    const float *c01 = d + (u0 * n + v1) * 4;
    const float *c10 = d + (u1 * n + v0) * 4;
    const float *c11 = d + (u1 * n + v1) * 4;
    float w00 = (1.0f - tu) * (1.0f - tv);
    float w01 = (1.0f - tu) * tv;
    float w10 = tu * (1.0f - tv);
    float w11 = tu * tv;

    float r = c00[0]*w00 + c01[0]*w01 + c10[0]*w10 + c11[0]*w11;
    float g = c00[1]*w00 + c01[1]*w01 + c10[1]*w10 + c11[1]*w11;
    float b = c00[2]*w00 + c01[2]*w01 + c10[2]*w10 + c11[2]*w11;
    float a = fmaxf(0.0f, fminf(1.0f, c00[3]*w00 + c01[3]*w01 + c10[3]*w10 + c11[3]*w11));

    // Δt scaling: LUT was built for d_ref = 1.0.
    float alpha_scaled = 1.0f - powf(fmaxf(1.0f - a, 1e-7f), delta_t);
    float scale = alpha_scaled / fmaxf(a, 1e-6f);
    rgb_out   = Vec3f(r * scale, g * scale, b * scale);
    alpha_out = alpha_scaled;
}

inline __device__ void sample_transfer_function(
    float v, const TransferFunctionTable &tf_table,
    Vec3f &rgb_out, float &alpha_out)
{
    int len = tf_table.size;
    const float *data = tf_table.data;
    v = fmaxf(0.0f, fminf(v, 1.0f));
    int i0 = static_cast<int>(v * (len - 1));
    int i1 = i0 + 1;
    float t = v * (len - 1) - i0;
    i0 = max(0, min(i0, len - 1));
    i1 = max(0, min(i1, len - 1));
    rgb_out = Vec3f(
        data[i0*4+0]*(1-t) + data[i1*4+0]*t,
        data[i0*4+1]*(1-t) + data[i1*4+1]*t,
        data[i0*4+2]*(1-t) + data[i1*4+2]*t);
    alpha_out = data[i0*4+3]*(1-t) + data[i1*4+3]*t;
}

inline __device__ Vec3f colormap(float v,
                                 ColorMap map,
                                 const CMapTable &cmap_table) {
    int map_len = cmap_table.sizes[map];
    const Vec3f *map_vals =
        reinterpret_cast<const Vec3f *>(cmap_table.data[map]);

    int i0 = static_cast<int>(v * (map_len - 1));
    int i1 = i0 + 1;
    float t = v * (map_len - 1) - i0;
    i0 = max(0, min(i0, map_len - 1));
    i1 = max(0, min(i1, map_len - 1));
    return map_vals[i0] * (1.0f - t) + map_vals[i1] * t;
}

} // namespace radfoam