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
          const uint32_t *__restrict__ tets,       // (T,4) flat uint32
          const uint32_t *__restrict__ tet_adj,    // (T,4) flat uint32 packed
          const uint32_t *__restrict__ permutation, // new→orig
          uint32_t start_tet,
          uint32_t max_steps,
          float t_enter,
          float t_exit,
          TetFunctor functor) {

    uint32_t cur_tet = start_tet;

    // Helper: compute barycentrics of point x in tet, return in L[4].
    // L[1..3] from solving [e1|e2|e3]*L123 = x-v0, L[0]=1-sum.
    // det is the signed volume * 6; returns false if tet is degenerate.
    auto compute_bary = [&](const Vec3f v[4], const Vec3f &x, float L[4]) -> bool {
        Vec3f e1 = v[1] - v[0];
        Vec3f e2 = v[2] - v[0];
        Vec3f e3 = v[3] - v[0];
        Vec3f n1 = e2.cross(e3);
        Vec3f n2 = e3.cross(e1);
        Vec3f n3 = e1.cross(e2);
        float det = e1.dot(n1);
        if (fabsf(det) < 1e-30f) return false;
        float inv_det = 1.0f / det;
        Vec3f rhs = x - v[0];
        L[1] = n1.dot(rhs) * inv_det;
        L[2] = n2.dot(rhs) * inv_det;
        L[3] = n3.dot(rhs) * inv_det;
        L[0] = 1.0f - L[1] - L[2] - L[3];
        return true;
    };

    // Seek phase: walk from start_tet to the tet containing (ray.origin + t_enter*ray.dir).
    constexpr int MAX_SEEK = 64;
    {
        Vec3f x_enter = ray.origin + t_enter * ray.direction;
        for (int s = 0; s < MAX_SEEK; s++) {
            Vec3f v[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                v[k] = points[permutation[tets[cur_tet * 4 + k]]];
            float L[4];
            if (!compute_bary(v, x_enter, L)) break;
            int worst = 0;
            for (int k = 1; k < 4; k++) if (L[k] < L[worst]) worst = k;
            if (L[worst] >= -1e-5f) break; // contained
            uint32_t packed = tet_adj[cur_tet * 4 + worst];
            if (packed == UINT32_MAX) break; // boundary
            cur_tet = packed >> 2;
        }
    }

    // Integration phase: march segment-by-segment through tets.
    float t_0 = t_enter;
    uint32_t n = 0;

    while (n < max_steps && t_0 < t_exit) {
        Vec3f v[4];
#pragma unroll
        for (int k = 0; k < 4; k++)
            v[k] = points[permutation[tets[cur_tet * 4 + k]]];

        // Compute dL/dt and L at t_0 (one 3x3 inverse via cross products).
        Vec3f e1 = v[1] - v[0];
        Vec3f e2 = v[2] - v[0];
        Vec3f e3 = v[3] - v[0];
        Vec3f n1 = e2.cross(e3);
        Vec3f n2 = e3.cross(e1);
        Vec3f n3 = e1.cross(e2);
        float det = e1.dot(n1);
        if (fabsf(det) < 1e-30f) break;
        float inv_det = 1.0f / det;

        float dL[4];
        dL[1] = n1.dot(ray.direction) * inv_det;
        dL[2] = n2.dot(ray.direction) * inv_det;
        dL[3] = n3.dot(ray.direction) * inv_det;
        dL[0] = -(dL[1] + dL[2] + dL[3]);

        Vec3f rhs = (ray.origin + t_0 * ray.direction) - v[0];
        float L[4];
        L[1] = n1.dot(rhs) * inv_det;
        L[2] = n2.dot(rhs) * inv_det;
        L[3] = n3.dot(rhs) * inv_det;
        L[0] = 1.0f - L[1] - L[2] - L[3];

        // Find exit: smallest t > t_0 where some L[k] hits 0.
        float t_1 = t_exit;
        int exit_face = -1;
        for (int k = 0; k < 4; k++) {
            if (dL[k] < -1e-10f) {
                float tk = t_0 - L[k] / dL[k];
                if (tk > t_0 - 1e-8f && tk < t_1) {
                    t_1 = tk;
                    exit_face = k;
                }
            }
        }

        float dt = t_1 - t_0;
        if (dt > 1e-10f) {
            float t_mid = 0.5f * (t_0 + t_1);
            float Lm[4];
            for (int k = 0; k < 4; k++) Lm[k] = L[k] + dL[k] * (t_mid - t_0);
            // Compute density indices (permutation[tets[cur_tet*4+k]]) for this tet
            uint32_t di[4];
            for (int k = 0; k < 4; k++) di[k] = permutation[tets[cur_tet * 4 + k]];
            if (!functor(v, di, Lm, t_0, t_1)) break;
        }

        n++;
        t_0 = t_1;

        if (exit_face < 0) break; // capped at t_exit
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