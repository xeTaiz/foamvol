#pragma once

#include <memory>

#include "../utils/typing.h"
#include "camera.h"

namespace radfoam {

struct TraceSettings {
    float weight_threshold;
    uint32_t max_intersections;
    float gradient_max_slope = 5.0f;
    bool interpolation_mode = false;
    float idw_sigma = 0.01f;
    float idw_sigma_v = 0.1f;
    bool per_cell_sigma = false;
    bool per_neighbor_sigma = false;
    bool gaussian_mode = false;
    bool thin_surface_mode = false;
    int  thin_K = 4;
    float thin_temp = 10.0f;
    float thin_height_eps = 1e-4f;
    // Relative-delta parameterization (M5 chest rescue prototype).
    // When true, the kernel computes delta = rho * mu_bar * tanh(raw_delta)
    // instead of using density_delta as a raw additive offset, so the split
    // is bounded by rho * mu_bar and both sides remain nonneg.  Geometry
    // is untouched.  See CH5-CH7 in specs/SPLIT-CELL-EXECUTION-LOG.md.
    bool  thin_surface_relative_delta = false;
    float thin_surface_delta_max_frac = 0.5f;   // rho in (0, 1]
    // LC64 plan v3 Commit 2A -- independent-side density mode. When
    // true, the kernel reads raw_plus / raw_minus (each (N,1)) and
    // computes mu_plus = activation_scale * softplus(raw_plus, beta=10)
    // and mu_minus = activation_scale * softplus(raw_minus, beta=10)
    // independently (no mu_bar / delta).  The thin-surface geometry
    // (quaternion, K texel sites, heights, cell_radius) and the
    // crossing/non-crossing / dp-sign semantics are unchanged from
    // the legacy absolute/relative branches -- only the per-side
    // density parameterization is replaced.  Mutually exclusive with
    // thin_surface_relative_delta at the scene level (enforced in
    // Python); the kernel sees only the flag below.
    bool  thin_surface_independent_mode = false;
    float thin_surface_activation_scale = 1.0f;  // multiplies softplus; 1.0 = legacy
};

inline TraceSettings default_trace_settings() {
    TraceSettings settings;
    settings.weight_threshold = 0.01f;
    settings.max_intersections = 8192;
    return settings;
}

enum VisualizationMode {
    RGB = 0,
    Depth = 1,
    Alpha = 2,
    Intersections = 3,
    VolumeDensity = 4,
};

enum InterpolationMode {
    PerCell = 0,
    IDW = 1,
    BarycentricTet = 2,
    LinearIDW = 3,
    Sibson = 4,
};

struct VisualizationSettings {
    VisualizationMode mode;
    ColorMap color_map;
    CVec3f bg_color;
    bool checker_bg;
    float max_depth;
    float depth_quantile;
    float density_scale;
    float activation_beta;
    float activation_scale;
    bool use_transfer_function;
    bool use_preintegrated_tf;
    PreIntegratedTFTable preint_tf;
    float tf_density_min;
    float tf_density_max;
    float tf_opacity_scale;
    InterpolationMode interpolation_mode;
    float idw_sigma;
    float idw_sigma_v;
    float sigma_v_intra;
    uint32_t sibson_k_samples;
    float sibson_radius_scale;
    float sibson_sigma_v;
    int smooth_T;
    float smooth_alpha;
    float smooth_sigma_v;
    float smooth_sigma_s_scale;
    bool phong_enabled;
    float phong_ambient;
    float phong_diffuse;
    float phong_specular;
    float phong_shininess;
    bool ao_enabled;
    uint32_t ao_num_dirs;
    float ao_max_distance;
    float ao_strength;
    CVec3f slice_min;
    CVec3f slice_max;
};

inline VisualizationSettings default_visualization_settings() {
    VisualizationSettings settings;
    settings.mode = VolumeDensity;
    settings.color_map = Gray;
    settings.bg_color = Vec3f(0.0f, 0.0f, 0.0f);
    settings.checker_bg = false;
    settings.max_depth = 10.0f;
    settings.depth_quantile = 0.5f;
    settings.density_scale = 1.0f;
    settings.activation_beta = 10.0f;
    settings.activation_scale = 1.0f;
    settings.use_transfer_function = false;
    settings.use_preintegrated_tf = false;
    settings.preint_tf = {nullptr, 0};
    settings.tf_density_min = 0.0f;
    settings.tf_density_max = 1.0f;
    settings.tf_opacity_scale = 100.0f;
    settings.interpolation_mode = PerCell;
    settings.idw_sigma = 0.01f;
    settings.idw_sigma_v = 0.05f;
    settings.sigma_v_intra = 0.0f;
    settings.sibson_k_samples = 16;
    settings.sibson_radius_scale = 1.5f;
    settings.sibson_sigma_v = 0.0f;
    settings.smooth_T = 0;
    settings.smooth_alpha = 1.0f;
    settings.smooth_sigma_v = 0.3f;
    settings.smooth_sigma_s_scale = 2.0f;
    settings.phong_enabled = false;
    settings.phong_ambient = 0.3f;
    settings.phong_diffuse = 0.7f;
    settings.phong_specular = 0.3f;
    settings.phong_shininess = 32.0f;
    settings.ao_enabled = false;
    settings.ao_num_dirs = 32;
    settings.ao_max_distance = 0.05f;
    settings.ao_strength = 1.0f;
    settings.slice_min = Vec3f(-1.0f, -1.0f, -1.0f);
    settings.slice_max = Vec3f( 1.0f,  1.0f,  1.0f);
    return settings;
}

/// @brief Compute per-cell radius (max edge length to any neighbor) from adjacent_diff.
/// Must be called after prefetch_adjacent_diff.
void compute_cell_radius(uint32_t num_points,
                         const Vec4h *adjacent_diff,
                         const uint32_t *adj_offsets,
                         float *cell_radius,
                         const void *stream = nullptr);

/// @brief Prefetch offset for each edge in the adjacency matrix
void prefetch_adjacent_diff(const Vec3f *points,
                            uint32_t num_points,
                            uint32_t point_adjacency_size,
                            const uint32_t *point_adjacency,
                            const uint32_t *point_adjacency_offsets,
                            const float *cell_radius,
                            Vec4h *adjacent_diff,
                            const void *stream);

class Pipeline {
  public:
    virtual ~Pipeline() = default;

    virtual void trace_forward(const TraceSettings &settings,
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
                               // thin-surface parameters (nullptr = disabled)
                               const float *density_delta = nullptr,
                               const float *quaternions = nullptr,
                               const float *texel_sites_2d = nullptr,
                               const float *texel_heights = nullptr,
                               // LC64 plan v3 Commit 2A -- independent-side
                               // raw logits (each (N,1)).  Only read when
                               // settings.thin_surface_independent_mode is
                               // true; nullptr for legacy / absolute / relative.
                               const float *raw_plus = nullptr,
                               const float *raw_minus = nullptr) = 0;

    virtual void trace_backward(const TraceSettings &settings,
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
                                // thin-surface parameters (nullptr = disabled)
                                const float *density_delta = nullptr,
                                const float *quaternions = nullptr,
                                const float *texel_sites_2d = nullptr,
                                const float *texel_heights = nullptr,
                                float *density_delta_grad = nullptr,
                                float *quaternions_grad = nullptr,
                                float *texel_sites_2d_grad = nullptr,
                                float *texel_heights_grad = nullptr,
                                // LC64 plan v3 Commit 2B -- independent-side
                                // raw logits + their per-cell gradients.
                                // raw_plus / raw_minus are read when
                                // settings.thin_surface_independent_mode is
                                // true; raw_plus_grad / raw_minus_grad
                                // accumulate dL/draw_p, dL/draw_n via the
                                // chain rule  dmu_side / draw_side =
                                // activation_scale * sigmoid(beta * raw_side).
                                // nullptr in legacy / absolute / relative mode.
                                const float *raw_plus = nullptr,
                                const float *raw_minus = nullptr,
                                float *raw_plus_grad = nullptr,
                                float *raw_minus_grad = nullptr) = 0;

    // Stub for viewer compatibility — CT pipeline does not implement this
    virtual void trace_visualization(const TraceSettings &settings,
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
                                     const void *stream = nullptr) {}

    // Stub for viewer compatibility
    virtual void trace_benchmark(const TraceSettings &settings,
                                 uint32_t num_points,
                                 const Vec3f *points,
                                 const void *attributes,
                                 const uint32_t *point_adjacency,
                                 const uint32_t *point_adjacency_offsets,
                                 const Vec4h *adjacent_diff,
                                 Camera camera,
                                 const uint32_t *start_point_index,
                                 uint32_t *ray_rgba) {}

    virtual uint32_t attribute_dim() const = 0;

    virtual ScalarType attribute_type() const = 0;
};

std::shared_ptr<Pipeline> create_ct_pipeline();

/// @brief DVR of a voxel grid stored in a CUDA 3D texture (trilinear, normalized coords).
/// vol_tex_handle is a cudaTextureObject_t cast to uint64_t.
/// output_surface is a CUsurfObject cast to uint64_t.
/// Reuses vis_settings.slice_min/max for ray clipping; no separate bbox.
void launch_volume_visualization(const TraceSettings &trace_settings,
                                 const VisualizationSettings &vis_settings,
                                 int num_steps,
                                 float voxel_eps,
                                 const Camera &camera,
                                 int x_offset,
                                 CMapTable cmap_table,
                                 TransferFunctionTable tf_table,
                                 uint64_t vol_tex_handle,
                                 uint64_t output_surface,
                                 const void *stream = nullptr);

} // namespace radfoam
