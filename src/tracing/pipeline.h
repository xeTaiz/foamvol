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
};

inline TraceSettings default_trace_settings() {
    TraceSettings settings;
    settings.weight_threshold = 0.001f;
    settings.max_intersections = 2048;
    return settings;
}

enum VisualizationMode {
    RGB = 0,
    Depth = 1,
    Alpha = 2,
    Intersections = 3,
    VolumeDensity = 4,
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
    float tf_density_min;
    float tf_density_max;
    float tf_opacity_scale;
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
    settings.tf_density_min = 0.0f;
    settings.tf_density_max = 1.0f;
    settings.tf_opacity_scale = 100.0f;
    return settings;
}

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
                               const float *cell_radius = nullptr) = 0;

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
                                const float *cell_radius = nullptr) = 0;

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
                                     uint32_t start_index,
                                     uint64_t output_surface,
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

} // namespace radfoam
