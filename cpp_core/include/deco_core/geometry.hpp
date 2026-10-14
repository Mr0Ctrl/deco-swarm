#pragma once

#include <Eigen/Dense>
#include <optional>
#include <cmath>

namespace deco {

// 1. Type Aliases
// In modern C++, 'using' replaces C's 'typedef'.
// Eigen::Vector2f is a 2D float vector with SIMD hardware optimization.
using Vec2 = Eigen::Vector2f;

struct Segment2D {
    Vec2 p1;
    Vec2 p2;
};

struct RayHit {
    Vec2 endpoint;     // Coordinates of hit point or ray limit endpoint (x, y)
    float norm_dist;   // Normalized distance [0.0, 1.0] (dist / max_dist)
    bool has_hit;      // True if segment was hit, false if open space
};

// --- Function Declarations for You to Implement ---

// Clamps or wraps angle in degrees to [0, 360)
float wrap_angle_deg(float angle_deg);

// Returns shortest angular difference in [-180, +180] degrees
float shortest_angle_diff_deg(float target_deg, float source_deg);

// Euclidean distance between two 2D points
float distance(const Vec2& p1, const Vec2& p2);

// Parametric ray-segment intersection.
// Returns std::optional<RayHit>:
//   - If intersection occurs: returns RayHit{hit_point, distance}
//   - If parallel or misses: returns std::nullopt (C++'s safe equivalent of NULL/None)
std::optional<RayHit> intersect_ray_segment(
    const Vec2& ray_origin,
    const Vec2& ray_dir,
    float max_dist,
    const Segment2D& segment
);

// Struct for returning polar coordinates to target
struct EgoPolarTarget {
    float norm_dist;   // Normalized distance [0.0, 1.0]
    float norm_angle;  // Normalized angle [-1.0, 1.0] (relative heading)
};

// Computes relative polar coordinates (distance and heading) to target
EgoPolarTarget compute_ego_polar_target(
    const Vec2& agent_pos,
    float agent_angle_deg,
    const Vec2& target_pos,
    float max_map_distance
);

} // namespace deco
