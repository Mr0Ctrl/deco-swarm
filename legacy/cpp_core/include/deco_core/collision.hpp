#pragma once

#include <vector>
#include <deco_core/geometry.hpp>

namespace deco
{

    struct CollisionResult
    {
        Vec2 new_position; // Circle position after sliding push-out
        bool collided;     // True if circle overlapped with the segment
    };

    // 1. Single circle-segment sliding collision resolver
    CollisionResult resolve_circle_segment(
        const Vec2 &circle_center,
        float radius,
        const Segment2D &segment);

    // 2. Multi-segment solver (slides agent around multiple walls)
    CollisionResult resolve_circle_segments(
        const Vec2 &circle_center,
        float radius,
        const std::vector<Segment2D> &segments);

    // 3. Batch raycaster: casts num_rays over fov_deg
    std::vector<RayHit> cast_rays(
        const Vec2 &origin,
        float heading_deg,
        float fov_deg,
        int num_rays,
        float max_dist,
        const std::vector<Segment2D> &segments);

}