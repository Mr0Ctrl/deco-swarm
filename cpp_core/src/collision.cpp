#include <deco_core/collision.hpp>
#include <cmath>
#include <algorithm>

namespace deco
{
    constexpr float PI_F = 3.14159265358979323846f;
    constexpr float DEG2RAD_F = PI_F / 180.0f;

    // 1. Single circle-segment sliding collision resolver
    // Given an agent's circle center and radius, projects onto segment and pushes
    // the circle along the contact normal if penetration is detected.
    CollisionResult resolve_circle_segment(
        const Vec2 &circle_center,
        float radius,
        const Segment2D &segment)
    {
        CollisionResult result;
        result.new_position = circle_center;
        result.collided = false;

        Vec2 seg = segment.p2 - segment.p1;
        float seg_len_sqr = seg.squaredNorm();
        if (seg_len_sqr < 1e-6f)
        {
            return result; // Degenerate zero-length segment
        }

        // Projection scalar t in [0.0, 1.0] along segment: t = ((c - p1) . seg) / ||seg||^2
        Vec2 cp1 = circle_center - segment.p1;
        float t = std::clamp(cp1.dot(seg) / seg_len_sqr, 0.0f, 1.0f);

        Vec2 closest = segment.p1 + t * seg;
        Vec2 diff = circle_center - closest;
        float dist_sqr = diff.squaredNorm();

        // Check penetration (avoiding std::sqrt if no collision occurs)
        if (dist_sqr < radius * radius)
        {
            float dist = std::sqrt(dist_sqr);
            Vec2 push_n;
            if (dist > 1e-6f)
            {
                push_n = diff / dist; // Unit normal pointing outward
            }
            else
            {
                push_n = Vec2(1.0f, 0.0f); // Fallback normal if center is exactly on the wall
            }

            float overlap = radius - dist;
            result.new_position = circle_center + push_n * overlap;
            result.collided = true;
        }

        return result;
    }

    // 2. Multi-segment solver (slides agent smoothly around corners and multiple walls)
    // Runs up to 2 iterative relaxation passes to resolve corner intersections cleanly.
    CollisionResult resolve_circle_segments(
        const Vec2 &circle_center,
        float radius,
        const std::vector<Segment2D> &segments)
    {
        Vec2 current_pos = circle_center;
        bool any_collided = false;

        constexpr size_t MAX_SOLVER_ITERATIONS = 2;
        for (size_t iter = 0; iter < MAX_SOLVER_ITERATIONS; ++iter)
        {
            bool moved_in_this_pass = false;
            for (const auto &segment : segments)
            {
                CollisionResult res = resolve_circle_segment(current_pos, radius, segment);
                if (res.collided)
                {
                    current_pos = res.new_position;
                    any_collided = true;
                    moved_in_this_pass = true;
                }
            }

            // Early exit if settled (no penetration in this pass)
            if (!moved_in_this_pass)
            {
                break;
            }
        }

        return CollisionResult{current_pos, any_collided};
    }

    // 3. Batch raycaster: casts num_rays across fov_deg around heading_deg
    // Finds the closest intersection for each ray against all obstacle segments.
    std::vector<RayHit> cast_rays(
        const Vec2 &origin,
        float heading_deg,
        float fov_deg,
        int num_rays,
        float max_dist,
        const std::vector<Segment2D> &segments)
    {
        std::vector<RayHit> results;
        if (num_rays <= 0)
        {
            return results;
        }

        results.reserve(num_rays); // Pre-allocate contiguous buffer to avoid reallocations

        float start_angle = heading_deg;
        float step = 0.0f;
        if (num_rays > 1)
        {
            start_angle = heading_deg - (fov_deg * 0.5f);
            step = fov_deg / static_cast<float>(num_rays - 1);
        }

        for (int i = 0; i < num_rays; ++i)
        {
            float angle_deg = start_angle + static_cast<float>(i) * step;
            float rad = angle_deg * DEG2RAD_F;
            Vec2 ray_dir(std::cos(rad), std::sin(rad));

            // Default ray result (open air / max range)
            RayHit closest_hit;
            closest_hit.norm_dist = 1.0f;
            closest_hit.has_hit = false;
            closest_hit.endpoint = origin + ray_dir * max_dist;

            // Find closest intersecting segment
            for (const auto &segment : segments)
            {
                auto hit_opt = intersect_ray_segment(origin, ray_dir, max_dist, segment);
                if (hit_opt.has_value()) // Checks if std::optional contains a hit
                {
                    if (hit_opt->norm_dist < closest_hit.norm_dist)
                    {
                        closest_hit = *hit_opt;
                    }
                }
            }

            results.push_back(closest_hit);
        }

        return results;
    }

} // namespace deco