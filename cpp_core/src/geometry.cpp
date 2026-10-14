#include <deco_core/geometry.hpp>

namespace deco
{

    // Wraps any angle into the positive range [0.0, 360.0) degrees
    float wrap_angle_deg(float angle_deg)
    {
        float deg = std::fmod(angle_deg, 360.0f);
        if (deg < 0.0f)
            deg += 360.0f;
        return deg;
    }

    // Computes shortest signed angular difference in [-180.0, +180.0] degrees
    // Positive: clockwise turn, Negative: counter-clockwise turn
    float shortest_angle_diff_deg(float target_deg, float source_deg)
    {
        float diff = std::fmod((target_deg - source_deg + 180.0f), 360.0f);
        if (diff < 0.0f)
            diff += 360.0f;
        return diff - 180.0f;
    }

    // Euclidean distance between two points using Eigen's SIMD-accelerated norm
    float distance(const Vec2 &p1, const Vec2 &p2)
    {
        return (p1 - p2).norm();
    }

    // Fast 2D Parametric Ray-Segment Intersection (Determinant / Cross Product)
    // Ray: R(t) = ray_origin + t * (ray_dir * max_dist), t in [0, 1]
    // Segment: S(u) = p1 + u * (p2 - p1), u in [0, 1]
    std::optional<RayHit> intersect_ray_segment(
        const Vec2 &ray_origin,
        const Vec2 &ray_dir,
        float max_dist,
        const Segment2D &segment)
    {
        Vec2 ray = ray_dir * max_dist;
        Vec2 seg = segment.p2 - segment.p1;

        // Vector from ray origin to segment start
        Vec2 qp = segment.p1 - ray_origin;

        // 2D Cross product determinant: ray x seg
        float det = ray.x() * seg.y() - ray.y() * seg.x();
        if (std::abs(det) < 1e-8f)
            return std::nullopt; // Lines are parallel or collinear

        // Solve parameters t and u using Cramer's rule
        float t = (qp.x() * seg.y() - qp.y() * seg.x()) / det;
        float u = (qp.x() * ray.y() - qp.y() * ray.x()) / det;

        // Check if intersection point lies within both the ray length and the segment span
        if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f)
        {
            RayHit hit;
            hit.endpoint = ray_origin + t * ray;
            hit.norm_dist = t; // Since ray length is max_dist, t is exactly (actual_dist / max_dist) in [0.0, 1.0]
            hit.has_hit = true;
            return hit;
        }

        return std::nullopt;
    }

    // Computes relative polar coordinates (distance and heading) to target for agent observation
    EgoPolarTarget compute_ego_polar_target(
        const Vec2 &agent_pos,
        float agent_angle_deg,
        const Vec2 &target_pos,
        float max_map_distance)
    {
        float dist = distance(agent_pos, target_pos);
        EgoPolarTarget target;
        target.norm_dist = std::min(1.0f, dist / max_map_distance);

        // Vector pointing from agent to target
        Vec2 delta = target_pos - agent_pos;
        float target_angle_rad = std::atan2(delta.y(), delta.x());
        float target_angle_deg = target_angle_rad * (180.0f / 3.14159265358979323846f);

        // Relative heading error from agent's current orientation to target
        float angle_diff_deg = shortest_angle_diff_deg(target_angle_deg, agent_angle_deg);
        target.norm_angle = angle_diff_deg / 180.0f;

        return target;
    }

} // namespace deco
