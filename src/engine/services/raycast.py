# src/engine/services/raycast.py
import math
from typing import List
from engine.core.geometry import Point2D, Segment2D, intersect_ray_segment
from engine.entities.agent import RayHit


class RaycastService:
    @staticmethod
    def cast_rays(
        origin: Point2D,
        heading_deg: float,
        fov_deg: float,
        num_rays: int,
        max_dist: float,
        segments: List[Segment2D],
        cpp_segments: list = None,
    ) -> List[RayHit]:
        try:
            import deco_core_cpp
            import numpy as np

            # Fast C++ path
            origin_arr = np.array(origin, dtype=np.float32)
            if cpp_segments is None:
                cpp_segments = [
                    deco_core_cpp.Segment2D(np.array(s[0], dtype=np.float32), np.array(s[1], dtype=np.float32))
                    for s in segments
                ]
            raw_hits = deco_core_cpp.cast_rays(
                origin_arr, float(heading_deg), float(fov_deg), int(num_rays), float(max_dist), cpp_segments
            )
            # Wrap into RayHit dataclass expected by observation builder & renderer
            return [
                RayHit(
                    endpoint=(float(h.endpoint[0]), float(h.endpoint[1])),
                    distance=h.norm_dist * max_dist,
                    norm_distance=h.norm_dist,
                    has_hit=h.has_hit,
                )
                for h in raw_hits
            ]
        except ImportError:
            pass

        # Fallback pure-Python path
        results: List[RayHit] = []

        if num_rays == 1:
            angles = [heading_deg]
        else:
            half_fov = fov_deg / 2.0
            start_angle = heading_deg - half_fov
            step = fov_deg / (num_rays - 1)
            angles = [start_angle + i * step for i in range(num_rays)]

        px, py = origin

        for angle in angles:
            rad = math.radians(angle)
            ray_dir = (math.cos(rad), math.sin(rad))

            closest_hit = None
            min_dist = max_dist

            for seg in segments:
                hit = intersect_ray_segment(origin, ray_dir, max_dist, seg)
                if hit is not None:
                    hit_pt, dist = hit
                    if dist < min_dist:
                        min_dist = dist
                        closest_hit = hit_pt

            if closest_hit is not None:
                endpoint = closest_hit
                has_hit = True
            else:
                endpoint = (px + ray_dir[0] * max_dist, py + ray_dir[1] * max_dist)
                has_hit = False

            norm_dist = min_dist / max_dist
            results.append(RayHit(endpoint, min_dist, norm_dist, has_hit))

        return results