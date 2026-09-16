# src/deco_swarm/engine/raycast.py
import math
from typing import List, Tuple

Segment = Tuple[Tuple[float, float], Tuple[float, float]]
# RayHit: (Uç Nokta (x, y), Mesafe, Normalize Mesafe [0-1], Çarptı Mı?)
RayHit = Tuple[Tuple[float, float], float, float, bool]


def intersect_ray_segment(
    ray_origin: Tuple[float, float],
    ray_dir: Tuple[float, float],
    max_dist: float,
    segment: Segment
) -> Tuple[Tuple[float, float], float] | None:
    """
    Parametrik doğru-doğru kesişimi (2D Cross Product / Determinant).
    """
    px, py = ray_origin
    rx, ry = ray_dir[0] * max_dist, ray_dir[1] * max_dist

    (x1, y1), (x2, y2) = segment
    sx, sy = x2 - x1, y2 - y1

    # Determinant (r x s)
    det = rx * sy - ry * sx

    # Doğrular paralel ise kesişim yoktur
    if abs(det) < 1e-8:
        return None

    # (q - p) vektörü
    qpx = x1 - px
    qpy = y1 - py

    # t: Işın üzerindeki oran, u: Duvar parçası üzerindeki oran
    t = (qpx * sy - qpy * sx) / det
    u = (qpx * ry - qpy * rx) / det

    # Kesişimin hem ışının boyu hem de duvarın boyu içinde olması şartı
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        hit_x = px + t * rx
        hit_y = py + t * ry
        actual_distance = t * max_dist
        return (hit_x, hit_y), actual_distance

    return None


def cast_rays(
    origin: Tuple[float, float],
    heading_deg: float,
    fov_deg: float,
    num_rays: int,
    max_dist: float,
    segments: List[Segment]
) -> List[RayHit]:
    """
    Ajanın baktığı yöne göre fov_deg açısına num_rays kadar ışın dağıtır 
    ve her ışın için en yakın temas noktasını döndürür.
    """
    results: List[RayHit] = []

    if num_rays == 1:
        angles = [heading_deg]
    else:
        # Açıları eşit aralıklarla yay (Örn: -90 dereceden +90 dereceye)
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

        # Işın bir şeye çarptı mı?
        if closest_hit is not None:
            endpoint = closest_hit
            has_hit = True
        else:
            # Çarpmadıysa maksimum menzile kadar uzat
            endpoint = (px + ray_dir[0] * max_dist, py + ray_dir[1] * max_dist)
            has_hit = False

        norm_dist = min_dist / max_dist
        results.append((endpoint, min_dist, norm_dist, has_hit))

    return results