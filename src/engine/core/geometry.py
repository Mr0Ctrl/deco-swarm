# src/engine/core/geometry.py
import math
from typing import Tuple, Optional

Point2D = Tuple[float, float]
Segment2D = Tuple[Point2D, Point2D]


def wrap_angle_deg(angle_deg: float) -> float:
    """Açıyı [0, 360) aralığında tutar."""
    return angle_deg % 360.0


def shortest_angle_diff_deg(target_angle_deg: float, source_angle_deg: float) -> float:
    """
    İki açı arasındaki en kısa dönüş açısını [-180, +180] aralığında döndürür.
    Pozitif: Saat yönü (sağa dönüş), Negatif: Saat yönü tersi (sola dönüş).
    """
    return (target_angle_deg - source_angle_deg + 180.0) % 360.0 - 180.0


def distance(p1: Point2D, p2: Point2D) -> float:
    """İki nokta arasındaki Öklid mesafesi."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def intersect_ray_segment(
    ray_origin: Point2D,
    ray_dir: Point2D,
    max_dist: float,
    segment: Segment2D,
) -> Optional[Tuple[Point2D, float]]:
    """
    Parametrik doğru-doğru kesişimi (Determinant / 2D Cross Product).
    Kesişim varsa (çarpışma_noktası, mesafe) döner, yoksa None.
    """
    px, py = ray_origin
    rx, ry = ray_dir[0] * max_dist, ray_dir[1] * max_dist

    (x1, y1), (x2, y2) = segment
    sx, sy = x2 - x1, y2 - y1

    det = rx * sy - ry * sx
    if abs(det) < 1e-8:
        return None  # Doğrular paralel

    qpx = x1 - px
    qpy = y1 - py

    t = (qpx * sy - qpy * sx) / det
    u = (qpx * ry - qpy * rx) / det

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        hit_x = px + t * rx
        hit_y = py + t * ry
        actual_dist = t * max_dist
        return (hit_x, hit_y), actual_dist

    return None

def compute_ego_polar_target(
    agent_pos: Point2D,
    agent_angle_deg: float,
    target_pos: Point2D,
    max_map_distance: float,
) -> Tuple[float, float]:
    """
    Saf Geometrik Hesaplama:
    Ajanın gövdesini merkez alan bağıl polar hedef koordinatlarını hesaplar.
    İleride C++ tarafına (SIMD/Fast Math) birebir taşınacaktır.
    """
    dist = distance(agent_pos, target_pos)
    norm_dist = min(1.0, dist / max_map_distance)

    dx = target_pos[0] - agent_pos[0]
    dy = target_pos[1] - agent_pos[1]
    target_angle_deg = math.degrees(math.atan2(dy, dx))

    angle_diff_deg = shortest_angle_diff_deg(target_angle_deg, agent_angle_deg)
    norm_angle = angle_diff_deg / 180.0

    return norm_dist, norm_angle