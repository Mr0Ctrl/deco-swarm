# src/engine/core/collision.py
import math
from typing import Tuple
from src.engine.core.geometry import Point2D, Segment2D


def resolve_circle_segment(
    circle_center: Point2D,
    radius: float,
    segment: Segment2D,
) -> Tuple[Point2D, bool]:
    """
    Bir dairenin doğru parçasıyla çarpışmasını çözer.
    Daire içerideyse en yakın normal doğrultusunda dışarı itilmiş yeni koordinatı döner.

    Döndürür:
        (yeni_merkez, carpti_mi)
    """
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return circle_center, False

    cx, cy = circle_center
    # İzdüşüm oranı t [0.0, 1.0]
    t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    diff_x = cx - closest_x
    diff_y = cy - closest_y
    dist = math.hypot(diff_x, diff_y)

    if dist < radius:
        if dist > 1e-6:
            push_nx = diff_x / dist
            push_ny = diff_y / dist
        else:
            push_nx, push_ny = 1.0, 0.0

        overlap = radius - dist
        new_x = cx + push_nx * overlap
        new_y = cy + push_ny * overlap
        return (new_x, new_y), True

    return circle_center, False