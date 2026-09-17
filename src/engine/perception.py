# src/deco_swarm/engine/perception.py
import math
from typing import Tuple


def compute_ego_polar_target(
    agent_pos: Tuple[float, float],
    agent_angle_deg: float,
    target_pos: Tuple[float, float],
    max_map_distance: float,
) -> Tuple[float, float]:
    """
    Ajanın gövdesini merkez alan bağıl polar hedef koordinatlarını hesaplar.

    Döndürür:
        norm_distance: [0.0, 1.0] aralığında hedefe kuş uçuşu uzaklık.
        norm_angle: [-1.0, 1.0] aralığında hedefin yönü.
                    -1.0: Tam arkada solda
                     0.0: Tam ajanın baktığı doğrultuda
                    +1.0: Tam arkada sağda
    """
    ax, ay = agent_pos
    gx, gy = target_pos

    # 1. Kartezyen fark vektörü
    dx = gx - ax
    dy = gy - ay

    # 2. Kuş uçuşu mesafe
    distance = math.hypot(dx, dy)
    norm_distance = min(1.0, distance / max_map_distance)

    # 3. Dünyadaki mutlak hedef açısı (Derece)
    target_angle_deg = math.degrees(math.atan2(dy, dx))

    # 4. Açı Sarmalama (Angle Wrapping): En kısa dönüş açısını bul
    diff_deg = (target_angle_deg - agent_angle_deg + 180.0) % 360.0 - 180.0

    # 5. [-180, 180] aralığını [-1.0, 1.0] aralığına sıkıştır
    norm_angle = diff_deg / 180.0

    return norm_distance, norm_angle