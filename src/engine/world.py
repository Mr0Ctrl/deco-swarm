# src/engine/world.py
import math
from typing import List, Tuple
from src.engine.map import GameMap, Segment
from src.engine.agent import Agent
from src.engine.raycast import cast_rays, RayHit


def resolve_circle_segment_collision(agent: Agent, segment: Segment):
    """
    Ajanı en yakın doğru parçasına göre test eder ve içeri girdiyse dışarı iter.
    """
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return

    # Ajanın merkeze olan izdüşüm oranı t'nin hesabı
    cx, cy = agent.x, agent.y
    t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
    # Doğru parçası sınırları içine kırp [0.0, 1.0]
    t = max(0.0, min(1.0, t))

    # Doğru parçası üzerindeki en yakın nokta
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    # En yakın noktadan merkeze olan fark
    diff_x = cx - closest_x
    diff_y = cy - closest_y
    dist = math.hypot(diff_x, diff_y)

    # Çarpışma varsa ajanı dışarı it
    if dist < agent.radius:
        if dist > 1e-6:
            push_nx = diff_x / dist
            push_ny = diff_y / dist
        else:
            # Tam üst üste binme durumu için yapay normal
            push_nx, push_ny = 1.0, 0.0

        overlap = agent.radius - dist
        agent.x += push_nx * overlap
        agent.y += push_ny * overlap


class World:
    def __init__(self, width: float, height: float, ray_config: dict, agent_config: dict):
        self.width = width
        self.height = height
        self.map = GameMap(width, height)
        
        # Ajanı haritanın sol-alt köşesine yakın başlat
        self.agent = Agent(
            x=width * 0.15,
            y=height * 0.2,
            angle_deg=0.0,
            radius=agent_config.get("radius", 16.0),
            linear_speed=agent_config.get("linear_speed", 180.0),
            angular_speed=agent_config.get("angular_speed", 200.0),
        )

        self.ray_config = ray_config
        self.latest_rays: List[RayHit] = []

    def step(self, move_fwd: float, strafe: float, rotate_dir: float, dt: float):
        """
        Tek bir fizik adımı:
        1. Ajan hareket eder
        2. Duvar çarpışmaları çözülür (Kayma sağlanır)
        3. Yeni gözler (Raycast) hesaplanır
        """
        # 1. Hareket
        self.agent.rotate(rotate_dir, dt)
        self.agent.move(move_fwd, strafe, dt)

        # 2. Duvar çarpışmaları (Ardışık çarpışmalar için 2 iterasyon)
        for _ in range(2):
            for seg in self.map.segments:
                resolve_circle_segment_collision(self.agent, seg)

        # 3. Raycast taraması
        self.latest_rays = cast_rays(
            origin=self.agent.position,
            heading_deg=self.agent.angle_deg,
            fov_deg=self.ray_config["fov_degrees"],
            num_rays=self.ray_config["ray_count"],
            max_dist=self.ray_config["max_distance"],
            segments=self.map.segments,
        )