# src/deco_swarm/engine/world.py
import math
from typing import Dict, List, Tuple
from src.engine.map import Segment
from src.engine.agent import Agent
from src.engine.raycast import cast_rays, RayHit
from src.engine.map_loader import MapConfig


def resolve_circle_segment_collision(agent: Agent, segment: Segment):
    (x1, y1), (x2, y2) = segment
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return

    cx, cy = agent.x, agent.y
    t = ((cx - x1) * dx + (cy - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    diff_x = cx - closest_x
    diff_y = cy - closest_y
    dist = math.hypot(diff_x, diff_y)

    if dist < agent.radius:
        if dist > 1e-6:
            push_nx = diff_x / dist
            push_ny = diff_y / dist
        else:
            push_nx, push_ny = 1.0, 0.0

        overlap = agent.radius - dist
        agent.x += push_nx * overlap
        agent.y += push_ny * overlap


class World:
    """
    PettingZoo ve çoklu ajan mimarisine tam uyumlu simülasyon dünyası.
    """

    def __init__(self, map_config: MapConfig, ray_config: dict, agent_config: dict):
        self.map_config = map_config
        self.ray_config = ray_config
        self.agent_config = agent_config

        self.width = map_config.width
        self.height = map_config.height
        self.segments = map_config.segments
        self.goal_pos = map_config.goal_pos
        self.goal_radius = map_config.goal_radius

        # ÇOKLU AJAN SÖZLÜĞÜ (1v0'da sadece 'agent_0' var, 3v3'te 6 ajan olacak)
        self.agents: Dict[str, Agent] = {}
        self.latest_rays: Dict[str, List[RayHit]] = {}

        self.reset()

    def reset(self):
        """Tüm ajanları haritadaki başlangıç konumlarına döndürür."""
        self.agents.clear()
        self.latest_rays.clear()

        for agent_id, spawn in self.map_config.spawns.items():
            sx, sy, s_angle = spawn
            self.agents[agent_id] = Agent(
                x=sx,
                y=sy,
                angle_deg=s_angle,
                radius=self.agent_config.get("radius", 16.0),
                linear_speed=self.agent_config.get("linear_speed", 180.0),
                angular_speed=self.agent_config.get("angular_speed", 200.0),
            )

        # İlk raycast taramasını yap
        self._update_all_rays()

    def step(self, actions: Dict[str, Tuple[float, float, float]], dt: float):
        """
        actions: {'agent_0': (move_fwd, strafe, rotate_dir), ...}
        """
        # 1. Her ajanın hareketini uygula
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                move_fwd, strafe, rotate_dir = actions[agent_id]
                agent.rotate(rotate_dir, dt)
                agent.move(move_fwd, strafe, dt)

        # 2. Duvar çarpışmalarını çöz (Kayma fiziği)
        for _ in range(2):
            for agent in self.agents.values():
                for seg in self.segments:
                    resolve_circle_segment_collision(agent, seg)

        # 3. Gözleri (Raycast) güncelle
        self._update_all_rays()

    def _update_all_rays(self):
        for agent_id, agent in self.agents.items():
            self.latest_rays[agent_id] = cast_rays(
                origin=agent.position,
                heading_deg=agent.angle_deg,
                fov_deg=self.ray_config["fov_degrees"],
                num_rays=self.ray_config["ray_count"],
                max_dist=self.ray_config["max_distance"],
                segments=self.segments,
            )

    def is_goal_reached(self, agent_id: str) -> bool:
        """Ajanın hedefe ulaşıp ulaşmadığını test eder."""
        if agent_id not in self.agents:
            return False
        agent = self.agents[agent_id]
        dist_to_goal = math.hypot(agent.x - self.goal_pos[0], agent.y - self.goal_pos[1])
        return dist_to_goal <= (agent.radius + self.goal_radius)