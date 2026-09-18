# src/engine/world.py
import math
from typing import Dict, Tuple
from engine.core.geometry import Point2D, distance, compute_ego_polar_target
from engine.core.collision import resolve_circle_segment
from engine.entities.agent import Agent
from engine.entities.map import GameMap
from engine.services.raycast import RaycastService
from engine.map_loader import MapConfig


class World:
    def __init__(
        self,
        map_config: MapConfig,
        sim_config: dict,
        agent_config: dict,
    ):
        self.map = GameMap(map_config.width, map_config.height, map_config.segments)
        self.map_config = map_config
        self.sim_config = sim_config
        self.agent_config = agent_config

        self.goal_pos: Point2D = map_config.goal_pos
        self.goal_radius: float = map_config.goal_radius
        self.max_map_distance: float = math.hypot(map_config.width, map_config.height)

        self.agents: Dict[str, Agent] = {}
        self.last_collisions: Dict[str, bool] = {}
        self.reset()

    def reset(self):
        """Ajanları başlangıç noktalarına yerleştirir ve ilk algılarını günceller."""
        self.agents.clear()
        body_cfg = self.agent_config["body"]

        for agent_id, spawn in self.map_config.spawns.items():
            sx, sy, s_angle = spawn
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                x=sx,
                y=sy,
                angle_deg=s_angle,
                radius=body_cfg["radius"],
                linear_speed=body_cfg["linear_speed"],
                angular_speed=body_cfg["angular_speed"],
            )

        self._update_perceptions()

    def step(self, actions: Dict[str, Tuple[float, float, float]], dt: float):
        """
        actions: {'agent_0': (move_fwd, strafe, rotate_dir)}
        """
        # 1. Hareket ve İç Durum Güncellemesi
        for agent_id, agent in self.agents.items():
            if agent_id in actions:
                move_fwd, strafe, rotate_dir = actions[agent_id]

                # Ajanın hafızasını güncelle (a_{t-1} ve anlık hız)
                agent.last_action = (move_fwd, strafe, rotate_dir)
                agent.current_vel = (move_fwd, strafe, rotate_dir)

                agent.rotate(rotate_dir, dt)
                agent.move(move_fwd, strafe, dt)

        # 2. Çarpışma Çözümü (Fizik motoru adımı)
        self.last_collisions = {agent_id: False for agent_id in self.agents}
        iterations = self.sim_config["physics"].get("collision_iterations", 2)
        for _ in range(iterations):
            for agent_id, agent in self.agents.items():
                for seg in self.map.segments:
                    new_pos, collided = resolve_circle_segment(agent.position, agent.radius, seg)
                    agent.set_position(new_pos)
                    if collided:
                        self.last_collisions[agent_id] = True

        # 3. Gözlem ve Algıları Güncelle
        self._update_perceptions()

    def _update_perceptions(self):
        """Ajanların içindeki rays ve polar_target değişkenlerini günceller."""
        ray_cfg = self.agent_config["perception"]

        for agent in self.agents.values():
            # Işınları hesapla ve ajanın içine ata
            agent.rays = RaycastService.cast_rays(
                origin=agent.position,
                heading_deg=agent.angle_deg,
                fov_deg=ray_cfg["fov_degrees"],
                num_rays=ray_cfg["ray_count"],
                max_dist=ray_cfg["max_distance"],
                segments=self.map.segments,
            )

            # Bağıl hedef vektörünü hesapla ve ajanın içine ata
            agent.polar_target = compute_ego_polar_target(
                agent_pos=agent.position,
                agent_angle_deg=agent.angle_deg,
                target_pos=self.goal_pos,
                max_map_distance=self.max_map_distance,
            )

    def is_goal_reached(self, agent_id: str) -> bool:
        if agent_id not in self.agents:
            return False
        agent = self.agents[agent_id]
        return distance(agent.position, self.goal_pos) <= (agent.radius + self.goal_radius)