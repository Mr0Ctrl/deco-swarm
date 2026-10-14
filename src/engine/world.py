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
        self._init_cpp_segments()
        self.reset()

    def _init_cpp_segments(self):
        """Pre-converts Python map segments into C++ Segment2D objects for zero-allocation stepping."""
        try:
            import deco_core_cpp
            import numpy as np
            self.cpp_segments = [
                deco_core_cpp.Segment2D(np.array(s[0], dtype=np.float32), np.array(s[1], dtype=np.float32))
                for s in self.map.segments
            ]
            self.has_cpp = True
        except ImportError:
            self.cpp_segments = None
            self.has_cpp = False

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

    def load_map(self, map_config: MapConfig):
        """Yeni bir harita yükler ve dünyayı baştan başlatır."""
        self.map_config = map_config
        self.map = GameMap(map_config.width, map_config.height, map_config.segments)
        self.goal_pos = map_config.goal_pos
        self.goal_radius = map_config.goal_radius
        self.max_map_distance = math.hypot(map_config.width, map_config.height)
        self._init_cpp_segments()
        self.reset()

    def step(self, actions: Dict[str, Tuple[float, float, float]], dt: float):
        """
        actions: {'agent_0': (move_fwd, strafe, rotate_dir)}
        """
        # 1. Hareket ve İç Durum Güncellemesi
        for agent_id, agent in self.agents.items():
            agent.collided = False
            if agent_id in actions:
                move_fwd, strafe, rotate_dir = actions[agent_id]

                # Ajanın hafızasını güncelle (a_{t-1} ve anlık hız)
                agent.last_action = (move_fwd, strafe, rotate_dir)
                agent.current_vel = (move_fwd, strafe, rotate_dir)

                agent.rotate(rotate_dir, dt)
                agent.move(move_fwd, strafe, dt)

        # 2. Çarpışma Çözümü (Fizik motoru adımı)
        if self.has_cpp:
            import deco_core_cpp
            import numpy as np
            for agent in self.agents.values():
                pos_arr = np.array(agent.position, dtype=np.float32)
                res = deco_core_cpp.resolve_circle_segments(pos_arr, agent.radius, self.cpp_segments)
                if res.collided:
                    agent.collided = True
                agent.set_position((float(res.new_position[0]), float(res.new_position[1])))
        else:
            iterations = self.sim_config["physics"].get("collision_iterations", 2)
            for _ in range(iterations):
                for agent in self.agents.values():
                    for seg in self.map.segments:
                        new_pos, has_hit = resolve_circle_segment(agent.position, agent.radius, seg)
                        if has_hit:
                            agent.collided = True
                        agent.set_position(new_pos)

        # 3. Gözlem ve Algıları Güncelle
        self._update_perceptions()

    def _update_perceptions(self):
        """Ajanların içindeki rays ve polar_target değişkenlerini günceller."""
        ray_cfg = self.agent_config["perception"]

        for agent in self.agents.values():
            # Işınları hesapla ve ajanın içine ata (C++ hızlandırılmış)
            agent.rays = RaycastService.cast_rays(
                origin=agent.position,
                heading_deg=agent.angle_deg,
                fov_deg=ray_cfg["fov_degrees"],
                num_rays=ray_cfg["ray_count"],
                max_dist=ray_cfg["max_distance"],
                segments=self.map.segments,
                cpp_segments=self.cpp_segments,
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