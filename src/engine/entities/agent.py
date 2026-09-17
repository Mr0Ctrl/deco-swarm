# src/engine/entities/agent.py
import math
from dataclasses import dataclass, field
from typing import List, Tuple
from engine.core.geometry import Point2D, wrap_angle_deg


@dataclass
class RayHit:
    endpoint: Point2D
    distance: float
    norm_distance: float  # [0.0, 1.0] (RL için normalize)
    has_hit: bool


class Agent:
    def __init__(
        self,
        agent_id: str,
        x: float,
        y: float,
        angle_deg: float = 0.0,
        radius: float = 16.0,
        linear_speed: float = 180.0,
        angular_speed: float = 200.0,
        health: float = 100.0,
    ):
        self.id = agent_id
        self.x = x
        self.y = y
        self.angle_deg = wrap_angle_deg(angle_deg)
        self.radius = radius
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.health = health

        # --- AJANIN KENDİ ALGI DURUMU (State) ---
        # Artık world sözlüklerinde savrulmuyor, ajana ait birer özellik:
        self.rays: List[RayHit] = []
        self.polar_target: Tuple[float, float] = (1.0, 0.0)  # (norm_dist, norm_angle)

    @property
    def position(self) -> Point2D:
        return (self.x, self.y)

    def set_position(self, pos: Point2D):
        self.x, self.y = pos

    def rotate(self, direction: float, dt: float):
        """direction: -1.0 (Sol), +1.0 (Sağ), 0.0 (Sabit)"""
        self.angle_deg = wrap_angle_deg(self.angle_deg + direction * self.angular_speed * dt)

    def move(self, move_forward: float, strafe_right: float, dt: float):
        """İleri/Geri ve Sağa/Sola kayma hareketlerini uygular."""
        rad = math.radians(self.angle_deg)
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)

        # Sağa dik vektör
        right_x = -forward_y
        right_y = forward_x

        vx = (forward_x * move_forward + right_x * strafe_right) * self.linear_speed
        vy = (forward_y * move_forward + right_y * strafe_right) * self.linear_speed

        self.x += vx * dt
        self.y += vy * dt