# src/deco_swarm/engine/agent.py
import math
from typing import Tuple


class Agent:
    def __init__(
        self,
        x: float,
        y: float,
        angle_deg: float = 0.0,
        radius: float = 16.0,
        linear_speed: float = 180.0,
        angular_speed: float = 200.0,
    ):
        self.x = x
        self.y = y
        self.angle_deg = angle_deg % 360.0
        self.radius = radius
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed

    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def rotate(self, direction: float, dt: float):
        """
        direction: -1.0 (Sola dön), +1.0 (Sağa dön), 0.0 (Dönme)
        """
        self.angle_deg = (self.angle_deg + direction * self.angular_speed * dt) % 360.0

    def move(self, move_forward: float, strafe_right: float, dt: float):
        """
        move_forward: +1.0 (İleri), -1.0 (Geri), 0.0
        strafe_right: +1.0 (Sağa kay), -1.0 (Sola kay), 0.0
        """
        rad = math.radians(self.angle_deg)
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)

        # Sağa dik vektör (90 derece saat yönü)
        right_x = -forward_y
        right_y = forward_x

        # İki eksenli hareket vektörü
        vx = (forward_x * move_forward + right_x * strafe_right) * self.linear_speed
        vy = (forward_y * move_forward + right_y * strafe_right) * self.linear_speed

        self.x += vx * dt
        self.y += vy * dt