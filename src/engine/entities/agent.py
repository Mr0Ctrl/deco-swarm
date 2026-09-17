# src/engine/entities/agent.py
import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from engine.core.geometry import Point2D, wrap_angle_deg


@dataclass
class RayHit:
    endpoint: Point2D
    distance: float
    norm_distance: float       # [0.0, 1.0]
    has_hit: bool
    # Gelecek için genişletilmiş varlık özellikleri
    is_wall: float = 1.0
    is_friend: float = 0.0
    is_enemy: float = 0.0
    is_objective: float = 0.0
    rel_cos: float = 0.0       # Görülen ajanın cos açısı (duvarsa 0.0)
    rel_sin: float = 0.0       # Görülen ajanın sin açısı (duvarsa 0.0)


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
        self.max_health = health
        self.health = health
        self.ammo = 1.0  # Normalize mermi durumu

        # --- ALGI DURUMU ---
        self.rays: List[RayHit] = []
        self.polar_target: Tuple[float, float] = (1.0, 0.0)  # (norm_dist, norm_angle)

        # --- İÇ KİNEMATİK VE ÖNCEKİ AKSİYON (a_{t-1}) ---
        self.current_vel = (0.0, 0.0, 0.0)  # (v_ileri, v_yan, omega_donus) [-1, 1]
        self.last_action = (0.0, 0.0, 0.0)  # Önceki hareket kararı

    @property
    def position(self) -> Point2D:
        return (self.x, self.y)

    def set_position(self, pos: Point2D):
        self.x, self.y = pos

    def rotate(self, direction: float, dt: float):
        self.angle_deg = wrap_angle_deg(self.angle_deg + direction * self.angular_speed * dt)

    def move(self, move_forward: float, strafe_right: float, dt: float):
        # ÇAPRAZ HIZ HATASI DÜZELTMESİ (Quake Strafe-Run Fix):
        # Bileşke kuvvet 1.0'dan büyükse birim vektöre normalize et
        mag = math.hypot(move_forward, strafe_right)
        if mag > 1.0:
            move_forward /= mag
            strafe_right /= mag

        rad = math.radians(self.angle_deg)
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)
        right_x = -forward_y
        right_y = forward_x

        vx = (forward_x * move_forward + right_x * strafe_right) * self.linear_speed
        vy = (forward_y * move_forward + right_y * strafe_right) * self.linear_speed

        self.x += vx * dt
        self.y += vy * dt

    def get_observation(self) -> np.ndarray:
        """
        Model için 640 Float'lık resmi Ego-Merkezli Gözlem Vektörünü üretir:
          - 90 Ray x 7 Kanal = 630 Float
          - Polar Hedef (r, d_theta) = 2 Float
          - Kendi Hızları (v_x, v_y, omega) = 3 Float
          - Önceki Aksiyon (a_{t-1}) = 3 Float
          - Öz Durum (Can, Mermi) = 2 Float
        """
        obs = []

        # 1. Raycast Bloğu (90 x 7 = 630)
        for ray in self.rays:
            obs.extend([
                ray.norm_distance,
                ray.is_wall if ray.has_hit else 0.0,
                ray.is_friend,
                ray.is_enemy,
                ray.is_objective,
                ray.rel_cos,
                ray.rel_sin,
            ])

        # 2. Polar Hedef (2)
        obs.extend([self.polar_target[0], self.polar_target[1]])

        # 3. Kendi Hızları / Odometri (3)
        obs.extend([self.current_vel[0], self.current_vel[1], self.current_vel[2]])

        # 4. Önceki Aksiyon (3)
        obs.extend([self.last_action[0], self.last_action[1], self.last_action[2]])

        # 5. Öz Durum (2)
        norm_health = max(0.0, min(1.0, self.health / self.max_health))
        obs.extend([norm_health, self.ammo])

        return np.array(obs, dtype=np.float32)

