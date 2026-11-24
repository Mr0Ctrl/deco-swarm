# deco/rendering/drawers.py
"""Specialized drawers: map walls, goal zone, raycasts and agents."""
import math
from typing import Sequence, Tuple

import pygame

from deco.rendering.camera import Camera

Point = Tuple[float, float]

COLOR_WALL = (180, 190, 205)
COLOR_GOAL_FILL = (52, 152, 219, 90)     # Mavi saydam dolgu
COLOR_GOAL_BORDER = (41, 128, 185)       # Dış mavi halka
COLOR_RAY = (231, 76, 60, 60)            # Saydam kırmızı lazer
COLOR_HIT = (231, 76, 60)                # Çarpma noktası
COLOR_AGENT = (46, 204, 113)
COLOR_HEADING = (241, 196, 15)


class MapDrawer:
    def __init__(self, wall_color: Tuple[int, int, int] = COLOR_WALL):
        self.wall_color = wall_color

    def draw(self, surface: pygame.Surface, camera: Camera, segments: Sequence[Tuple[Point, Point]]):
        line_thickness = camera.scale_to_screen(3.0)
        for p1, p2 in segments:
            pygame.draw.line(
                surface, self.wall_color, camera.world_to_screen(p1), camera.world_to_screen(p2), line_thickness
            )


class GoalDrawer:
    def draw(
        self,
        overlay_surface: pygame.Surface,
        main_surface: pygame.Surface,
        camera: Camera,
        position: Point,
        radius: float = 24.0,
    ):
        center_screen = camera.world_to_screen(position)
        radius_screen = camera.scale_to_screen(radius)
        border_thickness = camera.scale_to_screen(2.0)

        # Saydam iç alan
        pygame.draw.circle(overlay_surface, COLOR_GOAL_FILL, center_screen, radius_screen)
        # Belirgin dış çember
        pygame.draw.circle(main_surface, COLOR_GOAL_BORDER, center_screen, radius_screen, border_thickness)


class RayDrawer:
    def __init__(self):
        self.visible = True

    def toggle(self):
        self.visible = not self.visible

    def draw(
        self,
        ray_surface: pygame.Surface,
        hit_surface: pygame.Surface,
        camera: Camera,
        agent_pos: Point,
        endpoints: Sequence[Point],
        hits: Sequence[bool],
    ):
        if not self.visible:
            return

        start_screen = camera.world_to_screen(agent_pos)
        ray_thickness = camera.scale_to_screen(1.0)
        hit_radius = camera.scale_to_screen(3.0)

        for endpoint, has_hit in zip(endpoints, hits):
            end_screen = camera.world_to_screen(endpoint)
            # Lazer çizgisi
            pygame.draw.line(ray_surface, COLOR_RAY, start_screen, end_screen, ray_thickness)
            # Çarpışma noktası
            if has_hit:
                pygame.draw.circle(hit_surface, COLOR_HIT, end_screen, hit_radius)


class AgentDrawer:
    def __init__(self, body_color: Tuple[int, int, int] = COLOR_AGENT):
        self.body_color = body_color

    def draw(self, surface: pygame.Surface, camera: Camera, position: Point, angle_deg: float, radius: float):
        center_screen = camera.world_to_screen(position)

        # 1. Gövde
        pygame.draw.circle(surface, self.body_color, center_screen, camera.scale_to_screen(radius))

        # 2. Baktığı yön (Burun çizgisi)
        rad = math.radians(angle_deg)
        nose_world_len = radius + 8.0
        nose_world_pos = (position[0] + math.cos(rad) * nose_world_len, position[1] + math.sin(rad) * nose_world_len)
        pygame.draw.line(
            surface, COLOR_HEADING, center_screen, camera.world_to_screen(nose_world_pos), camera.scale_to_screen(2.5)
        )
