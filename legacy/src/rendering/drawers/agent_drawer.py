# src/deco_swarm/rendering/drawers/agent_drawer.py
import math
from typing import Tuple
import pygame
from src.rendering.camera import Camera

COLOR_AGENT = (46, 204, 113)
COLOR_HEADING = (241, 196, 15)


class AgentDrawer:
    def __init__(self, body_color: Tuple[int, int, int] = COLOR_AGENT):
        self.body_color = body_color

    def draw(
        self,
        surface: pygame.Surface,
        camera: Camera,
        position: Tuple[float, float],
        angle_deg: float,
        radius: float,
    ):
        center_screen = camera.world_to_screen(position)
        radius_screen = camera.scale_to_screen(radius)

        # 1. Gövde
        pygame.draw.circle(surface, self.body_color, center_screen, radius_screen)

        # 2. Baktığı yön (Burun çizgisi)
        rad = math.radians(angle_deg)
        nose_world_len = radius + 8.0
        nose_world_pos = (
            position[0] + math.cos(rad) * nose_world_len,
            position[1] + math.sin(rad) * nose_world_len,
        )
        nose_screen = camera.world_to_screen(nose_world_pos)
        line_thickness = camera.scale_to_screen(2.5)

        pygame.draw.line(surface, COLOR_HEADING, center_screen, nose_screen, line_thickness)