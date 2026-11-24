# src/deco_swarm/rendering/drawers/goal_drawer.py
from typing import Tuple
import pygame
from src.rendering.camera import Camera

COLOR_GOAL_FILL = (52, 152, 219, 90)     # Mavi saydam dolgu
COLOR_GOAL_BORDER = (41, 128, 185)       # Dış mavi halka


class GoalDrawer:
    def draw(
        self,
        overlay_surface: pygame.Surface,
        main_surface: pygame.Surface,
        camera: Camera,
        position: Tuple[float, float],
        radius: float = 24.0,
    ):
        center_screen = camera.world_to_screen(position)
        radius_screen = camera.scale_to_screen(radius)
        border_thickness = camera.scale_to_screen(2.0)

        # Saydam iç alan
        pygame.draw.circle(overlay_surface, COLOR_GOAL_FILL, center_screen, radius_screen)
        # Belirgin dış çember
        pygame.draw.circle(main_surface, COLOR_GOAL_BORDER, center_screen, radius_screen, border_thickness)