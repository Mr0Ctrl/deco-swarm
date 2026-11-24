# src/deco_swarm/rendering/drawers/map_drawer.py
from typing import List, Tuple
import pygame
from src.rendering.camera import Camera

COLOR_WALL = (180, 190, 205)


class MapDrawer:
    def __init__(self, wall_color: Tuple[int, int, int] = COLOR_WALL):
        self.wall_color = wall_color

    def draw(
        self,
        surface: pygame.Surface,
        camera: Camera,
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    ):
        line_thickness = camera.scale_to_screen(3.0)

        for (x1, y1), (x2, y2) in segments:
            # Dünya koordinatlarını ekran koordinatına çevir
            p1_screen = camera.world_to_screen((x1, y1))
            p2_screen = camera.world_to_screen((x2, y2))
            
            pygame.draw.line(surface, self.wall_color, p1_screen, p2_screen, line_thickness)