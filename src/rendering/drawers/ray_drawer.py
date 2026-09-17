# src/rendering/drawers/ray_drawer.py
from typing import List, Tuple
import pygame
from rendering.camera import Camera

COLOR_RAY = (231, 76, 60, 60)   # Saydam kırmızı lazer
COLOR_HIT = (231, 76, 60)       # Çarpma noktası


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
        agent_pos: Tuple[float, float],
        rays: list,
    ):
        if not self.visible or not rays:
            return

        start_screen = camera.world_to_screen(agent_pos)
        ray_thickness = camera.scale_to_screen(1.0)
        hit_radius = camera.scale_to_screen(3.0)

        # Dataclass niteliklerine doğrudan isimleriyle erişiyoruz (Temiz Kod)
        for hit in rays:
            end_screen = camera.world_to_screen(hit.endpoint)

            # Lazer çizgisi
            pygame.draw.line(ray_surface, COLOR_RAY, start_screen, end_screen, ray_thickness)

            # Çarpışma noktası
            if hit.has_hit:
                pygame.draw.circle(hit_surface, COLOR_HIT, end_screen, hit_radius)