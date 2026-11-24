# deco/rendering/renderer.py
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import pygame

from deco.rendering.camera import Camera
from deco.rendering.drawers import AgentDrawer, GoalDrawer, MapDrawer, RayDrawer

COLOR_BG = (24, 28, 36)
COLOR_TEXT = (220, 220, 220)

Point = Tuple[float, float]


@dataclass
class AgentView:
    position: Point
    angle_deg: float
    radius: float
    ray_endpoints: Sequence[Point] = ()
    ray_hits: Sequence[bool] = ()


@dataclass
class SceneView:
    """Everything the renderer needs, as plain Python/numpy data (decoupled from the JAX state)."""
    segments: Sequence[Tuple[Point, Point]]
    goal_pos: Point
    goal_radius: float
    agents: List[AgentView] = field(default_factory=list)


class PygameRenderer:
    def __init__(self, width: int, height: int, title: str = "DECO-SWARM | Modüler Taktik Sahası"):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 13)

        # Saydam çizimler için katman yüzeyi
        self.overlay_surface = pygame.Surface((width, height), pygame.SRCALPHA)

        # Modüler Çiziciler (Drawers)
        self.map_drawer = MapDrawer()
        self.ray_drawer = RayDrawer()
        self.agent_drawer = AgentDrawer()
        self.goal_drawer = GoalDrawer()

    def toggle_rays(self):
        self.ray_drawer.toggle()

    def render(self, scene: SceneView, camera: Camera, current_fps: float, hud_info: Dict[str, Any] = None):
        # 1. Ekranları Temizle
        self.screen.fill(COLOR_BG)
        self.overlay_surface.fill((0, 0, 0, 0))

        # 2. Katmanları Sırayla Çizdir: hedef, duvarlar, ışınlar, ajanlar
        self.goal_drawer.draw(self.overlay_surface, self.screen, camera, scene.goal_pos, scene.goal_radius)
        self.map_drawer.draw(self.screen, camera, scene.segments)
        for agent in scene.agents:
            self.ray_drawer.draw(
                self.overlay_surface, self.screen, camera, agent.position, agent.ray_endpoints, agent.ray_hits
            )
        for agent in scene.agents:
            self.agent_drawer.draw(self.screen, camera, agent.position, agent.angle_deg, agent.radius)

        # Saydam katmanı ana ekrana yapıştır
        self.screen.blit(self.overlay_surface, (0, 0))

        # 3. Bilgi Paneli (HUD)
        lines = [
            f"FPS: {current_fps:.1f} | Zoom: {camera.zoom:.2f}x",
            "Kontroller: W/S: İleri/Geri | A/D: Kayma | Q/E: Dönüş",
            "Kamera: Fare Tekerleği: Zoom | Sağ Tık Sürükle: Pan | C: Ajanı Takip Et",
            "F1: Işınları Aç/Kapa | F2: Reset",
        ]
        if hud_info:
            lines.extend(f"{k}: {v}" for k, v in hud_info.items())

        for idx, line in enumerate(lines):
            self.screen.blit(self.font.render(line, True, COLOR_TEXT), (15, 12 + idx * 18))

        pygame.display.flip()

    def close(self):
        pygame.quit()
