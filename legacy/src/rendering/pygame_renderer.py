# src/deco_swarm/rendering/pygame_renderer.py
from typing import Dict, Any
import pygame
from src.engine.world import World
from src.rendering.camera import Camera
from src.rendering.drawers import MapDrawer, RayDrawer, AgentDrawer, GoalDrawer

COLOR_BG = (24, 28, 36)
COLOR_TEXT = (220, 220, 220)


class PygameRenderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DECO-SWARM | Modüler Taktik Sahası")
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

    def render(self, world: World, camera: Camera, current_fps: float, hud_info: Dict[str, Any] = None):
        # 1. Ekranları Temizle
        self.screen.fill(COLOR_BG)
        self.overlay_surface.fill((0, 0, 0, 0))

        # 2. Katmanları Sırayla Çizdir
        # A) Hedef Alanı (Goal)
        self.goal_drawer.draw(
            overlay_surface=self.overlay_surface,
            main_surface=self.screen,
            camera=camera,
            position=world.goal_pos,
            radius=world.goal_radius,
        )

        # B) Duvarlar (Map)
        self.map_drawer.draw(
            surface=self.screen,
            camera=camera,
            segments=world.map.segments,
        )

        # C) Işınlar (Rays)
        for agent in world.agents.values():
            self.ray_drawer.draw(
                ray_surface=self.overlay_surface,
                hit_surface=self.screen,
                camera=camera,
                agent_pos=agent.position,
                rays=agent.rays,
            )

        # D) Ajanlar (Agents)
        for agent_id, agent in world.agents.items():
            self.agent_drawer.draw(
                surface=self.screen,
                camera=camera,
                position=agent.position,
                angle_deg=agent.angle_deg,
                radius=agent.radius,
            )

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
            for k, v in hud_info.items():
                lines.append(f"{k}: {v}")

        for idx, line in enumerate(lines):
            text_surf = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(text_surf, (15, 12 + idx * 18))

        pygame.display.flip()

    def close(self):
        pygame.quit()