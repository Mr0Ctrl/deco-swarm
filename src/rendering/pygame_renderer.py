# src/deco_swarm/rendering/pygame_renderer.py
import math
import pygame
from src.engine.world import World

# Taktiksel Renk Paleti
COLOR_BG = (24, 28, 36)          # Koyu arduvaz
COLOR_WALL = (180, 190, 205)      # Duvarlar (Açık gri)
COLOR_AGENT = (46, 204, 113)      # Ajan (Yeşil)
COLOR_HEADING = (241, 196, 15)    # Ajanın baktığı yön (Sarı)
COLOR_RAY = (231, 76, 60, 90)     # Işınlar (Saydam Kırmızı)
COLOR_HIT = (231, 76, 60)         # Çarpışma noktası (Kırmızı)
COLOR_TEXT = (200, 200, 200)


class PygameRenderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("DECO-SWARM | Algı ve Fizik Test Sahası")
        self.font = pygame.font.SysFont("monospace", 14)
        
        # Saydam ışın çizgileri çizmek için alfa yüzeyi
        self.ray_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self.show_rays = True

    def toggle_rays(self):
        self.show_rays = not self.show_rays

    def render(self, world: World, current_fps: float):
        self.screen.fill(COLOR_BG)
        self.ray_surface.fill((0, 0, 0, 0))

        # 1. Duvarları Çiz
        for (x1, y1), (x2, y2) in world.map.segments:
            pygame.draw.line(self.screen, COLOR_WALL, (x1, y1), (x2, y2), 3)

        # 2. Işınlar (Her durumda çizilir, sadece çarpışmada nokta konur)
        if self.show_rays:
            ax, ay = world.agent.position
            for endpoint, dist, norm_dist, has_hit in world.latest_rays:
                ex, ey = endpoint
                # Ajan merkezinden ışının vardığı son noktaya çizgi
                pygame.draw.line(self.ray_surface, COLOR_RAY, (ax, ay), (ex, ey), 1)
                
                # Sadece duvara çarptıysa temas noktası çiz
                if has_hit:
                    pygame.draw.circle(self.screen, COLOR_HIT, (int(ex), int(ey)), 3)

            self.screen.blit(self.ray_surface, (0, 0))

        # 3. Ajanı Çiz
        ax, ay = int(world.agent.x), int(world.agent.y)
        pygame.draw.circle(self.screen, COLOR_AGENT, (ax, ay), int(world.agent.radius))
        
        # Ajanın baktığı yönü gösteren burun çizgisi
        rad = math.radians(world.agent.angle_deg)
        nose_len = world.agent.radius + 10.0
        nose_x = ax + math.cos(rad) * nose_len
        nose_y = ay + math.sin(rad) * nose_len
        pygame.draw.line(self.screen, COLOR_HEADING, (ax, ay), (nose_x, nose_y), 3)

        # 4. Bilgi Panelini Çiz
        info_lines = [
            f"FPS: {current_fps:.1f}",
            f"Işın Sayısı: {world.ray_config['ray_count']} (F1: Aç/Kapa)",
            f"Görüş Açısı: {world.ray_config['fov_degrees']}°",
            f"Kontroller: W/S: İleri/Geri | A/D: Kayma | Q/E: Dönüş",
        ]
        for idx, line in enumerate(info_lines):
            surf = self.font.render(line, True, COLOR_TEXT)
            self.screen.blit(surf, (15, 15 + idx * 20))

        pygame.display.flip()

    def close(self):
        pygame.quit()