# play.py
import sys
import os
import math
import yaml
import pygame

# src dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath("src"))

from src.engine.map_loader import MapLoader
from src.engine.world import World
from src.engine.controllers import KeyboardController
from src.engine.perception import compute_ego_polar_target
from src.rendering.camera import Camera
from src.rendering.pygame_renderer import PygameRenderer


def main():
    # 1. Konfigürasyonları Yükle
    with open("configs/physics.yaml", "r", encoding="utf-8") as f:
        phys_cfg = yaml.safe_load(f)

    # 1400x1000'lik U-Labirent haritamızı yüklüyoruz
    map_cfg = MapLoader.load("configs/maps/arena_1v0.yaml")

    # 2. Dünyayı ve Görselleştiriciyi Başlat
    world = World(
        map_config=map_cfg,
        ray_config=phys_cfg["raycast"],
        agent_config=phys_cfg["agent"],
    )

    screen_w = phys_cfg["screen"]["width"]
    screen_h = phys_cfg["screen"]["height"]
    renderer = PygameRenderer(screen_w, screen_h)
    camera = Camera(screen_w, screen_h)

    # Kamerayı haritanın merkezine odakla ve haritayı görecek şekilde biraz uzaklaştır
    camera.x = map_cfg.width / 2.0
    camera.y = map_cfg.height / 2.0
    camera.zoom = 0.75

    controller = KeyboardController()
    clock = pygame.time.Clock()
    target_fps = phys_cfg["screen"]["fps"]

    follow_agent = False
    dragging_camera = False
    last_mouse_pos = (0, 0)
    running = True

    # Haritanın maksimum diyagonal mesafesi (Normalize polar mesafe için)
    max_map_dist = math.hypot(map_cfg.width, map_cfg.height)

    while running:
        dt = clock.tick(target_fps) / 1000.0

        # --- OLAYLARI DİNLE ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    renderer.toggle_rays()
                elif event.key == pygame.K_F2:
                    world.reset()
                elif event.key == pygame.K_c:
                    follow_agent = not follow_agent

            # Fare Tekerleği ile ZOOM
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.adjust_zoom(1.1)  # Yakınlaş
                elif event.y < 0:
                    camera.adjust_zoom(0.9)  # Uzaklaş

            # Sağ Tık ile Haritayı SÜRÜKLEME (PAN)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:  # Sağ tık
                    dragging_camera = True
                    last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    dragging_camera = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging_camera:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    camera.pan(dx, dy)
                    last_mouse_pos = event.pos

        # --- SİMÜLASYON ADIMI ---
        # Kontrolcüden aksiyonu al (LSP ilkesi: İleride buraya AIController bağlanacak)
        action = controller.get_action()

        # 'agent_0' için aksiyonu ilet
        world.step(
            actions={"agent_0": action.to_movement_tuple()},
            dt=dt,
        )

        # Kamera Ajanı Takip Ediyorsa
        if follow_agent and "agent_0" in world.agents:
            camera.follow(world.agents["agent_0"].position, smoothing=0.1)

        # --- HEDEFE ULAŞMA KONTROLÜ ---
        goal_reached = world.is_goal_reached("agent_0")
        if goal_reached:
            # Hedefe ulaşıldığında ajanı yeniden başlat
            world.reset()

        # --- EGO-POLAR ALGI METRİKLERİ ---
        agent = world.agents["agent_0"]
        norm_dist, norm_angle = compute_ego_polar_target(
            agent_pos=agent.position,
            agent_angle_deg=agent.angle_deg,
            target_pos=world.goal_pos,
            max_map_distance=max_map_dist,
        )

        hud_info = {
            "Polar Mesafe (r)": f"{norm_dist:.3f} (0=Vardı, 1=Uzak)",
            "Polar Açı (Δθ)": f"{norm_angle:+.3f} (0=Tam Karşıda, +=Sağ, -=Sol)",
            "Ateş Tuşu (Space)": "BASILDI" if action.fire else "Boşta",
            "Kamera Modu": "Ajan Takip (C)" if follow_agent else "Serbest Gezinme",
        }

        # --- ÇİZDİR ---
        renderer.render(world, camera, current_fps=clock.get_fps(), hud_info=hud_info)

    renderer.close()


if __name__ == "__main__":
    main()