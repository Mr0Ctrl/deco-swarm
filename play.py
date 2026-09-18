# play.py
import sys
import os
import yaml
import pygame

sys.path.insert(0, os.path.abspath("src"))

from src.engine.map_loader import MapLoader
from src.engine.world import World
from src.engine.controllers import KeyboardController
from src.rendering.camera import Camera
from src.rendering.pygame_renderer import PygameRenderer


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 1. Ayrık Konfigürasyonları Yükle
    render_cfg = load_yaml("configs/render.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")
    agent_cfg = load_yaml("configs/agent.yaml")
    map_cfg = MapLoader.load("configs/maps/arena_1v0.yaml")

    # 2. Dünyayı Başlat
    world = World(map_config=map_cfg, sim_config=sim_cfg, agent_config=agent_cfg)

    # 3. Görselleştirici ve Kamerayı Başlat
    screen_w = render_cfg["screen"]["width"]
    screen_h = render_cfg["screen"]["height"]
    renderer = PygameRenderer(screen_w, screen_h)
    camera = Camera(screen_w, screen_h)

    camera.x = map_cfg.width / 2.0
    camera.y = map_cfg.height / 2.0
    camera.zoom = render_cfg["camera"]["default_zoom"]

    controller = KeyboardController()
    clock = pygame.time.Clock()
    target_fps = render_cfg["screen"]["fps"]

    follow_agent = False
    dragging_camera = False
    last_mouse_pos = (0, 0)
    running = True

    while running:
        dt = clock.tick(target_fps) / 1000.0

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

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.adjust_zoom(1.1)
                elif event.y < 0:
                    camera.adjust_zoom(0.9)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
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

        # Kontrolcüden aksiyon al
        action = controller.get_action()

        world.step(
            actions={"agent_0": action.to_movement_tuple()},
            dt=dt,
        )

        if follow_agent and "agent_0" in world.agents:
            camera.follow(world.agents["agent_0"].position, smoothing=0.1)

        if world.is_goal_reached("agent_0"):
            world.reset()

        # Ekranda gösterilecek metrikler (Artık doğrudan ajanın kendi içinden okunuyor!)
        agent = world.agents["agent_0"]
        hud_info = {
            "Polar Mesafe (r)": f"{agent.polar_target[0]:.3f}",
            "Polar Açı (Δθ)": f"{agent.polar_target[1]:+.3f}",
            "Kamera": "Takip (C)" if follow_agent else "Serbest",
        }

        renderer.render(world, camera, current_fps=clock.get_fps(), hud_info=hud_info)

    renderer.close()


if __name__ == "__main__":
    main()