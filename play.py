# play.py
import sys
import os
import yaml
import pygame

# src dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath("src"))

from src.engine.world import World
from src.rendering.pygame_renderer import PygameRenderer


def load_config():
    with open("configs/physics.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    world = World(
        width=cfg["screen"]["width"],
        height=cfg["screen"]["height"],
        ray_config=cfg["raycast"],
        agent_config=cfg["agent"],
    )

    renderer = PygameRenderer(
        width=cfg["screen"]["width"],
        height=cfg["screen"]["height"],
    )

    clock = pygame.time.Clock()
    target_fps = cfg["screen"]["fps"]
    running = True

    while running:
        # Delta time (saniye cinsinden adım süresi)
        dt = clock.tick(target_fps) / 1000.0

        # Olay Dinleme
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    renderer.toggle_rays()

        # Klavye Girdileri (Sürekli Basım)
        keys = pygame.key.get_pressed()

        move_fwd = 0.0
        strafe = 0.0
        rotate_dir = 0.0

        # İleri - Geri (W - S)
        if keys[pygame.K_w]:
            move_fwd += 1.0
        if keys[pygame.K_s]:
            move_fwd -= 1.0

        # Sağa - Sola Kayma (D - A)
        if keys[pygame.K_d]:
            strafe += 1.0
        if keys[pygame.K_a]:
            strafe -= 1.0

        # Dönüş (E - Q veya Sağ/Sol Oklar)
        if keys[pygame.K_e] or keys[pygame.K_RIGHT]:
            rotate_dir += 1.0
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            rotate_dir -= 1.0

        # Dünyayı 1 Adım İlerlet
        world.step(move_fwd=move_fwd, strafe=strafe, rotate_dir=rotate_dir, dt=dt)

        # Çizdir
        renderer.render(world, current_fps=clock.get_fps())

    renderer.close()


if __name__ == "__main__":
    main()