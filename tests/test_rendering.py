"""Unit tests for Camera transforms and headless PygameRenderer smoke test."""

import os

import numpy as np
import pytest

from src.configs import load_arena, load_project_config
from src.rendering import Camera, PygameRenderer
from src.sim import get_observations, reset_world


def test_camera_transforms_clamp_pan_and_follow():
    cam = Camera(1280, 760, x=600.0, y=400.0, zoom=0.75, min_zoom=0.2, max_zoom=4.0)
    sx, sy = cam.world_to_screen(600.0, 400.0)
    assert (sx, sy) == (640, 380)

    wx, wy = cam.screen_to_world(640, 380)
    assert wx == pytest.approx(600.0)
    assert wy == pytest.approx(400.0)

    cam.adjust_zoom(100.0)
    assert cam.zoom == 4.0
    cam.adjust_zoom(0.001)
    assert cam.zoom == 0.2

    cam.pan(20.0, -10.0)
    assert cam.x == pytest.approx(600.0 - 20.0 / 0.2)
    assert cam.y == pytest.approx(400.0 + 10.0 / 0.2)

    cam.follow([0.0, 0.0], smoothing=0.5)
    assert cam.x == pytest.approx(250.0)
    assert cam.y == pytest.approx(225.0)


def test_pygame_renderer_headless_smoke(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    cfg = load_project_config("configs")
    arena = load_arena("configs/maps/ctf_pillars.yaml")
    state = reset_world(arena, team_size=2)
    rays, _ = get_observations(state, arena, cfg.agent)

    renderer = PygameRenderer(320, 240, "Test")
    cam = Camera(320, 240, x=600.0, y=400.0, zoom=0.25)
    pos_before = np.asarray(state.pos).copy()

    renderer.render(
        cam,
        arena,
        state,
        cfg.agent,
        rays=np.asarray(rays),
        watched_idx=0,
        show_rays=True,
        fps=60.0,
        hud_lines=["Test HUD"],
    )
    renderer.close()
    assert np.allclose(np.asarray(state.pos), pos_before)
