"""Interactive 60 FPS CTF viewer and human/random testbed."""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from src.configs import load_arena, load_project_config
from src.rendering import Camera, PygameRenderer
from src.sim import get_observations, reset_world, step_world


def _read_keyboard_action() -> list[int]:
    """Map WASD/QE/Space keys into [move_idx, rot_idx, tag_idx]."""
    keys = pygame.key.get_pressed()
    move_idx = 0
    if keys[pygame.K_w]:
        move_idx = 1
    elif keys[pygame.K_s]:
        move_idx = 2
    elif keys[pygame.K_a]:
        move_idx = 3
    elif keys[pygame.K_d]:
        move_idx = 4

    rot_idx = 0
    if keys[pygame.K_q] or keys[pygame.K_LEFT]:
        rot_idx = 1
    elif keys[pygame.K_e] or keys[pygame.K_RIGHT]:
        rot_idx = 2

    tag_idx = 1 if keys[pygame.K_SPACE] else 0
    return [move_idx, rot_idx, tag_idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="DECO-SWARM v2 Interactive CTF Viewer")
    parser.add_argument("--mode", choices=["human", "random"], default="human")
    parser.add_argument("--map", default="configs/maps/ctf_pillars.yaml")
    parser.add_argument("--team-size", type=int, default=2, choices=[2, 3, 5])
    parser.add_argument("--agent", type=int, default=0, help="Controlled/watched agent index")
    args = parser.parse_args()

    cfg = load_project_config("configs")
    arena = load_arena(args.map, max_team_size=5)
    n_agents = 2 * args.team_size
    watched_idx = max(0, min(n_agents - 1, args.agent))

    # Step 1 physics tick (60 Hz) per frame for smooth visual motion while polling decisions at 10 Hz
    substep_ep_cfg = cfg.episode._replace(action_repeat=1, max_steps=cfg.episode.max_steps * cfg.episode.action_repeat)
    step_jit = jax.jit(
        lambda st, act: step_world(st, arena, act, cfg.agent, cfg.physics, substep_ep_cfg)
    )

    state = reset_world(arena, args.team_size)
    rays, _ = get_observations(state, arena, cfg.agent)

    renderer = PygameRenderer(cfg.render.screen_width, cfg.render.screen_height, cfg.render.title)
    camera = Camera(
        cfg.render.screen_width,
        cfg.render.screen_height,
        x=float(arena.width) * 0.5,
        y=float(arena.height) * 0.5,
        zoom=cfg.render.default_zoom,
        min_zoom=cfg.render.min_zoom,
        max_zoom=cfg.render.max_zoom,
    )

    clock = pygame.time.Clock()
    rng = np.random.default_rng(0)
    show_rays = True
    follow_agent = False
    dragging = False
    frame = 0
    actions_np = np.zeros((n_agents, 3), dtype=np.int32)

    running = True
    while running:
        clock.tick(cfg.render.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    show_rays = not show_rays
                elif event.key == pygame.K_F2:
                    state = reset_world(arena, args.team_size)
                    rays, _ = get_observations(state, arena, cfg.agent)
                elif event.key == pygame.K_c:
                    follow_agent = not follow_agent
                elif event.key == pygame.K_TAB:
                    watched_idx = (watched_idx + 1) % n_agents
            elif event.type == pygame.MOUSEWHEEL:
                camera.adjust_zoom(1.1 if event.y > 0 else 0.9)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                camera.pan(event.rel[0], event.rel[1])

        # 10 Hz decision cadence (every action_repeat=6 physics frames)
        if frame % cfg.episode.action_repeat == 0:
            if args.mode == "random":
                actions_np[:, 0] = rng.integers(0, 5, size=n_agents)
                actions_np[:, 1] = rng.integers(0, 3, size=n_agents)
                actions_np[:, 2] = rng.integers(0, 2, size=n_agents)
            else:
                actions_np.fill(0)
                actions_np[watched_idx] = _read_keyboard_action()

        state, (rays, _), _, terminated, truncated = step_jit(state, jnp.asarray(actions_np))
        frame += 1

        if bool(terminated) or bool(truncated):
            state = reset_world(arena, args.team_size)
            rays, _ = get_observations(state, arena, cfg.agent)

        if follow_agent:
            camera.follow(np.asarray(state.pos[watched_idx]), smoothing=0.1)

        hud = [
            f"Mode: {args.mode} | Agent: {watched_idx} ({'RED' if int(state.teams[watched_idx]) == 0 else 'BLUE'})",
            f"Time: {frame / cfg.physics.tick_rate:.1f}s | Follow(C): {'ON' if follow_agent else 'OFF'} | Rays(F1): {'ON' if show_rays else 'OFF'}",
            "Controls: WASD Move | QE Turn | Space Tag | Tab Agent | F2 Reset",
        ]
        renderer.render(
            camera,
            arena,
            state,
            cfg.agent,
            rays=np.asarray(rays),
            watched_idx=watched_idx,
            show_rays=show_rays,
            fps=clock.get_fps(),
            hud_lines=hud,
        )

    renderer.close()


if __name__ == "__main__":
    main()
