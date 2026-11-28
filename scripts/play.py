"""Play CTF with the keyboard.  uv run python scripts/play.py --map ctf_pillars --others random"""
import argparse
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # leave GPU memory for others

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from deco_swarm.config import compile_arena, load_all
from deco_swarm.envs import CTFEnv
from deco_swarm.rendering.viewer import Viewer

ROOT = Path(__file__).resolve().parents[1]


def keyboard_action():
    k = pygame.key.get_pressed()
    move = 1 if k[pygame.K_w] else 2 if k[pygame.K_s] else 0
    strafe = 1 if k[pygame.K_a] else 2 if k[pygame.K_d] else 0
    turn = 1 if k[pygame.K_LEFT] else 2 if k[pygame.K_RIGHT] else 0
    fire = 1 if k[pygame.K_SPACE] else 0
    return [move, strafe, turn, fire]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="ctf_pillars")
    ap.add_argument("--others", choices=["idle", "random"], default="idle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = run until closed")
    args = ap.parse_args()

    cfg = load_all(ROOT / "configs")
    arena = compile_arena(ROOT / "configs" / "arenas" / f"{args.map}.yaml", cfg.simulation.physics)
    env = CTFEnv(cfg, arena)
    viewer = Viewer(env, cfg.render)
    step = jax.jit(env.step)
    decision_hz = cfg.simulation.physics.tick_hz / cfg.environment.episode.action_repeat

    key = jax.random.key(args.seed)
    key, k = jax.random.split(key)
    _, state = env.reset(k)
    selected, show_rays, frame = 0, True, 0
    clock = pygame.time.Clock()
    dims = jnp.array(env.action_dims)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_TAB:
                selected = (selected + 1) % env.num_agents
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                show_rays = not show_rays
            elif e.type == pygame.MOUSEWHEEL:
                viewer.zoom_by(1.1 ** e.y)

        key, k_act, k_step = jax.random.split(key, 3)
        if args.others == "random":
            actions = np.array(jax.random.randint(k_act, (env.num_agents, 4), 0, dims))
        else:
            actions = np.zeros((env.num_agents, 4), np.int32)
        actions[selected] = keyboard_action()

        _, state, reward, term, _, info = step(k_step, state, jnp.asarray(actions))
        if bool(info["events"].captured.any()):
            print("capture! score", info["score"].tolist())
        if bool(term):
            print("match over, final score", info["score"].tolist())

        viewer.draw(state, selected, show_rays)
        clock.tick(decision_hz)
        frame += 1
        if args.max_frames and frame >= args.max_frames:
            running = False
    viewer.close()


if __name__ == "__main__":
    main()