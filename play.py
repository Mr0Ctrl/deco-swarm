# play.py
"""
DECO-SWARM: interactive player & policy viewer (JAX).

  python play.py --mode human --map configs/maps/faz_1v0.1.yaml
  python play.py --mode watch --model-path checkpoints/model_final.msgpack --map configs/maps/faz_1v1.3.yaml

--model-path also accepts legacy PyTorch checkpoints (.pt); PyTorch does not need to be installed.
"""
import argparse
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from deco.checkpoint import load_checkpoint
from deco.config import load_configs
from deco.env import MOVE_TABLE, ROTATE_TABLE, NavEnv
from deco.maps import MapBank
from deco.model import RecurrentActorCritic, sample_action
from deco.rendering.camera import Camera
from deco.rendering.renderer import AgentView, PygameRenderer, SceneView


def parse_args():
    parser = argparse.ArgumentParser(description="DECO-SWARM: Interactive Player & Policy Viewer")
    parser.add_argument("--mode", type=str, default="human", choices=["human", "watch"],
                        help="'human' for keyboard control, 'watch' to view trained model")
    parser.add_argument("--model-path", type=str, default="checkpoints/model_final.msgpack",
                        help="Path to trained model checkpoint (.msgpack, or legacy .pt)")
    parser.add_argument("--map", type=str, default="configs/maps/faz_1v0.1.yaml",
                        help="Map configuration file to load")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for simulation/inference: auto, gpu, or cpu")
    return parser.parse_args()


def keyboard_action():
    """W/S forward-backward, A/D strafe, Q/E or arrows rotate -> (move_fwd, strafe, rotate_dir)."""
    keys = pygame.key.get_pressed()
    move_fwd = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
    strafe = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
    rotate_dir = float(keys[pygame.K_e] or keys[pygame.K_RIGHT]) - float(keys[pygame.K_q] or keys[pygame.K_LEFT])
    return np.array([move_fwd, strafe, rotate_dir])


def main():
    args = parse_args()
    if args.device == "cpu":
        jax.config.update("jax_platforms", "cpu")
    elif args.device in ("gpu", "cuda"):
        jax.config.update("jax_platforms", "cuda")

    # 1. Load configurations and the map
    configs = load_configs()
    render_cfg = configs.render
    bank = MapBank([args.map])
    map_data = bank.maps[0]

    # 2. Initialize the world (jitted single-environment functions)
    env = NavEnv(configs, bank)
    reset = jax.jit(partial(env.reset_to_map, 0))
    tick = jax.jit(env.world_tick)
    observe = jax.jit(env.observation)
    goal_reached = jax.jit(env.is_goal_reached)
    ray_endpoints = jax.jit(env.ray_endpoints)
    _, state = reset()

    # 3. Setup model if watch mode
    model = params = hidden = policy = None
    if args.mode == "watch":
        model = RecurrentActorCritic.from_agent_config(configs.agent)
        print(f"[Watch Mode] Loading model from {args.model_path} on {jax.default_backend()}...")
        if os.path.exists(args.model_path):
            params = load_checkpoint(args.model_path)["params"]
            print("[Watch Mode] Model weights loaded successfully.")
        else:
            print(f"[Warning] Checkpoint not found at {args.model_path}! Running with initial weights.")
            params = model.init(jax.random.PRNGKey(0), model.initial_hidden(1), jnp.zeros((1, env.obs_dim)))
        hidden = model.initial_hidden(1)

        @jax.jit
        def policy(params, hidden, obs):
            hidden, move_logits, rot_logits, _ = model.apply(params, hidden, obs[None])
            action, _ = sample_action(None, move_logits, rot_logits, deterministic=True)
            return hidden, action[0]

    # 4. Initialize renderer and camera
    screen_w = render_cfg["screen"]["width"]
    screen_h = render_cfg["screen"]["height"]
    renderer = PygameRenderer(screen_w, screen_h)
    camera = Camera(screen_w, screen_h)
    camera.x = map_data.width / 2.0
    camera.y = map_data.height / 2.0
    camera.zoom = render_cfg["camera"]["default_zoom"]

    clock = pygame.time.Clock()
    target_fps = render_cfg["screen"]["fps"]

    follow_agent = args.mode == "watch"
    dragging_camera = False
    last_mouse_pos = (0, 0)
    running = True

    # Policy runs at 10 Hz: one decision every `action_repeat` frames
    action_repeat = configs.env.action_repeat
    substep_counter = 0
    current_action = np.zeros(3)

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
                    _, state = reset()
                    if model is not None:
                        hidden = model.initial_hidden(1)
                elif event.key == pygame.K_c:
                    follow_agent = not follow_agent
            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    camera.adjust_zoom(1.1)
                elif event.y < 0:
                    camera.adjust_zoom(0.9)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                dragging_camera = True
                last_mouse_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                dragging_camera = False
            elif event.type == pygame.MOUSEMOTION and dragging_camera:
                camera.pan(event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1])
                last_mouse_pos = event.pos

        # Choose action
        if args.mode == "human":
            current_action = keyboard_action()
        else:
            if substep_counter % action_repeat == 0:
                hidden, act = policy(params, hidden, observe(state))
                act = np.asarray(act)
                current_action = np.concatenate([MOVE_TABLE[act[0]], ROTATE_TABLE[act[1]][None]])
            substep_counter += 1

        state = tick(state, jnp.asarray(current_action), dt)

        pos = np.asarray(state.pos)
        if follow_agent:
            camera.follow((float(pos[0]), float(pos[1])), smoothing=0.1)

        if bool(goal_reached(state)):
            _, state = reset()
            if model is not None:
                hidden = model.initial_hidden(1)
            pos = np.asarray(state.pos)

        polar = np.asarray(state.polar)
        hud_info = {
            "Mode": "AI Watch" if args.mode == "watch" else "Human (WASD)",
            "Map": map_data.name,
            "Polar Dist (r)": f"{polar[0]:.3f}",
            "Polar Heading (Δθ)": f"{polar[1]:+.3f}",
            "Camera": "Follow (C)" if follow_agent else "Free",
        }

        scene = SceneView(
            segments=map_data.segment_list,
            goal_pos=map_data.goal_pos,
            goal_radius=map_data.goal_radius,
            agents=[
                AgentView(
                    position=(float(pos[0]), float(pos[1])),
                    angle_deg=float(state.angle_deg),
                    radius=configs.agent.radius,
                    ray_endpoints=[tuple(p) for p in np.asarray(ray_endpoints(state)).tolist()],
                    ray_hits=np.asarray(state.ray_hit).tolist(),
                )
            ],
        )
        renderer.render(scene, camera, current_fps=clock.get_fps(), hud_info=hud_info)

    renderer.close()


if __name__ == "__main__":
    main()
