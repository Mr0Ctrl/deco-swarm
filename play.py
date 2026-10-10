# play.py
import sys
import os
import argparse
import yaml
import pygame
import torch

sys.path.insert(0, os.path.abspath("src"))

from engine.map_loader import MapLoader
from engine.world import World
from engine.controllers import KeyboardController, ActionDecoder
from rendering.camera import Camera
from rendering.pygame_renderer import PygameRenderer
from models.actor_critic import RecurrentActorCritic


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="DECO-SWARM: Interactive Player & Policy Viewer")
    parser.add_argument("--mode", type=str, default="human", choices=["human", "watch"],
                        help="'human' for keyboard control, 'watch' to view trained model")
    parser.add_argument("--model-path", type=str, default="checkpoints/model_final.pt",
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--map", type=str, default="configs/maps/faz_1v0.1.yaml",
                        help="Map configuration file to load")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for model inference: auto, cuda, or cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    render_cfg = load_yaml("configs/render.yaml")
    sim_cfg = load_yaml("configs/simulation.yaml")
    agent_cfg = load_yaml("configs/agent.yaml")
    map_cfg = MapLoader.load(args.map)

    # 2. Initialize World
    world = World(map_config=map_cfg, sim_config=sim_cfg, agent_config=agent_cfg)

    # 3. Setup Model if watch mode
    model = None
    device = None
    hidden_state = None
    if args.mode == "watch":
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        print(f"[Watch Mode] Loading model from {args.model_path} onto {device}...")
        model = RecurrentActorCritic().to(device)
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            hidden_state = model.initial_hidden_state(batch_size=1, device=device)
            print("[Watch Mode] Model weights loaded successfully.")
        else:
            print(f"[Warning] Checkpoint not found at {args.model_path}! Running with initial weights.")

    # 4. Initialize Renderer and Camera
    screen_w = render_cfg["screen"]["width"]
    screen_h = render_cfg["screen"]["height"]
    renderer = PygameRenderer(screen_w, screen_h)
    camera = Camera(screen_w, screen_h)

    camera.x = map_cfg.width / 2.0
    camera.y = map_cfg.height / 2.0
    camera.zoom = render_cfg["camera"]["default_zoom"]

    keyboard_controller = KeyboardController()
    clock = pygame.time.Clock()
    target_fps = render_cfg["screen"]["fps"]

    follow_agent = (args.mode == "watch")
    dragging_camera = False
    last_mouse_pos = (0, 0)
    running = True

    # Action repeat rate for watch mode (6 substeps = 10 Hz decision)
    action_repeat = 6
    substep_counter = 0
    current_action_tuple = (0.0, 0.0, 0.0)

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
                    if model is not None:
                        hidden_state = model.initial_hidden_state(batch_size=1, device=device)
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

        # Choose Action
        if args.mode == "human":
            action = keyboard_controller.get_action()
            current_action_tuple = action.to_movement_tuple()
        else:
            # Watch Mode: Run policy at 10 Hz (every 6 frames)
            if substep_counter % action_repeat == 0 and "agent_0" in world.agents:
                agent = world.agents["agent_0"]
                obs = torch.tensor(agent.get_observation(), dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    action_t, _, _, hidden_state = model.get_action(obs, hidden_state, deterministic=True)
                act_indices = action_t.cpu().numpy()[0]
                current_action_tuple = ActionDecoder.decode((act_indices[0], act_indices[1]))
            substep_counter += 1

        world.step(
            actions={"agent_0": current_action_tuple},
            dt=dt,
        )

        if follow_agent and "agent_0" in world.agents:
            camera.follow(world.agents["agent_0"].position, smoothing=0.1)

        if world.is_goal_reached("agent_0"):
            world.reset()
            if model is not None:
                hidden_state = model.initial_hidden_state(batch_size=1, device=device)

        agent = world.agents.get("agent_0")
        hud_info = {}
        if agent:
            hud_info = {
                "Mode": "AI Watch" if args.mode == "watch" else "Human (WASD)",
                "Map": map_cfg.name,
                "Polar Dist (r)": f"{agent.polar_target[0]:.3f}",
                "Polar Heading (Δθ)": f"{agent.polar_target[1]:+.3f}",
                "Camera": "Follow (C)" if follow_agent else "Free",
            }

        renderer.render(world, camera, current_fps=clock.get_fps(), hud_info=hud_info)

    renderer.close()


if __name__ == "__main__":
    main()