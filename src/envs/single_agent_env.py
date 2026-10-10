# src/envs/single_agent_env.py
import math
import yaml
from typing import Dict, Any, Optional, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from engine.map_loader import MapLoader
from engine.world import World
from engine.controllers import ActionDecoder
from engine.core.geometry import distance
from envs.curriculum import MapCurriculum


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class SingleAgentMazeEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for Phase 1 Navigation.
    Configured for single agent ('agent_0') to maintain seamless compatibility
    with future multi-agent phases.
    
    Key Features:
      - Action Repeat (N=6): 60 Hz continuous physics downsampled to 10 Hz agent decisions.
      - 640-float Ego-Centric Observation Vector.
      - Multi-Discrete Action Space: [Move: 5, Rotate: 3].
      - Sub-Curriculum integration across Phase 1 map tiers (v0 -> v1 -> v2).
    """

    metadata = {"render_modes": ["human"], "name": "deco_swarm_maze_v0"}

    def __init__(
        self,
        sim_config_path: str = "configs/simulation.yaml",
        agent_config_path: str = "configs/agent.yaml",
        render_config_path: str = "configs/render.yaml",
        curriculum: Optional[MapCurriculum] = None,
        default_map_path: str = "configs/maps/faz_1v0.1.yaml",
        max_steps: int = 500,  # 500 decisions at 10 Hz = 50 seconds in simulation
        action_repeat: int = 6,  # 6 physics substeps per RL action (60 Hz / 10 Hz)
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.sim_cfg = load_yaml(sim_config_path)
        self.agent_cfg = load_yaml(agent_config_path)
        self.render_cfg = load_yaml(render_config_path)

        self.curriculum = curriculum
        self.current_map_path = (
            self.curriculum.current_map_path if self.curriculum else default_map_path
        )
        map_config = MapLoader.load(self.current_map_path)

        self.world = World(
            map_config=map_config,
            sim_config=self.sim_cfg,
            agent_config=self.agent_cfg,
        )

        self.max_steps = max_steps
        self.action_repeat = action_repeat
        self.physics_dt = 1.0 / self.sim_cfg["physics"].get("fps", 60.0)
        self.render_mode = render_mode

        # Multi-agent PettingZoo agent identification
        self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]

        # Observation space: 640 floats normalized to [-1.0, 1.0] or [0.0, 1.0]
        self.observation_spaces = {
            "agent_0": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(640,),
                dtype=np.float32,
            )
        }

        # Multi-discrete action space: [Movement: 5, Rotation: 3]
        self.action_spaces = {
            "agent_0": spaces.MultiDiscrete([5, 3])
        }

        # Episode state tracking
        self.step_count = 0
        self.prev_dist_to_goal = 0.0

        # Exploration / Novelty tracking (Episodic spatial cell visit bonus)
        self.cell_size = 40.0  # ~Diameter of agent
        self.visited_cells = set()
        self.exploration_reward_weight = 0.02  # Reward for entering a new cell

        # Optional Pygame visualization
        self.renderer = None
        self.camera = None
        if self.render_mode == "human":
            self._init_renderer()

    def _init_renderer(self):
        from rendering.pygame_renderer import PygameRenderer
        from rendering.camera import Camera

        screen_w = self.render_cfg["screen"]["width"]
        screen_h = self.render_cfg["screen"]["height"]
        self.renderer = PygameRenderer(screen_w, screen_h)
        self.camera = Camera(screen_w, screen_h)
        self._update_camera_bounds()

    def _update_camera_bounds(self):
        if self.camera:
            self.camera.x = self.world.map_config.width / 2.0
            self.camera.y = self.world.map_config.height / 2.0
            self.camera.zoom = self.render_cfg["camera"]["default_zoom"]

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        self.agents = self.possible_agents[:]
        self.step_count = 0

        # 1. Update map from curriculum if it changed
        target_map = (
            self.curriculum.current_map_path if self.curriculum else self.current_map_path
        )
        if target_map != self.current_map_path:
            self.current_map_path = target_map
            map_config = MapLoader.load(self.current_map_path)
            self.world.load_map(map_config)
            self._update_camera_bounds()
        else:
            self.world.reset()

        agent = self.world.agents["agent_0"]
        self.prev_dist_to_goal = distance(agent.position, self.world.goal_pos)

        # Reset exploration tracking for the new episode
        self.visited_cells.clear()
        start_cell = (int(agent.x // self.cell_size), int(agent.y // self.cell_size))
        self.visited_cells.add(start_cell)

        observations = {"agent_0": agent.get_observation()}
        infos = {
            "agent_0": {
                "map_name": self.world.map_config.name,
                "curriculum_tier": (
                    self.curriculum.current_tier.name if self.curriculum else "None"
                ),
            }
        }

        if self.render_mode == "human":
            self.render()

        return observations, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        if "agent_0" not in actions:
            return {}, {}, {}, {}, {}

        self.step_count += 1
        raw_action = actions["agent_0"]

        # Decode discrete action indices (move_idx, rot_idx) to physical velocities
        decoded_action = ActionDecoder.decode((raw_action[0], raw_action[1]))

        # Action Repeat: Run N physics substeps at 60 Hz per 1 agent decision
        collided_during_action = False
        goal_reached = False

        for _ in range(self.action_repeat):
            self.world.step({"agent_0": decoded_action}, dt=self.physics_dt)
            agent = self.world.agents["agent_0"]
            if agent.collided:
                collided_during_action = True
            if self.world.is_goal_reached("agent_0"):
                goal_reached = True
                break

        agent = self.world.agents["agent_0"]
        curr_dist = distance(agent.position, self.world.goal_pos)

        # --- REWARD CALCULATION ---
        # 1. Potential-based progress delta (Positive when closing in, negative when moving away)
        dist_progress = (self.prev_dist_to_goal - curr_dist) * 0.05
        self.prev_dist_to_goal = curr_dist

        # 2. Step time penalty to discourage loitering
        time_penalty = -0.01

        # 3. Collision penalty
        collision_penalty = -0.05 if collided_during_action else 0.0

        # 4. Episodic Exploration / Novelty Bonus (Rewards uncovering new map areas)
        current_cell = (int(agent.x // self.cell_size), int(agent.y // self.cell_size))
        if current_cell not in self.visited_cells:
            self.visited_cells.add(current_cell)
            exploration_reward = self.exploration_reward_weight
        else:
            exploration_reward = 0.0

        # 5. Terminal reward
        terminal_reward = 10.0 if goal_reached else 0.0

        step_reward = (
            dist_progress
            + time_penalty
            + collision_penalty
            + exploration_reward
            + terminal_reward
        )

        # --- TERMINATION & TRUNCATION ---
        terminated = goal_reached
        truncated = (self.step_count >= self.max_steps) and not terminated

        if terminated or truncated:
            self.agents = []
            if self.curriculum:
                self.curriculum.record_episode(is_success=goal_reached, steps=self.step_count)

        observations = {"agent_0": agent.get_observation()}
        rewards = {"agent_0": float(step_reward)}
        terminations = {"agent_0": terminated}
        truncations = {"agent_0": truncated}
        infos = {
            "agent_0": {
                "is_success": goal_reached,
                "step_count": self.step_count,
                "dist_to_goal": curr_dist,
                "collided": collided_during_action,
                "explored_cells": len(self.visited_cells),
                "map_name": self.world.map_config.name,
            }
        }

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        if self.renderer is None or self.camera is None:
            return

        import pygame

        # Process window close events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        agent = self.world.agents.get("agent_0")
        metrics = {}
        if agent:
            # Smooth camera tracking
            self.camera.follow(agent.position, smoothing=0.1)
            metrics = {
                "Map": self.world.map_config.name,
                "Step": f"{self.step_count}/{self.max_steps}",
                "Dist to Goal": f"{self.prev_dist_to_goal:.1f} px",
                "Curriculum": (
                    f"{self.curriculum.current_tier.name} ({self.curriculum.success_rate:.0%})"
                    if self.curriculum
                    else "Manual"
                ),
            }

        self.renderer.render(self.world, self.camera, metrics=metrics)

    def close(self):
        if self.renderer is not None:
            import pygame
            pygame.quit()
            self.renderer = None
            self.camera = None
