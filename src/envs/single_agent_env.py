# src/envs/single_agent_env.py
import os
import math
import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, Set
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
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ==============================================================================
# CONFIGURATION DATA CLASSES
# ==============================================================================

@dataclass
class RewardConfig:
    """
    All reward and penalty coefficients in one place.
    You can easily modify these values directly in python or via configs/environment.yaml.
    """
    progress_weight: float = 0.01       # Distance progress gradient (+ when approaching goal)
    time_penalty: float = -1.1          # Urgency penalty deducted at each decision step
    collision_penalty: float = -0.5     # Penalty applied if the agent bumps into a wall
    exploration_weight: float = 0.1     # Novelty bonus for visiting a new 2D grid cell
    cell_size: float = 120.0            # Grid cell size in pixels for spatial visit tracking
    goal_reward: float = 100.0          # Terminal bonus awarded upon reaching the goal
    timeout_penalty: float = -1.0       # Terminal penalty if episode runs out of steps

    @classmethod
    def from_dict(cls, data: dict) -> "RewardConfig":
        return cls(
            progress_weight=float(data.get("progress_weight", cls.progress_weight)),
            time_penalty=float(data.get("time_penalty", cls.time_penalty)),
            collision_penalty=float(data.get("collision_penalty", cls.collision_penalty)),
            exploration_weight=float(data.get("exploration_weight", cls.exploration_weight)),
            cell_size=float(data.get("cell_size", cls.cell_size)),
            goal_reward=float(data.get("goal_reward", cls.goal_reward)),
            timeout_penalty=float(data.get("timeout_penalty", cls.timeout_penalty)),
        )


@dataclass
class EnvConfig:
    """
    Simulation timing, horizons, and config file paths.
    """
    max_steps: int = 100                # Maximum decision steps per episode before truncation (10 Hz)
    action_repeat: int = 6              # Physics substeps per agent decision (60 Hz / 6 = 10 Hz)
    default_map_path: str = "configs/maps/faz_1v0.1.yaml"
    env_config_path: str = "configs/environment.yaml"
    sim_config_path: str = "configs/simulation.yaml"
    agent_config_path: str = "configs/agent.yaml"
    render_config_path: str = "configs/render.yaml"

    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        return cls(
            max_steps=int(data.get("max_steps", cls.max_steps)),
            action_repeat=int(data.get("action_repeat", cls.action_repeat)),
            default_map_path=str(data.get("default_map", cls.default_map_path)),
        )


# ==============================================================================
# ENVIRONMENT CLASS
# ==============================================================================

class SingleAgentMazeEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv wrapper for Phase 1 Navigation.
    Configured for single agent ('agent_0') to maintain seamless compatibility
    with future multi-agent phases.

    Easy configuration access:
      - env.rewards: Access/modify reward parameters (RewardConfig)
      - env.cfg:     Access/modify timing & episode limits (EnvConfig)
    """

    metadata = {"render_modes": ["human"], "name": "deco_swarm_maze_v0"}

    def __init__(
        self,
        env_config: Optional[EnvConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        curriculum: Optional[MapCurriculum] = None,
        render_mode: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # 1. Load Configurations (YAML file defaults + dataclass overrides + kwargs)
        self.env_cfg = env_config or EnvConfig()
        yaml_env_data = load_yaml(self.env_cfg.env_config_path)

        if "episode" in yaml_env_data and env_config is None:
            self.env_cfg = EnvConfig.from_dict(yaml_env_data["episode"])

        # Allow passing max_steps or action_repeat as direct kwargs
        if "max_steps" in kwargs:
            self.env_cfg.max_steps = int(kwargs["max_steps"])
        if "action_repeat" in kwargs:
            self.env_cfg.action_repeat = int(kwargs["action_repeat"])

        # Rewards Configuration
        if reward_config is not None:
            self.reward_cfg = reward_config
        elif "rewards" in yaml_env_data:
            self.reward_cfg = RewardConfig.from_dict(yaml_env_data["rewards"])
        else:
            self.reward_cfg = RewardConfig()

        # Engine sub-configurations
        self.sim_cfg = load_yaml(self.env_cfg.sim_config_path)
        self.agent_cfg = load_yaml(self.env_cfg.agent_config_path)
        self.render_cfg = load_yaml(self.env_cfg.render_config_path)

        # 2. Curriculum & World Setup
        self.curriculum = curriculum
        self.current_map_path = (
            self.curriculum.current_map_path if self.curriculum else self.env_cfg.default_map_path
        )
        map_config = MapLoader.load(self.current_map_path)

        self.world = World(
            map_config=map_config,
            sim_config=self.sim_cfg,
            agent_config=self.agent_cfg,
        )

        self.physics_dt = 1.0 / self.sim_cfg.get("physics", {}).get("fps", 60.0)
        self.render_mode = render_mode

        # 3. PettingZoo Spaces
        self.possible_agents = ["agent_0"]
        self.agents = self.possible_agents[:]

        self.observation_spaces = {
            "agent_0": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(640,),
                dtype=np.float32,
            )
        }
        self.action_spaces = {
            "agent_0": spaces.MultiDiscrete([5, 3])
        }

        # 4. Episode Runtime State
        self.step_count = 0
        self.prev_dist_to_goal = 0.0
        self.visited_cells: Set[Tuple[int, int]] = set()

        # 5. Optional Renderer
        self.renderer = None
        self.camera = None
        if self.render_mode == "human":
            self._init_renderer()

    # --- Convenience Properties for Easy Access ---
    @property
    def rewards(self) -> RewardConfig:
        return self.reward_cfg

    @property
    def cfg(self) -> EnvConfig:
        return self.env_cfg

    @property
    def max_steps(self) -> int:
        return self.env_cfg.max_steps

    @max_steps.setter
    def max_steps(self, val: int):
        self.env_cfg.max_steps = val

    @property
    def action_repeat(self) -> int:
        return self.env_cfg.action_repeat

    @action_repeat.setter
    def action_repeat(self, val: int):
        self.env_cfg.action_repeat = val

    # --- PettingZoo API Methods ---
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

        # Update map from curriculum (picks new map on reset if domain randomization is on)
        target_map = (
            self.curriculum.get_episode_map() if self.curriculum else self.current_map_path
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

        # Reset episodic spatial exploration grid
        self.visited_cells.clear()
        start_cell = (
            int(agent.x // self.reward_cfg.cell_size),
            int(agent.y // self.reward_cfg.cell_size),
        )
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

        # 1. Action Decoding: (move_idx, rot_idx) -> continuous velocities
        decoded_action = ActionDecoder.decode((raw_action[0], raw_action[1]))

        # 2. Action Repeat: Run N physics substeps at 60 Hz per decision step
        collided_during_action = False
        goal_reached = False

        for _ in range(self.env_cfg.action_repeat):
            self.world.step({"agent_0": decoded_action}, dt=self.physics_dt)
            agent = self.world.agents["agent_0"]
            if agent.collided:
                collided_during_action = True
            if self.world.is_goal_reached("agent_0"):
                goal_reached = True
                break

        agent = self.world.agents["agent_0"]
        curr_dist = distance(agent.position, self.world.goal_pos)

        # 3. Termination / Truncation Checks
        terminated = goal_reached
        truncated = (self.step_count >= self.env_cfg.max_steps) and not terminated

        # 4. Reward Computation via Config
        step_reward, reward_breakdown = self._compute_step_reward(
            curr_dist=curr_dist,
            collided=collided_during_action,
            agent_pos=agent.position,
            goal_reached=goal_reached,
            timed_out=truncated,
        )

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
                "reward_breakdown": reward_breakdown,
            }
        }

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _compute_step_reward(
        self,
        curr_dist: float,
        collided: bool,
        agent_pos: Tuple[float, float],
        goal_reached: bool,
        timed_out: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the step reward using self.reward_cfg.
        Returns the scalar reward and a detailed breakdown dictionary.
        """
        cfg = self.reward_cfg

        # 1. Distance progress delta
        dist_progress = (self.prev_dist_to_goal - curr_dist) * cfg.progress_weight
        self.prev_dist_to_goal = curr_dist

        # 2. Time / step urgency penalty
        time_pen = cfg.time_penalty

        # 3. Collision penalty
        coll_pen = cfg.collision_penalty if collided else 0.0

        # 4. Episodic spatial exploration bonus
        cell = (
            int(agent_pos[0] // cfg.cell_size),
            int(agent_pos[1] // cfg.cell_size),
        )
        if cell not in self.visited_cells:
            self.visited_cells.add(cell)
            expl_rew = cfg.exploration_weight
        else:
            expl_rew = 0.0

        # 5. Terminal reward (scaled by time efficiency) or timeout penalty
        if goal_reached:
            # Efficiency scaling: fast completions receive maximum bonus (up to 100%),
            # while longer paths still receive a solid reward (at least 50% of goal_reward).
            # This completely eliminates constant step-penalty bleed while rewarding fast solutions!
            efficiency_ratio = 1.0 - (float(self.step_count) / float(self.max_steps)) * 0.5
            term_rew = cfg.goal_reward * max(0.5, efficiency_ratio)
        elif timed_out:
            term_rew = cfg.timeout_penalty
        else:
            term_rew = 0.0

        total_reward = dist_progress + time_pen + coll_pen + expl_rew + term_rew

        breakdown = {
            "progress": dist_progress,
            "time": time_pen,
            "collision": coll_pen,
            "exploration": expl_rew,
            "terminal": term_rew,
        }
        return total_reward, breakdown

    # --- Renderer Helpers ---
    def _init_renderer(self):
        from rendering.pygame_renderer import PygameRenderer
        from rendering.camera import Camera

        screen_w = self.render_cfg.get("screen", {}).get("width", 1280)
        screen_h = self.render_cfg.get("screen", {}).get("height", 720)
        self.renderer = PygameRenderer(screen_w, screen_h)
        self.camera = Camera(screen_w, screen_h)
        self._update_camera_bounds()

    def _update_camera_bounds(self):
        if self.camera:
            self.camera.x = self.world.map_config.width / 2.0
            self.camera.y = self.world.map_config.height / 2.0
            self.camera.zoom = self.render_cfg.get("camera", {}).get("default_zoom", 0.8)

    def render(self):
        if self.renderer is None or self.camera is None:
            return

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        agent = self.world.agents.get("agent_0")
        metrics = {}
        if agent:
            self.camera.follow(agent.position, smoothing=0.1)
            metrics = {
                "Map": self.world.map_config.name,
                "Step": f"{self.step_count}/{self.env_cfg.max_steps}",
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
