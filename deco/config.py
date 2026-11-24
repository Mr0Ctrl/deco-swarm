# deco/config.py
"""
YAML configuration loading.

Reads the same files as the original PyTorch version:
  - configs/environment.yaml  (episode limits + reward coefficients)
  - configs/simulation.yaml   (physics tick rate, collision iterations)
  - configs/agent.yaml        (body kinematics + raycast perception)
  - configs/render.yaml       (viewer window and camera)

All config classes are frozen dataclasses so they can be closed over by jitted functions.
"""
import os
from dataclasses import dataclass, fields
from typing import Any, Dict

import yaml


def load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _from_dict(cls, data: Dict[str, Any]):
    """Builds a dataclass from a dict, keeping defaults for missing keys and casting to the field type."""
    kwargs = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = type(f.default)(data[f.name])
    return cls(**kwargs)


@dataclass(frozen=True)
class RewardConfig:
    """All reward and penalty coefficients in one place (same semantics as the original env)."""
    progress_weight: float = 0.01       # Distance progress gradient (+ when approaching goal)
    time_penalty: float = -1.1          # Urgency penalty deducted at each decision step
    collision_penalty: float = -0.5     # Penalty applied if the agent bumps into a wall
    exploration_weight: float = 0.1     # Novelty bonus for visiting a new 2D grid cell
    cell_size: float = 120.0            # Grid cell size in pixels for spatial visit tracking
    goal_reward: float = 100.0          # Terminal bonus awarded upon reaching the goal
    goal_time_decay: float = 0.5        # Max fraction of goal reward deducted for the slowest completion
    min_goal_ratio: float = 0.5         # Minimum guaranteed fraction of goal_reward
    timeout_penalty: float = -1.0       # Terminal penalty if episode runs out of steps


@dataclass(frozen=True)
class EnvConfig:
    """Simulation timing and episode horizon."""
    max_steps: int = 100                # Maximum decision steps per episode before truncation (10 Hz)
    action_repeat: int = 6              # Physics substeps per agent decision (60 Hz / 6 = 10 Hz)
    physics_fps: float = 60.0           # Physics tick rate
    collision_iterations: int = 2       # Wall collision relaxation passes per physics tick
    default_map: str = "configs/maps/faz_1v0.1.yaml"


@dataclass(frozen=True)
class AgentConfig:
    """Agent body kinematics and raycast perception."""
    radius: float = 16.0
    linear_speed: float = 180.0
    angular_speed: float = 200.0
    ray_count: int = 90
    fov_degrees: float = 180.0
    max_distance: float = 500.0

    @property
    def obs_dim(self) -> int:
        # rays * 7 channels + polar target (2) + velocity (3) + previous action (3) + health/ammo (2)
        return self.ray_count * 7 + 10


@dataclass(frozen=True)
class Configs:
    env: EnvConfig
    rewards: RewardConfig
    agent: AgentConfig
    render: Dict[str, Any]

    def __hash__(self):  # render dict is not hashable; it never enters jitted code
        return hash((self.env, self.rewards, self.agent))


def load_configs(config_dir: str = "configs") -> Configs:
    env_yaml = load_yaml(os.path.join(config_dir, "environment.yaml"))
    sim_yaml = load_yaml(os.path.join(config_dir, "simulation.yaml"))
    agent_yaml = load_yaml(os.path.join(config_dir, "agent.yaml"))
    render_yaml = load_yaml(os.path.join(config_dir, "render.yaml"))

    episode = dict(env_yaml.get("episode", {}))
    physics = sim_yaml.get("physics", {})
    # simulation.yaml names the tick rate "tick_rate"; the original env looked up "fps" (default 60)
    episode["physics_fps"] = physics.get("fps", physics.get("tick_rate", 60.0))
    if "collision_iterations" in physics:
        episode["collision_iterations"] = physics["collision_iterations"]

    agent = {**agent_yaml.get("body", {}), **agent_yaml.get("perception", {})}

    return Configs(
        env=_from_dict(EnvConfig, episode),
        rewards=_from_dict(RewardConfig, env_yaml.get("rewards", {})),
        agent=_from_dict(AgentConfig, agent),
        render=render_yaml,
    )
