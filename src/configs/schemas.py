"""Immutable, JAX-compatible configuration schemas for DECO-SWARM v2."""

from typing import NamedTuple


class AgentConfig(NamedTuple):
    """Physical body, panoramic raycast sensor, and laser-tagger parameters."""

    radius: float = 10.0
    linear_speed: float = 180.0
    angular_speed: float = 200.0
    ray_count: int = 90
    fov_degrees: float = 180.0
    max_distance: float = 500.0
    tag_range: float = 250.0
    tag_cooldown_sec: float = 0.8
    respawn_sec: float = 2.5


class PhysicsConfig(NamedTuple):
    """Continuous 2D physics integration and collision relaxation settings."""

    tick_rate: float = 60.0
    collision_iterations: int = 2

    @property
    def dt(self) -> float:
        """Physics substep duration in seconds."""
        return 1.0 / self.tick_rate


class EpisodeConfig(NamedTuple):
    """RL macro-step cadence and DeepMind-style CTF match rules."""

    max_steps: int = 1800
    action_repeat: int = 6
    score_limit: int = 3
    flag_auto_return_sec: float = 15.0
    team_size: int = 2


class RewardConfig(NamedTuple):
    """Zero-sum team objective rewards and lightweight individual shaping."""

    capture_reward: float = 10.0
    win_reward: float = 20.0
    pickup_reward: float = 1.0
    return_reward: float = 1.0
    tag_reward: float = 0.5
    carrier_tag_bonus: float = 1.0
    exploration_weight: float = 0.05
    cell_size: float = 80.0
    collision_penalty: float = -0.02


class RenderConfig(NamedTuple):
    """Decoupled 2D viewer window and camera parameters."""

    screen_width: int = 1280
    screen_height: int = 760
    fps: int = 60
    title: str = "DECO-SWARM v2 | 2D Capture the Flag"
    default_zoom: float = 0.75
    min_zoom: float = 0.2
    max_zoom: float = 4.0


class ProjectConfig(NamedTuple):
    """Top-level immutable bundle of all domain configurations."""

    agent: AgentConfig = AgentConfig()
    physics: PhysicsConfig = PhysicsConfig()
    episode: EpisodeConfig = EpisodeConfig()
    rewards: RewardConfig = RewardConfig()
    render: RenderConfig = RenderConfig()
