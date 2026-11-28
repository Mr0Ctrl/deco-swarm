"""Unit tests for configuration schemas and YAML loading."""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from src.configs import (
    AgentConfig,
    EpisodeConfig,
    PhysicsConfig,
    ProjectConfig,
    RenderConfig,
    RewardConfig,
    load_project_config,
)


def test_load_repo_configs():
    cfg = load_project_config("configs")
    assert isinstance(cfg, ProjectConfig)
    assert cfg.agent == AgentConfig()
    assert cfg.physics == PhysicsConfig()
    assert cfg.episode == EpisodeConfig()
    assert cfg.rewards == RewardConfig()
    assert cfg.render == RenderConfig()
    assert cfg.physics.dt == pytest.approx(1.0 / 60.0)


def test_missing_dir_falls_back_to_defaults(tmp_path: Path):
    cfg = load_project_config(tmp_path / "nonexistent")
    assert cfg == ProjectConfig()


def test_partial_yaml_override(tmp_path: Path):
    (tmp_path / "agent.yaml").write_text(
        "body:\n  radius: 14.0\ntag_unused: true\n", encoding="utf-8"
    )
    (tmp_path / "environment.yaml").write_text(
        "episode:\n  team_size: 3\nrewards:\n  capture_reward: 25.0\n",
        encoding="utf-8",
    )
    cfg = load_project_config(tmp_path)
    assert cfg.agent.radius == 14.0
    assert cfg.agent.linear_speed == 180.0
    assert cfg.episode.team_size == 3
    assert cfg.episode.max_steps == 1800
    assert cfg.rewards.capture_reward == 25.0
    assert cfg.rewards.win_reward == 20.0


def test_configs_are_immutable_and_jax_compatible():
    cfg = load_project_config("configs")
    with pytest.raises(AttributeError):
        cfg.agent.radius = 99.0  # type: ignore[misc]

    @jax.jit
    def step_scale(x: jax.Array, phys: PhysicsConfig) -> jax.Array:
        return x * phys.dt

    out = step_scale(jnp.array(60.0), cfg.physics)
    assert float(out) == pytest.approx(1.0)
