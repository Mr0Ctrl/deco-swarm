"""YAML configuration loader with safe dataclass fallback defaults."""

from pathlib import Path
from typing import Any, Mapping, NamedTuple

import yaml

from .schemas import (
    AgentConfig,
    EpisodeConfig,
    PhysicsConfig,
    ProjectConfig,
    RenderConfig,
    RewardConfig,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping if the file exists, otherwise return an empty dict."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(data).__name__}")
    return dict(data)


def _pick(section: Mapping[str, Any], defaults: NamedTuple, key_map: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Extract known fields from a YAML mapping, falling back to NamedTuple defaults."""
    default_dict = defaults._asdict()
    mapping = key_map or {k: k for k in default_dict}
    resolved: dict[str, Any] = {}
    for target_field, yaml_key in mapping.items():
        if yaml_key in section:
            resolved[target_field] = type(default_dict[target_field])(section[yaml_key])
        else:
            resolved[target_field] = default_dict[target_field]
    return resolved


def load_agent_config(path: Path) -> AgentConfig:
    """Load AgentConfig from agent.yaml."""
    raw = _read_yaml(path)
    defaults = AgentConfig()
    body = _pick(raw.get("body", {}), defaults, {
        "radius": "radius",
        "linear_speed": "linear_speed",
        "angular_speed": "angular_speed",
    })
    perc = _pick(raw.get("perception", {}), defaults, {
        "ray_count": "ray_count",
        "fov_degrees": "fov_degrees",
        "max_distance": "max_distance",
    })
    tagger = _pick(raw.get("tagger", {}), defaults, {
        "tag_range": "range",
        "tag_cooldown_sec": "cooldown_sec",
        "respawn_sec": "respawn_sec",
    })
    return AgentConfig(**{**body, **perc, **tagger})


def load_physics_config(path: Path) -> PhysicsConfig:
    """Load PhysicsConfig from simulation.yaml."""
    raw = _read_yaml(path)
    defaults = PhysicsConfig()
    return PhysicsConfig(**_pick(raw.get("physics", {}), defaults))


def load_environment_configs(path: Path) -> tuple[EpisodeConfig, RewardConfig]:
    """Load EpisodeConfig and RewardConfig from environment.yaml."""
    raw = _read_yaml(path)
    ep_cfg = EpisodeConfig(**_pick(raw.get("episode", {}), EpisodeConfig()))
    rew_cfg = RewardConfig(**_pick(raw.get("rewards", {}), RewardConfig()))
    return ep_cfg, rew_cfg


def load_render_config(path: Path) -> RenderConfig:
    """Load RenderConfig from render.yaml."""
    raw = _read_yaml(path)
    defaults = RenderConfig()
    screen = _pick(raw.get("screen", {}), defaults, {
        "screen_width": "width",
        "screen_height": "height",
        "fps": "fps",
        "title": "title",
    })
    cam = _pick(raw.get("camera", {}), defaults, {
        "default_zoom": "default_zoom",
        "min_zoom": "min_zoom",
        "max_zoom": "max_zoom",
    })
    return RenderConfig(**{**screen, **cam})


def load_project_config(config_dir: str | Path = "configs") -> ProjectConfig:
    """Load all domain configs from config_dir with safe defaults for missing files."""
    root = Path(config_dir)
    agent = load_agent_config(root / "agent.yaml")
    physics = load_physics_config(root / "simulation.yaml")
    episode, rewards = load_environment_configs(root / "environment.yaml")
    render = load_render_config(root / "render.yaml")
    return ProjectConfig(
        agent=agent,
        physics=physics,
        episode=episode,
        rewards=rewards,
        render=render,
    )
