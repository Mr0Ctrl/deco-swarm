from dataclasses import dataclass

# agent.yaml
@dataclass(frozen=True)
class BodyConfig:
    radius_m: float
    linear_speed_mps: float
    angular_speed_dps: float

@dataclass(frozen=True)
class PerceptionConfig:
    ray_count: int
    fov_deg: float
    max_range_m: float

@dataclass(frozen=True)
class TaggerConfig:
    range_m: float
    cooldown_s: float
    hits_to_tag: int
    carrier_can_fire: bool

@dataclass(frozen=True)
class AgentConfig:
    body: BodyConfig
    perception: PerceptionConfig
    tagger: TaggerConfig

# ---- environment.yaml ----
@dataclass(frozen=True)
class EpisodeConfig:
    max_steps: int
    action_repeat: int
    team_size: int
    score_limit: int | None          # None = time limit only (FTW)

@dataclass(frozen=True)
class RulesConfig:
    requires_own_flag_at_home: bool
    flag_auto_return_s: float
    respawn_s: float

@dataclass(frozen=True)
class RewardsConfig:
    capture_reward: float
    win_reward: float
    pickup_reward: float
    return_reward: float
    tag_reward: float
    carrier_tag_bonus: float
    exploration_weight: float
    cell_size_m: float
    collision_penalty: float

@dataclass(frozen=True)
class EnvironmentConfig:
    episode: EpisodeConfig
    rules: RulesConfig
    rewards: RewardsConfig

# simulation.yaml
@dataclass(frozen=True)
class PhysicsConfig:
    tick_hz: float
    collision_iterations: int
    max_segments: int
    max_team_size: int

@dataclass(frozen=True)
class SimulationConfig:
    physics: PhysicsConfig

# render.yaml
@dataclass(frozen=True)
class ScreenConfig:
    width_px: int
    height_px: int
    fps: int
    title: str

@dataclass(frozen=True)
class CameraConfig:
    pixels_per_meter: float
    default_zoom: float
    min_zoom: float
    max_zoom: float

@dataclass(frozen=True)
class RenderConfig:
    screen: ScreenConfig
    camera: CameraConfig

# all together: field name = file name
@dataclass(frozen=True)
class Configs:
    agent: AgentConfig
    environment: EnvironmentConfig
    simulation: SimulationConfig
    render: RenderConfig