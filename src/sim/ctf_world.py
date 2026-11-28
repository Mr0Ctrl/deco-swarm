"""Pure-JAX 2D Capture the Flag (DeepMind FTW rules) simulation loop."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from src.configs import AgentConfig, ArenaData, EpisodeConfig, PhysicsConfig

from .collision import resolve_agent_agent, resolve_agent_walls
from .geometry import decode_actions, step_kinematics
from .raycast import _ray_circles_dist, _ray_segments_dist, cast_all_rays, compute_ego_scalars


class CTFState(NamedTuple):
    """Immutable PyTree holding the full state of one 2D CTF arena."""

    pos: jax.Array                  # (N, 2)
    heading_deg: jax.Array          # (N,)
    teams: jax.Array                # (N,) int32: 0=Red, 1=Blue
    alive: jax.Array                # (N,) bool
    respawn_timer: jax.Array        # (N,) float32
    tag_cooldown: jax.Array         # (N,) float32
    carrying_flag: jax.Array        # (N,) bool
    prev_action_vec: jax.Array      # (N, 4) float32
    collided: jax.Array             # (N,) bool
    red_flag_pos: jax.Array         # (2,)
    blue_flag_pos: jax.Array        # (2,)
    red_flag_home: jax.Array        # () bool
    blue_flag_home: jax.Array       # () bool
    red_flag_carrier: jax.Array     # () int32 (-1 if uncarried)
    blue_flag_carrier: jax.Array    # () int32 (-1 if uncarried)
    red_flag_drop_timer: jax.Array  # () float32
    blue_flag_drop_timer: jax.Array # () float32
    scores: jax.Array               # (2,) int32 [red, blue]
    step_count: jax.Array           # () int32
    last_beam_end: jax.Array        # (N, 2) float32 (for viewer)
    last_beam_fired: jax.Array      # (N,) bool (for viewer)


class StepEvents(NamedTuple):
    """Per-macro-step CTF events accumulated across physics substeps."""

    tagged_count: jax.Array         # (N,) int32 tags scored by each agent
    tagged_carrier: jax.Array       # (N,) bool tagged an enemy flag carrier
    picked_up: jax.Array            # (N,) bool picked up enemy flag
    returned: jax.Array             # (N,) bool returned own dropped flag
    captured: jax.Array             # (2,) int32 [red, blue] captures this step


def reset_world(arena: ArenaData, team_size: int) -> CTFState:
    """Create initial CTFState for a given ArenaData and team_size (2, 3, or 5)."""
    red_spawns = arena.red_spawns[:team_size]
    blue_spawns = arena.blue_spawns[:team_size]
    spawns = jnp.concatenate([red_spawns, blue_spawns], axis=0)  # (2*team_size, 3)

    n = 2 * team_size
    teams = jnp.concatenate([
        jnp.zeros((team_size,), dtype=jnp.int32),
        jnp.ones((team_size,), dtype=jnp.int32),
    ])

    return CTFState(
        pos=spawns[:, :2],
        heading_deg=spawns[:, 2],
        teams=teams,
        alive=jnp.ones((n,), dtype=bool),
        respawn_timer=jnp.zeros((n,), dtype=jnp.float32),
        tag_cooldown=jnp.zeros((n,), dtype=jnp.float32),
        carrying_flag=jnp.zeros((n,), dtype=bool),
        prev_action_vec=jnp.zeros((n, 4), dtype=jnp.float32),
        collided=jnp.zeros((n,), dtype=bool),
        red_flag_pos=arena.red_flag_pos,
        blue_flag_pos=arena.blue_flag_pos,
        red_flag_home=jnp.asarray(True),
        blue_flag_home=jnp.asarray(True),
        red_flag_carrier=jnp.asarray(-1, dtype=jnp.int32),
        blue_flag_carrier=jnp.asarray(-1, dtype=jnp.int32),
        red_flag_drop_timer=jnp.asarray(0.0, dtype=jnp.float32),
        blue_flag_drop_timer=jnp.asarray(0.0, dtype=jnp.float32),
        scores=jnp.zeros((2,), dtype=jnp.int32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        last_beam_end=spawns[:, :2],
        last_beam_fired=jnp.zeros((n,), dtype=bool),
    )


def _fire_laser_tags(
    pos: jax.Array,
    heading_deg: jax.Array,
    teams: jax.Array,
    alive: jax.Array,
    tag_cooldown: jax.Array,
    fire_tag: jax.Array,
    segments: jax.Array,
    segment_mask: jax.Array,
    agent_radius: float,
    tag_range: float,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Resolve 1-hit forward laser beam tagger for all agents.

    Returns:
        hit_matrix: (N, N) bool where [i, j] is True if shooter i tagged target j.
        can_fire: (N,) bool indicating which agents fired this substep.
        beam_end: (N, 2) float32 endpoint of each beam for visualization.
    """
    n = pos.shape[0]
    can_fire = fire_tag & alive & (tag_cooldown <= 0.0)
    radii = jnp.full((n,), agent_radius, dtype=jnp.float32)

    def single_shot(i: jax.Array) -> tuple[jax.Array, jax.Array]:
        rad = jnp.deg2rad(heading_deg[i])
        ray_dir = jnp.asarray([[jnp.cos(rad), jnp.sin(rad)]], dtype=jnp.float32)  # (1, 2)
        wall_d = _ray_segments_dist(pos[i], ray_dir, tag_range, segments, segment_mask)[0]

        opp_mask = alive & (teams != teams[i])
        opp_dists = _ray_circles_dist(pos[i], ray_dir, tag_range, pos, radii, opp_mask)[0]  # (N,)
        best_j = jnp.argmin(opp_dists)
        best_d = opp_dists[best_j]

        hit_opp = can_fire[i] & (best_d <= tag_range) & (best_d < wall_d)
        hit_row = jnp.zeros((n,), dtype=bool).at[best_j].set(hit_opp)
        end_d = jnp.where(can_fire[i], jnp.minimum(wall_d, jnp.minimum(best_d, tag_range)), 0.0)
        end_xy = pos[i] + ray_dir[0] * end_d
        return hit_row, end_xy

    hit_matrix, beam_end = jax.vmap(single_shot)(jnp.arange(n, dtype=jnp.int32))
    was_tagged = jnp.any(hit_matrix, axis=0)
    return hit_matrix, was_tagged, can_fire, beam_end


def _pick_closest_eligible(
    pos: jax.Array,
    flag_pos: jax.Array,
    eligible_mask: jax.Array,
    pickup_dist: float,
) -> tuple[jax.Array, jax.Array]:
    """Return (found: bool, closest_idx: int32) among eligible agents within pickup_dist."""
    dists = jnp.linalg.norm(pos - flag_pos[None, :], axis=-1)
    in_range = eligible_mask & (dists <= pickup_dist)
    masked_dists = jnp.where(in_range, dists, jnp.inf)
    best_idx = jnp.argmin(masked_dists)
    found = jnp.any(in_range)
    return found, best_idx


def _substep(
    state: CTFState,
    events: StepEvents,
    arena: ArenaData,
    spawn_poses: jax.Array,
    move_fwd: jax.Array,
    strafe: jax.Array,
    rot_dir: jax.Array,
    fire_tag: jax.Array,
    agent_cfg: AgentConfig,
    phys_cfg: PhysicsConfig,
    ep_cfg: EpisodeConfig,
) -> tuple[CTFState, StepEvents]:
    """Advance one 60 Hz physics + CTF rule substep."""
    dt = phys_cfg.dt
    n = state.pos.shape[0]
    pickup_dist = agent_cfg.radius + arena.flag_radius

    # 1. Cooldowns & Respawns
    tag_cooldown = jnp.maximum(0.0, state.tag_cooldown - dt)
    respawn_timer = jnp.where(state.alive, 0.0, jnp.maximum(0.0, state.respawn_timer - dt))
    just_respawned = (~state.alive) & (respawn_timer <= 0.0)

    pos = jnp.where(just_respawned[:, None], spawn_poses[:, :2], state.pos)
    heading_deg = jnp.where(just_respawned, spawn_poses[:, 2], state.heading_deg)
    alive = state.alive | just_respawned

    # 2. Kinematics & Sliding Collisions
    pos, heading_deg = step_kinematics(
        pos, heading_deg, move_fwd, strafe, rot_dir, alive,
        agent_cfg.linear_speed, agent_cfg.angular_speed, dt,
    )
    pos, wall_hit = resolve_agent_walls(
        pos, alive, agent_cfg.radius, arena.segments, arena.segment_mask, phys_cfg.collision_iterations
    )
    pos = resolve_agent_agent(pos, alive, agent_cfg.radius)
    collided = state.collided | wall_hit

    # 3. 1-Hit Laser Tagger
    hit_matrix, was_tagged, fired_now, beam_end = _fire_laser_tags(
        pos, heading_deg, state.teams, alive, tag_cooldown, fire_tag,
        arena.segments, arena.segment_mask, agent_cfg.radius, agent_cfg.tag_range,
    )
    tag_cooldown = jnp.where(fired_now, agent_cfg.tag_cooldown_sec, tag_cooldown)

    # Record shooter events before clearing carrier state
    shooter_tags = jnp.sum(hit_matrix.astype(jnp.int32), axis=1)
    shooter_hit_carrier = jnp.any(hit_matrix & state.carrying_flag[None, :], axis=1)

    # Drop any flag held by a newly tagged agent
    red_flag_carrier = state.red_flag_carrier
    blue_flag_carrier = state.blue_flag_carrier
    red_flag_pos = jnp.where(red_flag_carrier >= 0, pos[jnp.maximum(red_flag_carrier, 0)], state.red_flag_pos)
    blue_flag_pos = jnp.where(blue_flag_carrier >= 0, pos[jnp.maximum(blue_flag_carrier, 0)], state.blue_flag_pos)
    red_flag_home = state.red_flag_home
    blue_flag_home = state.blue_flag_home
    red_flag_drop_timer = state.red_flag_drop_timer
    blue_flag_drop_timer = state.blue_flag_drop_timer

    red_carrier_tagged = (red_flag_carrier >= 0) & was_tagged[jnp.maximum(red_flag_carrier, 0)]
    blue_carrier_tagged = (blue_flag_carrier >= 0) & was_tagged[jnp.maximum(blue_flag_carrier, 0)]
    red_flag_carrier = jnp.where(red_carrier_tagged, jnp.asarray(-1, dtype=jnp.int32), red_flag_carrier)
    blue_flag_carrier = jnp.where(blue_carrier_tagged, jnp.asarray(-1, dtype=jnp.int32), blue_flag_carrier)
    red_flag_drop_timer = jnp.where(red_carrier_tagged, 0.0, red_flag_drop_timer)
    blue_flag_drop_timer = jnp.where(blue_carrier_tagged, 0.0, blue_flag_drop_timer)

    carrying_flag = state.carrying_flag & (~was_tagged)
    alive = alive & (~was_tagged)
    respawn_timer = jnp.where(was_tagged, agent_cfg.respawn_sec, respawn_timer)

    # 4. Enemy Flag Pickup (Blue picks up Red flag; Red picks up Blue flag)
    blue_eligible = alive & (state.teams == 1)
    red_eligible = alive & (state.teams == 0)

    red_picked, red_picker = _pick_closest_eligible(
        pos, red_flag_pos, blue_eligible & (red_flag_carrier < 0), pickup_dist
    )
    red_flag_carrier = jnp.where(red_picked, red_picker, red_flag_carrier)
    red_flag_home = jnp.where(red_picked, jnp.asarray(False), red_flag_home)
    red_flag_drop_timer = jnp.where(red_picked, 0.0, red_flag_drop_timer)
    carrying_flag = carrying_flag.at[red_picker].set(carrying_flag[red_picker] | red_picked)

    blue_picked, blue_picker = _pick_closest_eligible(
        pos, blue_flag_pos, red_eligible & (blue_flag_carrier < 0), pickup_dist
    )
    blue_flag_carrier = jnp.where(blue_picked, blue_picker, blue_flag_carrier)
    blue_flag_home = jnp.where(blue_picked, jnp.asarray(False), blue_flag_home)
    blue_flag_drop_timer = jnp.where(blue_picked, 0.0, blue_flag_drop_timer)
    carrying_flag = carrying_flag.at[blue_picker].set(carrying_flag[blue_picker] | blue_picked)

    picked_up_step = (
        jnp.zeros((n,), dtype=bool)
        .at[red_picker].set(red_picked)
        .at[blue_picker].set(blue_picked)
    )

    # 5. Friendly Flag Touch-Return & 15s Dropped Auto-Return
    red_dropped = (~red_flag_home) & (red_flag_carrier < 0)
    red_touched, red_returner = _pick_closest_eligible(pos, red_flag_pos, red_eligible & red_dropped, pickup_dist)
    red_flag_drop_timer = jnp.where(red_dropped, red_flag_drop_timer + dt, 0.0)
    red_auto_return = red_dropped & (red_flag_drop_timer >= ep_cfg.flag_auto_return_sec)
    red_reset = red_touched | red_auto_return
    red_flag_pos = jnp.where(red_reset, arena.red_flag_pos, red_flag_pos)
    red_flag_home = red_flag_home | red_reset
    red_flag_drop_timer = jnp.where(red_reset, 0.0, red_flag_drop_timer)

    blue_dropped = (~blue_flag_home) & (blue_flag_carrier < 0)
    blue_touched, blue_returner = _pick_closest_eligible(pos, blue_flag_pos, blue_eligible & blue_dropped, pickup_dist)
    blue_flag_drop_timer = jnp.where(blue_dropped, blue_flag_drop_timer + dt, 0.0)
    blue_auto_return = blue_dropped & (blue_flag_drop_timer >= ep_cfg.flag_auto_return_sec)
    blue_reset = blue_touched | blue_auto_return
    blue_flag_pos = jnp.where(blue_reset, arena.blue_flag_pos, blue_flag_pos)
    blue_flag_home = blue_flag_home | blue_reset
    blue_flag_drop_timer = jnp.where(blue_reset, 0.0, blue_flag_drop_timer)

    returned_step = (
        jnp.zeros((n,), dtype=bool)
        .at[red_returner].set(red_touched)
        .at[blue_returner].set(blue_touched)
    )

    # 6. Scoring (DeepMind rule: own flag must be at home base)
    red_carrier_idx = jnp.maximum(blue_flag_carrier, 0)
    red_can_score = (blue_flag_carrier >= 0) & red_flag_home & (
        jnp.linalg.norm(pos[red_carrier_idx] - arena.red_base[:2]) <= arena.red_base[2] + agent_cfg.radius
    )
    blue_flag_pos = jnp.where(red_can_score, arena.blue_flag_pos, blue_flag_pos)
    blue_flag_home = blue_flag_home | red_can_score
    blue_flag_carrier = jnp.where(red_can_score, jnp.asarray(-1, dtype=jnp.int32), blue_flag_carrier)
    carrying_flag = carrying_flag.at[red_carrier_idx].set(carrying_flag[red_carrier_idx] & (~red_can_score))

    blue_carrier_idx = jnp.maximum(red_flag_carrier, 0)
    blue_can_score = (red_flag_carrier >= 0) & blue_flag_home & (
        jnp.linalg.norm(pos[blue_carrier_idx] - arena.blue_base[:2]) <= arena.blue_base[2] + agent_cfg.radius
    )
    red_flag_pos = jnp.where(blue_can_score, arena.red_flag_pos, red_flag_pos)
    red_flag_home = red_flag_home | blue_can_score
    red_flag_carrier = jnp.where(blue_can_score, jnp.asarray(-1, dtype=jnp.int32), red_flag_carrier)
    carrying_flag = carrying_flag.at[blue_carrier_idx].set(carrying_flag[blue_carrier_idx] & (~blue_can_score))

    scores = state.scores + jnp.asarray([red_can_score.astype(jnp.int32), blue_can_score.astype(jnp.int32)])

    # Keep beam visualization from whichever substep fired during the macro-step
    last_beam_fired = state.last_beam_fired | fired_now
    last_beam_end = jnp.where(fired_now[:, None], beam_end, state.last_beam_end)

    next_state = state._replace(
        pos=pos,
        heading_deg=heading_deg,
        alive=alive,
        respawn_timer=respawn_timer,
        tag_cooldown=tag_cooldown,
        carrying_flag=carrying_flag,
        collided=collided,
        red_flag_pos=red_flag_pos,
        blue_flag_pos=blue_flag_pos,
        red_flag_home=red_flag_home,
        blue_flag_home=blue_flag_home,
        red_flag_carrier=red_flag_carrier,
        blue_flag_carrier=blue_flag_carrier,
        red_flag_drop_timer=red_flag_drop_timer,
        blue_flag_drop_timer=blue_flag_drop_timer,
        scores=scores,
        last_beam_end=last_beam_end,
        last_beam_fired=last_beam_fired,
    )

    next_events = StepEvents(
        tagged_count=events.tagged_count + shooter_tags,
        tagged_carrier=events.tagged_carrier | shooter_hit_carrier,
        picked_up=events.picked_up | picked_up_step,
        returned=events.returned | returned_step,
        captured=events.captured + jnp.asarray([red_can_score.astype(jnp.int32), blue_can_score.astype(jnp.int32)]),
    )
    return next_state, next_events


def get_observations(
    state: CTFState,
    arena: ArenaData,
    agent_cfg: AgentConfig,
) -> tuple[jax.Array, jax.Array]:
    """Compute (rays: (N, 90, 12), ego: (N, 9)) at 10 Hz."""
    rays = cast_all_rays(
        state.pos,
        state.heading_deg,
        state.teams,
        state.alive,
        state.carrying_flag,
        arena.segments,
        arena.segment_mask,
        state.red_flag_pos,
        state.blue_flag_pos,
        arena.red_base,
        arena.blue_base,
        agent_cfg.radius,
        arena.flag_radius,
        agent_cfg.ray_count,
        agent_cfg.fov_degrees,
        agent_cfg.max_distance,
    )
    ego = compute_ego_scalars(
        state.pos,
        state.teams,
        state.prev_action_vec,
        state.carrying_flag,
        state.red_flag_home,
        state.blue_flag_home,
        arena.red_base,
        arena.blue_base,
        state.tag_cooldown,
        agent_cfg.radius,
        agent_cfg.tag_cooldown_sec,
    )
    return rays, ego


def step_world(
    state: CTFState,
    arena: ArenaData,
    actions: jax.Array,
    agent_cfg: AgentConfig,
    phys_cfg: PhysicsConfig,
    ep_cfg: EpisodeConfig,
) -> tuple[CTFState, tuple[jax.Array, jax.Array], StepEvents, jax.Array, jax.Array]:
    """Execute one 10 Hz RL macro-step (action_repeat 60 Hz substeps).

    Returns:
        (next_state, (rays, ego), events, terminated, truncated)
    """
    n = state.pos.shape[0]
    team_size = n // 2
    spawn_poses = jnp.concatenate([arena.red_spawns[:team_size], arena.blue_spawns[:team_size]], axis=0)

    move_fwd, strafe, rot_dir, fire_tag = decode_actions(actions)
    action_vec = jnp.stack([move_fwd, strafe, rot_dir, fire_tag.astype(jnp.float32)], axis=-1)

    init_state = state._replace(
        collided=jnp.zeros((n,), dtype=bool),
        last_beam_fired=jnp.zeros((n,), dtype=bool),
    )
    init_events = StepEvents(
        tagged_count=jnp.zeros((n,), dtype=jnp.int32),
        tagged_carrier=jnp.zeros((n,), dtype=bool),
        picked_up=jnp.zeros((n,), dtype=bool),
        returned=jnp.zeros((n,), dtype=bool),
        captured=jnp.zeros((2,), dtype=jnp.int32),
    )

    def scan_fn(carry: tuple[CTFState, StepEvents], _) -> tuple[tuple[CTFState, StepEvents], None]:
        st, ev = carry
        return _substep(
            st, ev, arena, spawn_poses, move_fwd, strafe, rot_dir, fire_tag,
            agent_cfg, phys_cfg, ep_cfg,
        ), None

    (next_state, events), _ = jax.lax.scan(
        scan_fn, (init_state, init_events), None, length=ep_cfg.action_repeat
    )
    next_state = next_state._replace(
        prev_action_vec=action_vec,
        step_count=state.step_count + 1,
    )

    obs = get_observations(next_state, arena, agent_cfg)
    terminated = jnp.any(next_state.scores >= ep_cfg.score_limit)
    truncated = (next_state.step_count >= ep_cfg.max_steps) & (~terminated)
    return next_state, obs, events, terminated, truncated
