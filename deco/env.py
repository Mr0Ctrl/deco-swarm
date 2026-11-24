# deco/env.py
"""
Pure-JAX single-agent navigation environment (port of legacy SingleAgentMazeEnv + World + Agent).

Behaviour is kept identical to the original engine:
  - Continuous kinematics: rotate first, then move along the new heading (diagonal speed normalized).
  - Wall collision: sequential circle-vs-segment push-out over all segments, `collision_iterations` passes.
  - Raycasting: `ray_count` rays spread over `fov_degrees`, parametric ray-segment intersection.
  - 640-float ego-centric observation: 90 rays x 7 channels + polar target + velocity + a_{t-1} + health/ammo.
  - Action repeat: N physics ticks per decision, stopping early when the goal is reached.
  - Rewards: progress, time penalty, collision, episodic cell exploration, time-scaled goal reward / timeout.

All functions handle ONE environment and are written to be `jax.jit` / `jax.vmap` friendly.
`step` auto-resets finished episodes (a new map is sampled from the given MapPool).
"""
import math
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from deco.config import Configs
from deco.maps import MapBank

# Same tables as the original ActionDecoder: move index -> (move_fwd, strafe), rotation index -> rotate_dir
MOVE_TABLE = np.array([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
ROTATE_TABLE = np.array([0.0, -1.0, 1.0])
NUM_MOVE_ACTIONS = len(MOVE_TABLE)
NUM_ROTATE_ACTIONS = len(ROTATE_TABLE)


class EnvState(NamedTuple):
    pos: jax.Array          # (2,) world position
    angle_deg: jax.Array    # () heading in [0, 360)
    last_action: jax.Array  # (3,) (move_fwd, strafe, rotate_dir); also used as the "current velocity" obs
    map_idx: jax.Array      # () index into the MapBank
    step_count: jax.Array   # () decision steps taken in this episode
    prev_dist: jax.Array    # () distance to goal after the previous decision (progress reward)
    visited: jax.Array      # (GX, GY) bool, episodic exploration grid
    ray_dist: jax.Array     # (K,) normalized hit distance in [0, 1]
    ray_hit: jax.Array      # (K,) bool, ray hit a wall within range
    polar: jax.Array        # (2,) (norm_dist, norm_angle) ego-polar vector to the goal


class MapPool(NamedTuple):
    """Maps an episode may be sampled from: first `count` entries of `indices` (fixed size => no recompiles)."""
    indices: jax.Array  # (M,) int32
    count: jax.Array    # () int32


# ==============================================================================
# GEOMETRY (jnp ports of engine/core/geometry.py and collision.py)
# ==============================================================================

def wrap_angle_deg(angle_deg):
    return jnp.mod(angle_deg, 360.0)


def shortest_angle_diff_deg(target_deg, source_deg):
    return jnp.mod(target_deg - source_deg + 180.0, 360.0) - 180.0


def ego_polar_target(pos, angle_deg, target_pos, max_map_distance):
    delta = target_pos - pos
    dist = jnp.sqrt(jnp.sum(delta * delta))
    norm_dist = jnp.minimum(1.0, dist / max_map_distance)
    target_angle_deg = jnp.degrees(jnp.arctan2(delta[1], delta[0]))
    norm_angle = shortest_angle_diff_deg(target_angle_deg, angle_deg) / 180.0
    return jnp.stack([norm_dist, norm_angle])


def ray_angles_deg(heading_deg, fov_deg: float, num_rays: int):
    if num_rays == 1:
        return heading_deg[None]
    step = fov_deg / (num_rays - 1)
    return heading_deg - fov_deg / 2.0 + jnp.arange(num_rays) * step


def cast_rays(origin, heading_deg, segments, fov_deg: float, num_rays: int, max_dist: float):
    """Returns (norm_dist (K,), has_hit (K,)) for the closest wall along each ray."""
    rad = jnp.radians(ray_angles_deg(heading_deg, fov_deg, num_rays))
    rx = (jnp.cos(rad) * max_dist)[:, None]  # (K, 1)
    ry = (jnp.sin(rad) * max_dist)[:, None]

    x1, y1, x2, y2 = segments[:, 0], segments[:, 1], segments[:, 2], segments[:, 3]  # (S,)
    sx, sy = x2 - x1, y2 - y1
    qpx, qpy = x1 - origin[0], y1 - origin[1]

    det = rx * sy - ry * sx  # (K, S)
    not_parallel = jnp.abs(det) >= 1e-8
    safe_det = jnp.where(not_parallel, det, 1.0)
    t = (qpx * sy - qpy * sx) / safe_det
    u = (qpx * ry - qpy * rx) / safe_det
    valid = not_parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)

    # A hit only counts if it is strictly closer than the current best (initialized at max range = 1.0)
    t_min = jnp.min(jnp.where(valid, t, 1.0), axis=1)
    return t_min, t_min < 1.0


def resolve_wall_collisions(pos, segments, radius: float, iterations: int):
    """Sequential circle-segment push-out (same order and rules as the original C++/Python solver)."""

    def push_out(carry, seg):
        p, collided = carry
        a = seg[:2]
        d = seg[2:] - a
        len_sq = jnp.sum(d * d)
        non_degenerate = len_sq >= 1e-6
        t = jnp.clip(jnp.sum((p - a) * d) / jnp.where(non_degenerate, len_sq, 1.0), 0.0, 1.0)
        diff = p - (a + t * d)
        dist_sq = jnp.sum(diff * diff)
        penetrating = non_degenerate & (dist_sq < radius * radius)

        dist = jnp.sqrt(dist_sq)
        normal = jnp.where(dist > 1e-6, diff / jnp.maximum(dist, 1e-6), jnp.array([1.0, 0.0]))
        p = jnp.where(penetrating, p + normal * (radius - dist), p)
        return (p, collided | penetrating), None

    collided = jnp.zeros((), dtype=bool)
    for _ in range(iterations):
        # An extra pass after a pass with no penetration changes nothing, so the C++ early exit is not needed.
        (pos, collided), _ = jax.lax.scan(push_out, (pos, collided), segments)
    return pos, collided


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

class NavEnv:
    """
    Single-agent goal navigation on a bank of maps.

      obs, state = env.reset(key, pool)
      obs, state, reward, done, info = env.step(key, state, action, pool)   # action: int[2] (move, rotate)
    """

    def __init__(self, configs: Configs, bank: MapBank):
        self.cfg = configs.env
        self.rew = configs.rewards
        self.agent = configs.agent
        self.bank = bank

        self.segments = jnp.asarray(bank.segments)
        self.spawn = jnp.asarray(bank.spawn)
        self.goal_pos = jnp.asarray(bank.goal_pos)
        self.goal_radius = jnp.asarray(bank.goal_radius)
        self.max_map_distance = jnp.asarray(bank.max_map_distance)
        self.move_table = jnp.asarray(MOVE_TABLE)
        self.rotate_table = jnp.asarray(ROTATE_TABLE)

        self.physics_dt = 1.0 / self.cfg.physics_fps
        self.grid_shape = (
            int(math.ceil(bank.max_width / self.rew.cell_size)) + 1,
            int(math.ceil(bank.max_height / self.rew.cell_size)) + 1,
        )
        self.obs_dim = self.agent.obs_dim

    # --- Pieces shared by training (step) and the interactive viewer (world_tick) ---
    def physics_tick(self, pos, angle_deg, action3, map_idx, dt):
        """One physics tick: rotate, move along the new heading, then resolve wall collisions."""
        move_fwd, strafe, rotate_dir = action3[0], action3[1], action3[2]
        angle_deg = wrap_angle_deg(angle_deg + rotate_dir * self.agent.angular_speed * dt)

        # Diagonal speed fix: normalize the combined command if its magnitude exceeds 1
        scale = 1.0 / jnp.maximum(jnp.hypot(move_fwd, strafe), 1.0)
        move_fwd, strafe = move_fwd * scale, strafe * scale

        rad = jnp.radians(angle_deg)
        forward = jnp.stack([jnp.cos(rad), jnp.sin(rad)])
        right = jnp.stack([-forward[1], forward[0]])
        velocity = (forward * move_fwd + right * strafe) * self.agent.linear_speed
        pos = pos + velocity * dt

        pos, collided = resolve_wall_collisions(
            pos, self.segments[map_idx], self.agent.radius, self.cfg.collision_iterations
        )
        return pos, angle_deg, collided

    def perceive(self, state: EnvState) -> EnvState:
        """Updates rays and the polar goal vector for the current pose."""
        ray_dist, ray_hit = cast_rays(
            state.pos,
            state.angle_deg,
            self.segments[state.map_idx],
            self.agent.fov_degrees,
            self.agent.ray_count,
            self.agent.max_distance,
        )
        polar = ego_polar_target(
            state.pos, state.angle_deg, self.goal_pos[state.map_idx], self.max_map_distance[state.map_idx]
        )
        return state._replace(ray_dist=ray_dist, ray_hit=ray_hit, polar=polar)

    def observation(self, state: EnvState) -> jax.Array:
        k = self.agent.ray_count
        zeros = jnp.zeros(k)
        # Per ray: [distance, is_wall (only if hit), is_friend, is_enemy, is_objective, rel_cos, rel_sin]
        rays = jnp.stack(
            [state.ray_dist, state.ray_hit.astype(zeros.dtype), zeros, zeros, zeros, zeros, zeros], axis=1
        )
        status = jnp.ones(2, dtype=zeros.dtype)  # normalized health (always full) and ammo
        obs = jnp.concatenate([rays.reshape(-1), state.polar, state.last_action, state.last_action, status])
        return obs.astype(jnp.float32)

    def cell_of(self, pos):
        cell = jnp.floor(pos / self.rew.cell_size).astype(jnp.int32)
        return (
            jnp.clip(cell[0], 0, self.grid_shape[0] - 1),
            jnp.clip(cell[1], 0, self.grid_shape[1] - 1),
        )

    def goal_distance(self, state: EnvState):
        delta = state.pos - self.goal_pos[state.map_idx]
        return jnp.sqrt(jnp.sum(delta * delta))

    def is_goal_reached(self, state: EnvState):
        return self.goal_distance(state) <= self.agent.radius + self.goal_radius[state.map_idx]

    # --- Episode API ---
    def reset_to_map(self, map_idx) -> Tuple[jax.Array, EnvState]:
        map_idx = jnp.asarray(map_idx, dtype=jnp.int32)
        spawn = self.spawn[map_idx]
        pos = spawn[:2]
        state = EnvState(
            pos=pos,
            angle_deg=wrap_angle_deg(spawn[2]),
            last_action=jnp.zeros(3, dtype=pos.dtype),
            map_idx=map_idx,
            step_count=jnp.zeros((), dtype=jnp.int32),
            prev_dist=jnp.zeros((), dtype=pos.dtype),
            visited=jnp.zeros(self.grid_shape, dtype=bool),
            ray_dist=jnp.ones(self.agent.ray_count, dtype=pos.dtype),
            ray_hit=jnp.zeros(self.agent.ray_count, dtype=bool),
            polar=jnp.zeros(2, dtype=pos.dtype),
        )
        state = state._replace(prev_dist=self.goal_distance(state), visited=state.visited.at[self.cell_of(pos)].set(True))
        state = self.perceive(state)
        return self.observation(state), state

    def reset(self, key, pool: MapPool) -> Tuple[jax.Array, EnvState]:
        choice = jax.random.randint(key, (), 0, pool.count)
        return self.reset_to_map(pool.indices[choice])

    def step_no_reset(self, state: EnvState, action):
        """One decision step (action repeat) without auto-reset. Returns obs, state, reward, terminated, truncated, info."""
        cfg, rew = self.cfg, self.rew
        step_count = state.step_count + 1
        move = self.move_table[action[0]]
        action3 = jnp.concatenate([move, self.rotate_table[action[1]][None]])

        goal = self.goal_pos[state.map_idx]
        reach = self.agent.radius + self.goal_radius[state.map_idx]

        def tick(carry, _):
            pos, angle, collided, reached = carry
            new_pos, new_angle, hit = self.physics_tick(pos, angle, action3, state.map_idx, self.physics_dt)
            active = ~reached  # the original loop breaks as soon as the goal is reached
            delta = new_pos - goal
            pos = jnp.where(active, new_pos, pos)
            angle = jnp.where(active, new_angle, angle)
            collided = collided | (active & hit)
            reached = reached | (active & (jnp.sqrt(jnp.sum(delta * delta)) <= reach))
            return (pos, angle, collided, reached), None

        false = jnp.zeros((), dtype=bool)
        (pos, angle, collided, reached), _ = jax.lax.scan(
            tick, (state.pos, state.angle_deg, false, false), None, length=cfg.action_repeat
        )

        state = self.perceive(state._replace(pos=pos, angle_deg=angle, last_action=action3, step_count=step_count))
        curr_dist = self.goal_distance(state)

        terminated = reached
        truncated = (step_count >= cfg.max_steps) & ~terminated

        # Rewards (same formula as SingleAgentMazeEnv._compute_step_reward)
        progress = (state.prev_dist - curr_dist) * rew.progress_weight
        time_pen = jnp.asarray(rew.time_penalty, dtype=curr_dist.dtype)
        coll_pen = jnp.where(collided, rew.collision_penalty, 0.0)

        cell = self.cell_of(pos)
        explore = jnp.where(state.visited[cell], 0.0, rew.exploration_weight)
        visited = state.visited.at[cell].set(True)

        efficiency = 1.0 - (step_count / cfg.max_steps) * rew.goal_time_decay
        goal_rew = rew.goal_reward * jnp.maximum(rew.min_goal_ratio, efficiency)
        terminal = jnp.where(reached, goal_rew, jnp.where(truncated, rew.timeout_penalty, 0.0))

        reward = progress + time_pen + coll_pen + explore + terminal
        state = state._replace(prev_dist=curr_dist, visited=visited)

        info = {
            "is_success": reached,
            "step_count": step_count,
            "dist_to_goal": curr_dist,
            "collided": collided,
            "explored_cells": jnp.sum(visited),
            "map_idx": state.map_idx,
            "reward_progress": progress,
            "reward_time": time_pen,
            "reward_collision": coll_pen,
            "reward_exploration": explore,
            "reward_terminal": terminal,
        }
        return self.observation(state), state, reward, terminated, truncated, info

    def step(self, key, state: EnvState, action, pool: MapPool):
        """Decision step with auto-reset: when the episode ends, the returned obs/state belong to a new episode."""
        obs, next_state, reward, terminated, truncated, info = self.step_no_reset(state, action)
        done = terminated | truncated
        reset_obs, reset_state = self.reset(key, pool)
        next_state = jax.tree.map(lambda r, s: jnp.where(done, r, s), reset_state, next_state)
        obs = jnp.where(done, reset_obs, obs)
        info = {**info, "terminated": terminated, "truncated": truncated}
        return obs, next_state, reward, done, info

    # --- Interactive viewer helpers ---
    def world_tick(self, state: EnvState, action3, dt) -> EnvState:
        """One free-running physics tick with a continuous (move_fwd, strafe, rotate) command (play.py)."""
        pos, angle, _ = self.physics_tick(state.pos, state.angle_deg, action3, state.map_idx, dt)
        return self.perceive(state._replace(pos=pos, angle_deg=angle, last_action=jnp.asarray(action3, state.pos.dtype)))

    def ray_endpoints(self, state: EnvState) -> jax.Array:
        rad = jnp.radians(ray_angles_deg(state.angle_deg, self.agent.fov_degrees, self.agent.ray_count))
        length = state.ray_dist * self.agent.max_distance
        return state.pos + jnp.stack([jnp.cos(rad), jnp.sin(rad)], axis=1) * length[:, None]
