"""Pure-JAX 2-layer team-relative semantic raycasting and ego observation."""

import jax
import jax.numpy as jnp


def _ray_segments_dist(
    origin: jax.Array,
    ray_dirs: jax.Array,
    max_distance: float,
    segments: jax.Array,
    segment_mask: jax.Array,
) -> jax.Array:
    """Compute nearest wall-segment hit distance per ray, or max_distance + 1.0 on miss."""
    no_hit = max_distance + 1.0
    r = ray_dirs * max_distance  # (R, 2)
    p1 = segments[:, 0]          # (M, 2)
    s = segments[:, 1] - p1      # (M, 2)

    det = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
    non_parallel = jnp.abs(det) >= 1e-8
    safe_det = jnp.where(non_parallel, det, 1.0)

    qp = p1[None, :, :] - origin[None, None, :]  # (1, M, 2)
    t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / safe_det
    u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / safe_det

    hit = segment_mask[None, :] & non_parallel & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    dists = jnp.where(hit, t * max_distance, no_hit)
    return jnp.min(dists, axis=1)


def _ray_circles_dist(
    origin: jax.Array,
    ray_dirs: jax.Array,
    max_distance: float,
    centers: jax.Array,
    radii: jax.Array,
    valid_mask: jax.Array,
) -> jax.Array:
    """Compute hit distance from each ray to each circle, or max_distance + 1.0 on miss.

    Returns shape (R, K). If origin is inside a circle, hit distance is 0.0.
    """
    no_hit = max_distance + 1.0
    oc = origin[None, :] - centers  # (K, 2)
    c = jnp.sum(oc * oc, axis=-1) - radii * radii  # (K,)
    inside = c <= 0.0

    # ray_dirs: (R, 2), oc: (K, 2) -> b: (R, K)
    b = jnp.einsum("rd,kd->rk", ray_dirs, oc)
    disc = b * b - c[None, :]
    has_root = disc >= 0.0
    sqrt_disc = jnp.sqrt(jnp.maximum(disc, 0.0))
    t = -b - sqrt_disc

    hit_forward = has_root & (t >= 0.0) & (t <= max_distance)
    dist = jnp.where(inside[None, :], 0.0, jnp.where(hit_forward, t, no_hit))
    return jnp.where(valid_mask[None, :], dist, no_hit)


def _cast_single_agent_rays(
    agent_idx: jax.Array,
    pos: jax.Array,
    heading_deg: jax.Array,
    teams: jax.Array,
    alive: jax.Array,
    carrying_flag: jax.Array,
    segments: jax.Array,
    segment_mask: jax.Array,
    red_flag_pos: jax.Array,
    blue_flag_pos: jax.Array,
    red_base: jax.Array,
    blue_base: jax.Array,
    agent_radius: float,
    flag_radius: float,
    ray_count: int,
    fov_degrees: float,
    max_distance: float,
) -> jax.Array:
    """Compute (ray_count, 12) 2-layer team-relative semantic rays for one agent."""
    no_hit = max_distance + 1.0
    origin = pos[agent_idx]
    my_team = teams[agent_idx]

    rel_deg = jnp.linspace(-0.5 * fov_degrees, 0.5 * fov_degrees, ray_count, dtype=jnp.float32)
    rel_rad = jnp.deg2rad(rel_deg)
    rel_cos = jnp.cos(rel_rad)
    rel_sin = jnp.sin(rel_rad)

    world_rad = jnp.deg2rad(heading_deg[agent_idx]) + rel_rad
    ray_dirs = jnp.stack([jnp.cos(world_rad), jnp.sin(world_rad)], axis=-1)  # (R, 2)

    # Layer 1: Solid hits (walls, allies, enemies)
    wall_dist = _ray_segments_dist(origin, ray_dirs, max_distance, segments, segment_mask)

    n_agents = pos.shape[0]
    not_self = jnp.arange(n_agents) != agent_idx
    agent_radii = jnp.full((n_agents,), agent_radius, dtype=jnp.float32)
    agent_dists = _ray_circles_dist(origin, ray_dirs, max_distance, pos, agent_radii, alive & not_self)

    is_ally_mask = (teams == my_team) & alive & not_self
    is_enemy_mask = (teams != my_team) & alive & not_self

    ally_dist = jnp.min(jnp.where(is_ally_mask[None, :], agent_dists, no_hit), axis=1)
    enemy_dist = jnp.min(jnp.where(is_enemy_mask[None, :], agent_dists, no_hit), axis=1)

    solid_dist_px = jnp.minimum(wall_dist, jnp.minimum(ally_dist, enemy_dist))
    any_solid_hit = solid_dist_px <= max_distance

    is_wall = (any_solid_hit & (wall_dist <= ally_dist) & (wall_dist <= enemy_dist)).astype(jnp.float32)
    is_ally = (any_solid_hit & (ally_dist < wall_dist) & (ally_dist <= enemy_dist)).astype(jnp.float32)
    is_enemy = (any_solid_hit & (enemy_dist < wall_dist) & (enemy_dist < ally_dist)).astype(jnp.float32)
    solid_dist = jnp.clip(solid_dist_px, 0.0, max_distance) / max_distance

    # Layer 2A: Flag hits (unoccluded by walls/agents; hidden if carried by self)
    own_flag_xy = jnp.where(my_team == 0, red_flag_pos, blue_flag_pos)
    enemy_flag_xy = jnp.where(my_team == 0, blue_flag_pos, red_flag_pos)
    flag_centers = jnp.stack([own_flag_xy, enemy_flag_xy], axis=0)
    flag_radii = jnp.asarray([flag_radius, flag_radius], dtype=jnp.float32)
    flag_valid = jnp.asarray([True, ~carrying_flag[agent_idx]], dtype=bool)

    flag_dists = _ray_circles_dist(origin, ray_dirs, max_distance, flag_centers, flag_radii, flag_valid)
    # Small tolerance (+1e-3) ensures carried flags aligned with carrier center remain visible
    own_flag_d = jnp.where(flag_dists[:, 0] <= solid_dist_px + 1e-3, flag_dists[:, 0], no_hit)
    enemy_flag_d = jnp.where(flag_dists[:, 1] <= solid_dist_px + 1e-3, flag_dists[:, 1], no_hit)

    nearest_flag_px = jnp.minimum(own_flag_d, enemy_flag_d)
    any_flag_hit = nearest_flag_px <= max_distance
    is_own_flag = (any_flag_hit & (own_flag_d <= enemy_flag_d)).astype(jnp.float32)
    is_enemy_flag = (any_flag_hit & (enemy_flag_d < own_flag_d)).astype(jnp.float32)
    flag_dist = jnp.clip(nearest_flag_px, 0.0, max_distance) / max_distance

    # Layer 2B: Base hits (unoccluded by walls/agents; 0.0 when standing inside)
    own_base = jnp.where(my_team == 0, red_base, blue_base)
    enemy_base = jnp.where(my_team == 0, blue_base, red_base)
    base_centers = jnp.stack([own_base[:2], enemy_base[:2]], axis=0)
    base_radii = jnp.stack([own_base[2], enemy_base[2]], axis=0)
    base_valid = jnp.asarray([True, True], dtype=bool)

    base_dists = _ray_circles_dist(origin, ray_dirs, max_distance, base_centers, base_radii, base_valid)
    own_base_d = jnp.where(base_dists[:, 0] <= solid_dist_px + 1e-3, base_dists[:, 0], no_hit)
    enemy_base_d = jnp.where(base_dists[:, 1] <= solid_dist_px + 1e-3, base_dists[:, 1], no_hit)

    nearest_base_px = jnp.minimum(own_base_d, enemy_base_d)
    any_base_hit = nearest_base_px <= max_distance
    is_own_base = (any_base_hit & (own_base_d <= enemy_base_d)).astype(jnp.float32)
    is_enemy_base = (any_base_hit & (enemy_base_d < own_base_d)).astype(jnp.float32)
    base_dist = jnp.clip(nearest_base_px, 0.0, max_distance) / max_distance

    return jnp.stack(
        [
            solid_dist,
            is_wall,
            is_ally,
            is_enemy,
            flag_dist,
            is_own_flag,
            is_enemy_flag,
            base_dist,
            is_own_base,
            is_enemy_base,
            rel_cos,
            rel_sin,
        ],
        axis=-1,
    )


def cast_all_rays(
    pos: jax.Array,
    heading_deg: jax.Array,
    teams: jax.Array,
    alive: jax.Array,
    carrying_flag: jax.Array,
    segments: jax.Array,
    segment_mask: jax.Array,
    red_flag_pos: jax.Array,
    blue_flag_pos: jax.Array,
    red_base: jax.Array,
    blue_base: jax.Array,
    agent_radius: float,
    flag_radius: float,
    ray_count: int = 90,
    fov_degrees: float = 180.0,
    max_distance: float = 500.0,
) -> jax.Array:
    """Compute (N, ray_count, 12) 2-layer team-relative semantic rays for all N agents."""
    n_agents = pos.shape[0]
    agent_indices = jnp.arange(n_agents, dtype=jnp.int32)
    return jax.vmap(
        _cast_single_agent_rays,
        in_axes=(
            0, None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, None, None,
        ),
    )(
        agent_indices,
        pos,
        heading_deg,
        teams,
        alive,
        carrying_flag,
        segments,
        segment_mask,
        red_flag_pos,
        blue_flag_pos,
        red_base,
        blue_base,
        agent_radius,
        flag_radius,
        ray_count,
        fov_degrees,
        max_distance,
    )


def compute_ego_scalars(
    pos: jax.Array,
    teams: jax.Array,
    prev_action_vec: jax.Array,
    carrying_flag: jax.Array,
    red_flag_home: jax.Array,
    blue_flag_home: jax.Array,
    red_base: jax.Array,
    blue_base: jax.Array,
    tag_cooldown: jax.Array,
    agent_radius: float,
    tag_cooldown_sec: float,
) -> jax.Array:
    """Compute (N, 9) ego/HUD scalars: prev_action(4) + 5 team-relative HUD flags."""
    own_flag_home = jnp.where(teams == 0, red_flag_home, blue_flag_home).astype(jnp.float32)
    enemy_flag_home = jnp.where(teams == 0, blue_flag_home, red_flag_home).astype(jnp.float32)

    own_base_xy = jnp.where((teams == 0)[:, None], red_base[:2][None, :], blue_base[:2][None, :])
    own_base_r = jnp.where(teams == 0, red_base[2], blue_base[2])
    in_own_base = (jnp.linalg.norm(pos - own_base_xy, axis=-1) <= own_base_r + agent_radius).astype(jnp.float32)

    denom = max(tag_cooldown_sec, 1e-6)
    cooldown_norm = jnp.clip(tag_cooldown / denom, 0.0, 1.0)

    hud = jnp.stack(
        [
            carrying_flag.astype(jnp.float32),
            own_flag_home,
            enemy_flag_home,
            in_own_base,
            cooldown_norm,
        ],
        axis=-1,
    )
    return jnp.concatenate([prev_action_vec, hud], axis=-1)
