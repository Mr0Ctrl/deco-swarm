from typing import NamedTuple

import jax
import jax.numpy as jnp

from deco_swarm.config.arena import Arena
from deco_swarm.config.schema import PerceptionConfig

# layer 1 (solid, blocks rays): class values
NONE, WALL, ALLY, ENEMY = 0, 1, 2, 3
# layers 2-3 (flags, bases; seen if nothing solid is in front, never block): class values
OWN, OPP = 1, 2
# per ray: [solid_d, wall, ally, enemy, carrier, flag_d, own_flag, opp_flag, base_d, own_base, opp_base]
RAY_FEATURES = 11
_EPS = 1e-6


class RayHits(NamedTuple):
    solid_dist: jax.Array   # (N, R) meters, max_range if nothing
    solid_cls: jax.Array    # (N, R) NONE / WALL / ALLY / ENEMY
    carrier: jax.Array      # (N, R) bool, solid hit is a flag carrier
    flag_dist: jax.Array    # (N, R)
    flag_cls: jax.Array     # (N, R) NONE / OWN / OPP
    base_dist: jax.Array    # (N, R)
    base_cls: jax.Array     # (N, R) NONE / OWN / OPP


def _cross(v, w):
    return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]


def _ray_segments(origin, dirs, segments, segment_mask):
    """origin (N, 2), dirs (N, R, 2), segments (S, 4) -> hit distance (N, R, S), inf = no hit."""
    o = origin[:, None, None, :]
    d = dirs[:, :, None, :]
    a = segments[None, None, :, :2]
    e = segments[None, None, :, 2:] - a
    denom = _cross(d, e)
    safe = jnp.where(jnp.abs(denom) > _EPS, denom, 1.0)
    t = _cross(a - o, e) / safe
    u = _cross(a - o, d) / safe
    hit = (jnp.abs(denom) > _EPS) & (t > _EPS) & (u >= 0.0) & (u <= 1.0) & segment_mask
    return jnp.where(hit, t, jnp.inf)


def _ray_circles(origin, dirs, centers, radii, valid):
    """Entry distance into circles, (N, R, M). Origin inside a circle -> that circle is not seen."""
    oc = origin[:, None, None, :] - centers[None, None, :, :]
    b = jnp.sum(oc * dirs[:, :, None, :], axis=-1)
    c = jnp.sum(oc * oc, axis=-1) - radii**2
    disc = b * b - c
    t = -b - jnp.sqrt(jnp.maximum(disc, 0.0))
    hit = (c > 0.0) & (disc >= 0.0) & (t > 0.0) & valid
    return jnp.where(hit, t, jnp.inf)


def _nearest(t, cls_table):
    """t (N, R, M) distances, cls_table (N, M) class per candidate -> dist, cls, index, hit (N, R)."""
    k = jnp.argmin(t, axis=-1)
    d = jnp.min(t, axis=-1)
    hit = jnp.isfinite(d)
    cls = jnp.where(hit, jnp.take_along_axis(cls_table, k, axis=1), NONE)
    return d, cls, k, hit


def cast_rays(pos, heading_deg, alive, carrying, team, flag_pos, flag_visible,
              arena: Arena, radius, perception: PerceptionConfig) -> RayHits:
    """3-group semantic rays: solid (walls, agents) + flags + bases, for every agent of one arena."""
    n = pos.shape[0]
    max_r = perception.max_range_m
    rel = jnp.linspace(-perception.fov_deg / 2, perception.fov_deg / 2, perception.ray_count)
    ang = jnp.deg2rad(heading_deg[:, None] + rel[None, :])
    dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)                # (N, R, 2)
    in_range = lambda t: jnp.where(t <= max_r, t, jnp.inf)

    # ---- layer 1: solid (walls + agents), nearest one blocks the ray ----
    s = arena.segments.shape[0]
    t_wall = _ray_segments(pos, dirs, arena.segments, arena.segment_mask)   # (N, R, S)
    t_agent = _ray_circles(pos, dirs, pos, jnp.full(n, radius), alive)      # (N, R, N)
    agent_cls = jnp.where(team[None, :] == team[:, None], ALLY, ENEMY)      # (N, N)
    table = jnp.concatenate([jnp.full((n, s), WALL), agent_cls], axis=1)
    solid_d, solid_cls, k, solid_hit = _nearest(in_range(jnp.concatenate([t_wall, t_agent], -1)), table)
    carrier = solid_hit & (k >= s) & carrying[jnp.clip(k - s, 0, n - 1)]

    # ---- layers 2-3: flags and bases, visible only if not behind a solid ----
    owner = jnp.array([0, 1])                                               # flag/base t belongs to team t

    def objects(centers, radii, valid):
        t = in_range(_ray_circles(pos, dirs, centers, radii, valid))
        t = jnp.where(t <= solid_d[..., None], t, jnp.inf)                 # occluded -> unseen
        cls_table = jnp.where(owner[None, :] == team[:, None], OWN, OPP)    # (N, 2)
        d, cls, _, _ = _nearest(t, cls_table)
        return d, cls

    flag_d, flag_cls = objects(flag_pos, jnp.full(2, arena.flag_radius), flag_visible)
    base_d, base_cls = objects(arena.base[:, :2], arena.base[:, 2], jnp.ones(2, bool))

    fill = lambda d: jnp.where(jnp.isfinite(d), d, max_r)
    return RayHits(fill(solid_d), solid_cls, carrier, fill(flag_d), flag_cls, fill(base_d), base_cls)


def ray_features(hits: RayHits, max_range_m):
    """RayHits -> (N, R, RAY_FEATURES) float32, all in [0, 1]."""
    f32 = jnp.float32
    oh = lambda c, k: jax.nn.one_hot(c, k)[..., 1:]                         # drop the NONE column
    return jnp.concatenate([
        (hits.solid_dist / max_range_m)[..., None], oh(hits.solid_cls, 4), hits.carrier[..., None].astype(f32),
        (hits.flag_dist / max_range_m)[..., None], oh(hits.flag_cls, 3),
        (hits.base_dist / max_range_m)[..., None], oh(hits.base_cls, 3),
    ], axis=-1).astype(f32)