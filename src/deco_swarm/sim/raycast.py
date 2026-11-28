import jax.numpy as jnp

from deco_swarm.config.arena import Arena
from deco_swarm.config.schema import PerceptionConfig

# ray classes (SIM_CONTRACT order)
NONE, WALL, ALLY, ENEMY, OWN_FLAG, ENEMY_FLAG, OWN_BASE, ENEMY_BASE = range(8)
NUM_CLASSES = 8
_EPS = 1e-6

def _cross(v, w):
    return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]

def _ray_segments(origin, dirs, segments, segment_mask):
    o = origin[:, None, None, :]                    # (N, 1, 1, 2)
    d = dirs[:, :, None, :]                         # (N, R, 1, 2)
    a = segments[None, None, :, :2]                 # (1, 1, S, 2)
    e = segments[None, None, :, 2:] - a             # segment direction
    denom = _cross(d, e)                            # 0 -> parallel
    safe = jnp.where(jnp.abs(denom) > _EPS, denom, 1.0)
    t = _cross(a - o, e) / safe                     # distance along the ray
    u = _cross(a - o, d) / safe                     # position along the segment, 0..1
    hit = (jnp.abs(denom) > _EPS) & (t > _EPS) & (u >= 0.0) & (u <= 1.0) & segment_mask
    return jnp.where(hit, t, jnp.inf)

def _ray_circles(origin, dirs, centers, radii, valid):
    """Entry distance into circles, (N, R, M). Origin inside a circle -> that circle is not seen."""
    oc = origin[:, None, None, :] - centers[None, None, :, :]     # (N, 1, M, 2)
    b = jnp.sum(oc * dirs[:, :, None, :], axis=-1)                # (N, R, M)
    c = jnp.sum(oc * oc, axis=-1) - radii**2                      # (N, 1, M)
    disc = b * b - c
    t = -b - jnp.sqrt(jnp.maximum(disc, 0.0))
    hit = (c > 0.0) & (disc >= 0.0) & (t > 0.0) & valid
    return jnp.where(hit, t, jnp.inf)

def cast_rays(pos, heading_deg, alive, carrying, team, flag_pos, flag_visible,  arena: Arena, radius, perception: PerceptionConfig):
    """Semantic rays for every agent of one arena.

    pos (N, 2), heading_deg (N,), alive/carrying (N,) bool, team (N,) int (0 red, 1 blue),
    flag_pos (2, 2), flag_visible (2,) bool (False while carried).
    Returns dist (N, R) meters, cls (N, R) int, carrier (N, R) bool.
    """
    n = pos.shape[0]
    rel = jnp.linspace(-perception.fov_deg / 2, perception.fov_deg / 2, perception.ray_count)
    ang = jnp.deg2rad(heading_deg[:, None] + rel[None, :])
    dirs = jnp.stack([jnp.cos(ang), jnp.sin(ang)], axis=-1)

    # all circles: N agents, 2 flags, 2 bases
    teams01 = jnp.array([0, 1])
    centers = jnp.concatenate([pos, flag_pos, arena.base[:, :2]])                       # (M, 2)
    radii = jnp.concatenate([jnp.full(n, radius), jnp.full(2, arena.flag_radius), arena.base[:, 2]])
    valid = jnp.concatenate([alive, flag_visible, jnp.ones(2, bool)])
    c_team = jnp.concatenate([team, teams01, teams01])
    own_code = jnp.concatenate([jnp.full(n, ALLY), jnp.full(2, OWN_FLAG), jnp.full(2, OWN_BASE)])
    c_carrier = jnp.concatenate([carrying, jnp.zeros(4, bool)])

    same = c_team[None, :] == team[:, None]
    c_cls = jnp.where(same, own_code[None, :], own_code[None, :] + 1)

    t_seg = _ray_segments(pos, dirs, arena.segments, arena.segment_mask)
    t_circ = _ray_circles(pos, dirs, centers, radii, valid)
    t_all = jnp.concatenate([t_seg, t_circ], axis=-1)
    t_all = jnp.where(t_all <= perception.max_range_m, t_all, jnp.inf)

    k = jnp.argmin(t_all, axis=-1)
    t_min = jnp.min(t_all, axis=-1)
    hit_any = jnp.isfinite(t_min)
    s = arena.segments.shape[0]
    is_circle = k >= s
    ci = jnp.clip(k - s, 0, centers.shape[0] - 1)

    cls_circle = jnp.take_along_axis(c_cls, ci, axis=1)
    cls = jnp.where(is_circle, cls_circle, WALL)
    cls = jnp.where(hit_any, cls, NONE)
    carrier = hit_any & is_circle & c_carrier[ci]
    dist = jnp.where(hit_any, t_min, perception.max_range_m)
    return dist, cls, carrier