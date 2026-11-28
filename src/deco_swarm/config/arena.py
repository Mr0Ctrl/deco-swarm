from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from .loader import ConfigError
from .schema import PhysicsConfig

TOL = 1e-5
RED, BLUE = 0, 1
MAP_KEYS = {"name", "symmetry", "dimensions", "bases", "flags", "spawns", "walls"}

class Arena(NamedTuple):
    segments: jax.Array      # (max_segments, 4)  [x1, y1, x2, y2], padding rows = 0
    segment_mask: jax.Array  # (max_segments,)    bool
    size: jax.Array          # (2,)               [W, H]
    base: jax.Array          # (2, 3)             [team, (x, y, radius)]
    flag_stand: jax.Array    # (2, 2)             [team, (x, y)]
    flag_radius: jax.Array   # ()
    spawns: jax.Array        # (2, max_team_size, 3)  [team, i, (x, y, heading_deg)]
    spawn_mask: jax.Array    # (2, max_team_size) bool

def _arr(value, shape, path):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
        raise ConfigError(f"{path}: expected shape {shape}, got {a.shape}")
    return a

def _wall_segments(W, H, walls):
    segs = [[0, 0, W, 0], [W, 0, W, H], [W, H, 0, H], [0, H, 0, 0]] # outer border
    for x, y, w, h in walls["boxes"]: # box -> 4 edges
        segs += [[x, y, x + w, y], [x + w, y, x + w, y + h],
                 [x + w, y + h, x, y + h], [x, y + h, x, y]]
    for (x1, y1), (x2, y2) in walls["segments"]:
        segs.append([x1, y1, x2, y2])
    return np.asarray(segs, dtype=np.float32).reshape(-1, 4)

def _rot_points(p, W, H): 
    return np.stack([W - p[..., 0], H - p[..., 1]], axis=-1)

def _same_set(a, b):
    if len(a) != len(b):
        return False
    if len(a) == 0:
        return True
    close = np.abs(a[:, None, :] - b[None, :, :]).max(-1) < TOL     # (len_a, len_b)
    return bool(close.any(1).all() and close.any(0).all())

def _check_rotate_180(segs, base, flag, spawns, W, H, name):
    def fail(what):
        raise ConfigError(f"{name}: not rotate_180 symmetric ({what})")

    if not np.allclose(_rot_points(base[RED, :2], W, H), base[BLUE, :2], atol=TOL) \
            or abs(base[RED, 2] - base[BLUE, 2]) > TOL:
        fail("bases")
    if not np.allclose(_rot_points(flag[RED], W, H), flag[BLUE], atol=TOL):
        fail("flags")

    def pose_feats(s):
        h = np.deg2rad(s[:, 2])
        return np.concatenate([s[:, :2], np.cos(h)[:, None], np.sin(h)[:, None]], axis=1)
    red_rot = np.concatenate([_rot_points(spawns[RED][:, :2], W, H),
                              (spawns[RED][:, 2:] + 180.0) % 360.0], axis=1)
    if not _same_set(pose_feats(red_rot), pose_feats(spawns[BLUE])):
        fail("spawns")

    rot = np.concatenate([_rot_points(segs[:, :2], W, H), _rot_points(segs[:, 2:], W, H)], axis=1)
    both = lambda s: np.concatenate([s, s[:, [2, 3, 0, 1]]])          # [a,b] == [b,a]
    if not _same_set(both(segs), both(rot)):
        fail("walls")

def compile_arena(path, physics: PhysicsConfig) -> Arena:
    path = Path(path)
    m = yaml.safe_load(path.read_text(encoding="utf-8"))
    name = path.stem
    if set(m) != MAP_KEYS:
        raise ConfigError(f"{name}: keys must be {sorted(MAP_KEYS)}, got {sorted(m)}")

    W, H = float(m["dimensions"]["width_m"]), float(m["dimensions"]["height_m"])
    segs = _wall_segments(W, H, m["walls"])
    base = np.stack([_arr(m["bases"][t], (3,), f"{name}.bases.{t}") for t in ("red", "blue")])
    flag = np.stack([_arr(m["flags"][t], (2,), f"{name}.flags.{t}") for t in ("red", "blue")])
    spawns = [_arr(m["spawns"][t], (None, 3), f"{name}.spawns.{t}") for t in ("red", "blue")]

    # --- validity ---
    if len(segs) > physics.max_segments:
        raise ConfigError(f"{name}: {len(segs)} segments > max_segments={physics.max_segments}")
    if len(spawns[RED]) != len(spawns[BLUE]):
        raise ConfigError(f"{name}: red/blue spawn counts differ")
    if len(spawns[RED]) > physics.max_team_size:
        raise ConfigError(f"{name}: {len(spawns[RED])} spawns > max_team_size={physics.max_team_size}")
    pts = np.concatenate([base[:, :2], flag, spawns[RED][:, :2], spawns[BLUE][:, :2]])
    if (pts < 0).any() or (pts[:, 0] > W).any() or (pts[:, 1] > H).any():
        raise ConfigError(f"{name}: base/flag/spawn outside the map")
    if m["symmetry"] == "rotate_180":
        _check_rotate_180(segs, base, flag, spawns, W, H, name)
    elif m["symmetry"] != "none":
        raise ConfigError(f"{name}: unknown symmetry {m['symmetry']!r}")

    # --- pad to fixed shapes ---
    S, K = physics.max_segments, physics.max_team_size
    seg_pad = np.zeros((S, 4), np.float32); seg_pad[: len(segs)] = segs
    seg_mask = np.zeros(S, bool);           seg_mask[: len(segs)] = True
    sp_pad = np.zeros((2, K, 3), np.float32)
    sp_mask = np.zeros((2, K), bool)
    for t in (RED, BLUE):
        sp_pad[t, : len(spawns[t])] = spawns[t]
        sp_mask[t, : len(spawns[t])] = True

    return Arena(
        segments=jnp.asarray(seg_pad),
        segment_mask=jnp.asarray(seg_mask),
        size=jnp.asarray([W, H], jnp.float32),
        base=jnp.asarray(base),
        flag_stand=jnp.asarray(flag),
        flag_radius=jnp.asarray(m["flags"]["radius_m"], jnp.float32),
        spawns=jnp.asarray(sp_pad),
        spawn_mask=jnp.asarray(sp_mask),
    )

def stack_arenas(arenas):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *arenas)