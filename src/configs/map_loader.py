"""Symmetric CTF map YAML loader producing padded JAX-compatible ArenaData PyTrees."""

from pathlib import Path
from typing import Any, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import yaml

DEFAULT_MAX_SEGMENTS = 64
DEFAULT_MAX_TEAM_SIZE = 5


class ArenaData(NamedTuple):
    """Immutable, fixed-shape 2D CTF arena representation for JAX simulation."""

    width: jax.Array
    height: jax.Array
    segments: jax.Array          # (max_segments, 2, 2)
    segment_mask: jax.Array      # (max_segments,) bool
    red_base: jax.Array          # (3,) -> [x, y, radius]
    blue_base: jax.Array         # (3,) -> [x, y, radius]
    red_flag_pos: jax.Array      # (2,) -> [x, y]
    blue_flag_pos: jax.Array     # (2,) -> [x, y]
    flag_radius: jax.Array       # ()
    red_spawns: jax.Array        # (max_team_size, 3) -> [x, y, heading_deg]
    blue_spawns: jax.Array       # (max_team_size, 3) -> [x, y, heading_deg]


def _compile_segments(width: float, height: float, walls_cfg: dict[str, Any]) -> np.ndarray:
    """Compile outer boundary walls, free segments, and box obstacles into (N, 2, 2)."""
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]

    for seg in walls_cfg.get("segments", []):
        p1, p2 = seg[0], seg[1]
        segs.append(((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1]))))

    for box in walls_cfg.get("boxes", []):
        bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        segs.extend([
            ((bx, by), (bx + bw, by)),
            ((bx + bw, by), (bx + bw, by + bh)),
            ((bx + bw, by + bh), (bx, by + bh)),
            ((bx, by + bh), (bx, by)),
        ])

    return np.asarray(segs, dtype=np.float32)


def _pad_spawns(spawns: Sequence[Sequence[float]], max_team_size: int) -> np.ndarray:
    """Pad or validate team spawn poses into (max_team_size, 3)."""
    arr = np.asarray(spawns, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected spawns shape (K, 3), got {arr.shape}")
    if arr.shape[0] > max_team_size:
        raise ValueError(f"Spawn count {arr.shape[0]} exceeds max_team_size={max_team_size}")
    out = np.zeros((max_team_size, 3), dtype=np.float32)
    out[: arr.shape[0]] = arr
    if arr.shape[0] < max_team_size:
        # Repeat existing spawns cyclically so any team_size <= max_team_size is valid
        for i in range(arr.shape[0], max_team_size):
            out[i] = arr[i % arr.shape[0]]
    return out


def load_arena(
    path: str | Path,
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_team_size: int = DEFAULT_MAX_TEAM_SIZE,
) -> ArenaData:
    """Load a single symmetric CTF map YAML into a padded ArenaData PyTree."""
    map_path = Path(path)
    with map_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    dims = raw["dimensions"]
    width = float(dims["width"])
    height = float(dims["height"])

    raw_segs = _compile_segments(width, height, raw.get("walls", {}))
    num_segs = raw_segs.shape[0]
    if num_segs > max_segments:
        raise ValueError(f"Map {map_path} has {num_segs} segments > max_segments={max_segments}")

    padded_segs = np.zeros((max_segments, 2, 2), dtype=np.float32)
    padded_segs[:num_segs] = raw_segs
    seg_mask = np.zeros((max_segments,), dtype=bool)
    seg_mask[:num_segs] = True

    bases = raw["bases"]
    flags = raw["flags"]
    spawns = raw["spawns"]

    return ArenaData(
        width=jnp.asarray(width, dtype=jnp.float32),
        height=jnp.asarray(height, dtype=jnp.float32),
        segments=jnp.asarray(padded_segs),
        segment_mask=jnp.asarray(seg_mask),
        red_base=jnp.asarray(bases["red"], dtype=jnp.float32),
        blue_base=jnp.asarray(bases["blue"], dtype=jnp.float32),
        red_flag_pos=jnp.asarray(flags["red"], dtype=jnp.float32),
        blue_flag_pos=jnp.asarray(flags["blue"], dtype=jnp.float32),
        flag_radius=jnp.asarray(float(flags.get("radius", 18.0)), dtype=jnp.float32),
        red_spawns=jnp.asarray(_pad_spawns(spawns["red"], max_team_size)),
        blue_spawns=jnp.asarray(_pad_spawns(spawns["blue"], max_team_size)),
    )


def load_arenas(
    paths: Sequence[str | Path],
    max_segments: int = DEFAULT_MAX_SEGMENTS,
    max_team_size: int = DEFAULT_MAX_TEAM_SIZE,
) -> ArenaData:
    """Load and stack multiple maps into a batched ArenaData PyTree for domain randomization."""
    if not paths:
        raise ValueError("paths must contain at least one map file")
    arenas = [load_arena(p, max_segments=max_segments, max_team_size=max_team_size) for p in paths]
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *arenas)
