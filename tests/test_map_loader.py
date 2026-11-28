"""Unit tests for symmetric CTF map loading and JAX array padding."""

import jax
import jax.numpy as jnp
import pytest

from src.configs import ArenaData, load_arena, load_arenas


def test_load_open_arena_segments_and_symmetry():
    arena = load_arena("configs/maps/ctf_open.yaml", max_segments=32, max_team_size=5)
    assert isinstance(arena, ArenaData)
    assert float(arena.width) == 1200.0
    assert float(arena.height) == 800.0
    assert arena.segments.shape == (32, 2, 2)
    assert arena.segment_mask.shape == (32,)
    assert int(jnp.sum(arena.segment_mask)) == 4
    assert bool(jnp.all(arena.segments[4:] == 0.0))

    # Red/Blue bases and flags are symmetric around arena center
    center = jnp.array([arena.width, arena.height])
    assert jnp.allclose(arena.red_base[:2] + arena.blue_base[:2], center)
    assert jnp.allclose(arena.red_flag_pos + arena.blue_flag_pos, center)
    assert arena.red_spawns.shape == (5, 3)
    assert arena.blue_spawns.shape == (5, 3)


def test_load_pillars_arena_boxes_and_segments():
    arena = load_arena("configs/maps/ctf_pillars.yaml", max_segments=32, max_team_size=5)
    # 4 outer walls + 1 center segment + 4 boxes * 4 edges = 21 segments
    assert int(jnp.sum(arena.segment_mask)) == 21
    assert bool(jnp.all(arena.segment_mask[:21]))
    assert bool(jnp.all(~arena.segment_mask[21:]))


def test_exceeding_max_segments_raises():
    with pytest.raises(ValueError, match="max_segments"):
        load_arena("configs/maps/ctf_pillars.yaml", max_segments=10)


def test_load_arenas_batches_and_jits():
    batch = load_arenas(
        ["configs/maps/ctf_open.yaml", "configs/maps/ctf_pillars.yaml"],
        max_segments=32,
        max_team_size=5,
    )
    assert batch.segments.shape == (2, 32, 2, 2)
    assert batch.segment_mask.shape == (2, 32)

    @jax.jit
    def active_counts(b: ArenaData) -> jax.Array:
        return jnp.sum(b.segment_mask, axis=1)

    counts = active_counts(batch)
    assert counts.tolist() == [4, 21]
