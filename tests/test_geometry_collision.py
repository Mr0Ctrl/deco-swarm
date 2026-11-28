"""Unit tests for pure-JAX kinematics, action decoding, and collisions."""

import jax
import jax.numpy as jnp
import pytest

from src.sim import (
    decode_actions,
    resolve_agent_agent,
    resolve_agent_walls,
    resolve_circle_segment,
    step_kinematics,
)


def test_decode_actions():
    actions = jnp.asarray([
        [0, 0, 0],  # No-Op
        [1, 1, 1],  # Forward, Left, Tag
        [2, 2, 0],  # Backward, Right, No-Tag
        [3, 0, 1],  # Strafe Left, Hold, Tag
        [4, 0, 0],  # Strafe Right, Hold, No-Tag
    ], dtype=jnp.int32)
    fwd, strafe, rot, tag = decode_actions(actions)
    assert fwd.tolist() == [0.0, 1.0, -1.0, 0.0, 0.0]
    assert strafe.tolist() == [0.0, 0.0, 0.0, -1.0, 1.0]
    assert rot.tolist() == [0.0, -1.0, 1.0, 0.0, 0.0]
    assert tag.tolist() == [False, True, False, True, False]


def test_step_kinematics_heading_wrap_quake_cap_and_dead_freeze():
    pos = jnp.asarray([[100.0, 100.0], [200.0, 200.0]], dtype=jnp.float32)
    heading = jnp.asarray([10.0, 45.0], dtype=jnp.float32)
    fwd = jnp.asarray([1.0, 1.0], dtype=jnp.float32)
    strafe = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    rot = jnp.asarray([-1.0, 1.0], dtype=jnp.float32)
    alive = jnp.asarray([True, False])

    new_pos, new_heading = jax.jit(
        step_kinematics, static_argnames=("linear_speed", "angular_speed", "dt")
    )(pos, heading, fwd, strafe, rot, alive, 180.0, 180.0, 0.1)

    # Alive agent rotated -18 deg from 10 -> 352 deg
    assert float(new_heading[0]) == pytest.approx(352.0, abs=1e-4)
    # Quake normalization: total displacement == 180 * 0.1 = 18.0 (not 18 * sqrt(2))
    disp = float(jnp.linalg.norm(new_pos[0] - pos[0]))
    assert disp == pytest.approx(18.0, abs=1e-4)
    # Dead agent stays completely frozen
    assert jnp.allclose(new_pos[1], pos[1])
    assert float(new_heading[1]) == 45.0


def test_circle_segment_edge_guards_and_sliding():
    # Zero-length segment and exact-center overlap must not produce NaN
    zero_seg = jnp.asarray([[50.0, 50.0], [50.0, 50.0]], dtype=jnp.float32)
    p_zero, hit_zero = resolve_circle_segment(
        jnp.asarray([50.0, 50.0]), 10.0, zero_seg, jnp.asarray(True)
    )
    assert not bool(hit_zero)
    assert bool(jnp.all(jnp.isfinite(p_zero)))

    wall = jnp.asarray([[0.0, 0.0], [100.0, 0.0]], dtype=jnp.float32)
    p_exact, hit_exact = resolve_circle_segment(
        jnp.asarray([50.0, 0.0]), 10.0, wall, jnp.asarray(True)
    )
    assert bool(hit_exact)
    assert bool(jnp.all(jnp.isfinite(p_exact)))

    # Sliding along y=0 wall preserves x=42.0 and pushes y to +10.0
    p_slide, hit_slide = resolve_circle_segment(
        jnp.asarray([42.0, 4.0]), 10.0, wall, jnp.asarray(True)
    )
    assert bool(hit_slide)
    assert float(p_slide[0]) == pytest.approx(42.0, abs=1e-5)
    assert float(p_slide[1]) == pytest.approx(10.0, abs=1e-5)


def test_resolve_agent_walls_corner_and_mask():
    # Corner at (0, 0): walls x=0 and y=0 plus a masked dummy wall near the agent
    segments = jnp.asarray([
        [[0.0, 0.0], [100.0, 0.0]],
        [[0.0, 0.0], [0.0, 100.0]],
        [[5.0, 0.0], [5.0, 100.0]],  # Masked out!
    ], dtype=jnp.float32)
    mask = jnp.asarray([True, True, False])
    pos = jnp.asarray([[3.0, 3.0]], dtype=jnp.float32)
    alive = jnp.asarray([True])

    new_pos, hit = jax.jit(resolve_agent_walls, static_argnames=("radius", "iterations"))(
        pos, alive, 10.0, segments, mask, 2
    )
    assert bool(hit[0])
    assert float(new_pos[0, 0]) == pytest.approx(10.0, abs=1e-4)
    assert float(new_pos[0, 1]) == pytest.approx(10.0, abs=1e-4)


def test_resolve_agent_agent_symmetric_and_ignores_dead():
    pos = jnp.asarray([[100.0, 100.0], [110.0, 100.0], [105.0, 100.0]], dtype=jnp.float32)
    alive = jnp.asarray([True, True, False])

    new_pos = jax.jit(resolve_agent_agent, static_argnames=("radius",))(pos, alive, 10.0)
    # Agents 0 and 1 were 10 px apart (min_dist=20) -> pushed 5 px each to 95 and 115
    assert float(new_pos[0, 0]) == pytest.approx(95.0, abs=1e-4)
    assert float(new_pos[1, 0]) == pytest.approx(115.0, abs=1e-4)
    # Dead agent 2 untouched
    assert jnp.allclose(new_pos[2], pos[2])
