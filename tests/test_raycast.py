"""Unit tests for 2-layer team-relative semantic raycasting and ego scalars."""

import jax
import jax.numpy as jnp
import pytest

from src.sim import cast_all_rays, compute_ego_scalars


def _base_scene():
    # Simple wall at x=400 blocking anything behind it
    segments = jnp.asarray([[[400.0, 0.0], [400.0, 500.0]]], dtype=jnp.float32)
    segment_mask = jnp.asarray([True], dtype=bool)
    red_base = jnp.asarray([100.0, 200.0, 50.0], dtype=jnp.float32)
    blue_base = jnp.asarray([350.0, 200.0, 50.0], dtype=jnp.float32)
    red_flag = jnp.asarray([100.0, 200.0], dtype=jnp.float32)
    blue_flag = jnp.asarray([350.0, 200.0], dtype=jnp.float32)
    return segments, segment_mask, red_base, blue_base, red_flag, blue_flag


def test_raycast_team_relative_and_non_blocking_base():
    segments, seg_mask, red_base, blue_base, red_flag, blue_flag = _base_scene()
    # Agent 0 (Red) at (200, 200) looking 0 deg (+x) toward:
    # - Blue base & flag at x=350 (radius 50 -> edge at 300; flag radius 18 -> edge at 332)
    # - Wall behind base at x=400
    # - Agent 1 (Blue) at (370, 200) (radius 10 -> edge at 360, behind flag/inside base!)
    pos = jnp.asarray([[200.0, 200.0], [370.0, 200.0]], dtype=jnp.float32)
    heading = jnp.asarray([0.0, 180.0], dtype=jnp.float32)
    teams = jnp.asarray([0, 1], dtype=jnp.int32)
    alive = jnp.asarray([True, True])
    carrying = jnp.asarray([False, False])

    rays = jax.jit(
        cast_all_rays,
        static_argnames=("agent_radius", "flag_radius", "ray_count", "fov_degrees", "max_distance"),
    )(
        pos, heading, teams, alive, carrying,
        segments, seg_mask, red_flag, blue_flag, red_base, blue_base,
        10.0, 18.0, 91, 180.0, 500.0,
    )
    assert rays.shape == (2, 91, 12)
    center = 45  # 0 deg relative angle -> straight ahead

    # Red agent (0) center ray sees:
    # - Solid hit = Blue enemy at 360 - 200 = 160 px -> 0.32 (base did NOT block enemy!)
    assert float(rays[0, center, 0]) == pytest.approx(160.0 / 500.0, abs=1e-4)
    assert float(rays[0, center, 3]) == 1.0  # is_enemy
    # - Flag hit = Blue (enemy) flag at 332 - 200 = 132 px -> 0.264
    assert float(rays[0, center, 4]) == pytest.approx(132.0 / 500.0, abs=1e-4)
    assert float(rays[0, center, 6]) == 1.0  # is_enemy_flag
    # - Base hit = Blue (enemy) base edge at 300 - 200 = 100 px -> 0.20
    assert float(rays[0, center, 7]) == pytest.approx(100.0 / 500.0, abs=1e-4)
    assert float(rays[0, center, 9]) == 1.0  # is_enemy_base

    # Blue agent (1) is standing INSIDE Blue (own) base looking 180 deg (-x) toward Red agent (0):
    # - Base dist = 0.0 and is_own_base = 1.0
    assert float(rays[1, center, 7]) == pytest.approx(0.0, abs=1e-5)
    assert float(rays[1, center, 8]) == 1.0  # is_own_base
    # - Solid hit = Red enemy at 370 - 210 = 160 px -> 0.32
    assert float(rays[1, center, 0]) == pytest.approx(160.0 / 500.0, abs=1e-4)
    assert float(rays[1, center, 3]) == 1.0  # is_enemy


def test_wall_occludes_flag_and_carried_flag_aligns_with_carrier():
    # Wall at x=250 blocks Red agent at x=100 from seeing Blue flag at x=350
    segments = jnp.asarray([[[250.0, 0.0], [250.0, 500.0]]], dtype=jnp.float32)
    seg_mask = jnp.asarray([True], dtype=bool)
    red_base = jnp.asarray([50.0, 200.0, 50.0], dtype=jnp.float32)
    blue_base = jnp.asarray([450.0, 200.0, 50.0], dtype=jnp.float32)

    # Red agent 0 at (100, 200), Blue enemy 1 at (200, 200) carrying Red flag!
    pos = jnp.asarray([[100.0, 200.0], [200.0, 200.0]], dtype=jnp.float32)
    heading = jnp.asarray([0.0, 180.0], dtype=jnp.float32)
    teams = jnp.asarray([0, 1], dtype=jnp.int32)
    alive = jnp.asarray([True, True])
    carrying = jnp.asarray([False, True])
    red_flag = jnp.asarray([200.0, 200.0], dtype=jnp.float32)   # carried by agent 1
    blue_flag = jnp.asarray([450.0, 200.0], dtype=jnp.float32)  # behind wall x=250

    rays = cast_all_rays(
        pos, heading, teams, alive, carrying,
        segments, seg_mask, red_flag, blue_flag, red_base, blue_base,
        10.0, 18.0, 91, 180.0, 500.0,
    )
    center = 45
    # Agent 0 sees Blue carrier at 90 px AND own_flag on carrier, NOT blue_flag behind wall
    assert float(rays[0, center, 3]) == 1.0  # is_enemy
    assert float(rays[0, center, 5]) == 1.0  # is_own_flag
    assert float(rays[0, center, 6]) == 0.0  # enemy_flag occluded by wall!

    # Carrier (agent 1) has carrying=True -> enemy_flag (red_flag for Blue) hidden from self
    assert float(rays[1, center, 6]) == 0.0


def test_compute_ego_scalars():
    pos = jnp.asarray([[100.0, 200.0], [350.0, 200.0]], dtype=jnp.float32)
    teams = jnp.asarray([0, 1], dtype=jnp.int32)
    prev_act = jnp.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 1.0, 1.0]], dtype=jnp.float32)
    carrying = jnp.asarray([False, True])
    red_home = jnp.asarray(False)
    blue_home = jnp.asarray(True)
    red_base = jnp.asarray([100.0, 200.0, 50.0], dtype=jnp.float32)
    blue_base = jnp.asarray([900.0, 200.0, 50.0], dtype=jnp.float32)
    cooldown = jnp.asarray([0.4, 0.0], dtype=jnp.float32)

    ego = compute_ego_scalars(
        pos, teams, prev_act, carrying, red_home, blue_home, red_base, blue_base,
        cooldown, 10.0, 0.8,
    )
    assert ego.shape == (2, 9)
    # Agent 0 (Red): in_own_base=1, own_flag_home=0, enemy_flag_home=1, cooldown=0.5
    assert ego[0, 4:].tolist() == pytest.approx([0.0, 0.0, 1.0, 1.0, 0.5])
    # Agent 1 (Blue): carrying=1, own_flag_home=1, enemy_flag_home=0, in_own_base=0, cooldown=0.0
    assert ego[1, 4:].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0])
