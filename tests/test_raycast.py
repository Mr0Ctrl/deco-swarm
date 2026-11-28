from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.config.schema import PerceptionConfig
from deco_swarm.sim.raycast import ALLY, ENEMY, NONE, OPP, OWN, RAY_FEATURES, WALL, cast_rays, ray_features

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
ARENA = compile_arena(CONFIGS / "arenas" / "ctf_open.yaml", CFG.simulation.physics)
R = CFG.agent.body.radius_m
P3 = PerceptionConfig(ray_count=3, fov_deg=180.0, max_range_m=25.0)   # rays at -90, 0, +90
TEAM = jnp.array([0, 0, 1, 1])                                      # red_0, red_1, blue_0, blue_1
FAR = [[10.0, 5.0], [20.0, 5.0], [40.0, 5.0], [50.0, 5.0]]          # out of the way
MID = 1                                                             # index of the 0-degree ray


def cast(pos, heading, alive=None, carrying=None, perception=P3):
    pos = jnp.array(pos, jnp.float32)
    alive = jnp.ones(4, bool) if alive is None else jnp.array(alive)
    carrying = jnp.zeros(4, bool) if carrying is None else jnp.array(carrying)
    return cast_rays(pos, jnp.array(heading, jnp.float32), alive, carrying, TEAM,
                     ARENA.flag_stand, jnp.ones(2, bool), ARENA, R, perception)


def place(**where):
    pos = [p[:] for p in FAR]
    for i, p in where.items():
        pos[int(i[1:])] = p
    return pos


def test_wall_ahead_and_nothing_sideways():
    h = cast(place(a0=[30.0, 30.0]), [90.0, 0, 0, 0])                # red_0 looks up
    assert int(h.solid_cls[0, MID]) == WALL
    np.testing.assert_allclose(h.solid_dist[0, MID], 10.0, atol=1e-4)
    assert int(h.solid_cls[0, 0]) == NONE and int(h.solid_cls[0, 2]) == NONE
    np.testing.assert_allclose(h.solid_dist[0, 0], 25.0)


def test_nearest_solid_wins_and_team_relative():
    h = cast(place(a0=[30.0, 30.0], a2=[30.0, 35.0]), [90.0, 0, 270.0, 0])
    assert int(h.solid_cls[0, MID]) == ENEMY
    np.testing.assert_allclose(h.solid_dist[0, MID], 5.0 - R, atol=1e-4)
    assert int(h.solid_cls[2, MID]) == ENEMY                          # blue_0 sees red_0 as enemy
    h = cast(place(a0=[30.0, 30.0], a1=[30.0, 35.0]), [90.0, 0, 0, 0])
    assert int(h.solid_cls[0, MID]) == ALLY


def test_enemy_inside_base_is_visible():
    # blue base (54, 20) r 2.5. blue_0 stands inside it; red_0 looks at it from 10 m.
    h = cast(place(a0=[44.0, 21.5], a2=[54.0, 21.5]), [0.0, 0, 0, 0])
    assert int(h.solid_cls[0, MID]) == ENEMY                          # base no longer hides the defender
    np.testing.assert_allclose(h.solid_dist[0, MID], 10.0 - R, atol=1e-4)
    assert int(h.base_cls[0, MID]) == OPP                             # and the base is seen too
    np.testing.assert_allclose(h.base_dist[0, MID], 8.0, atol=1e-4)  # enters the circle at x = 52


def test_objects_behind_a_solid_are_hidden():
    h = cast(place(a0=[44.0, 20.0], a2=[50.0, 20.0]), [0.0, 0, 0, 0])  # blue_0 between red_0 and flag/base
    assert int(h.solid_cls[0, MID]) == ENEMY
    assert int(h.flag_cls[0, MID]) == NONE and int(h.base_cls[0, MID]) == NONE


def test_inside_own_base_sees_flag_not_base():
    h = cast(place(a0=[6.0, 21.5]), [270.0, 0, 0, 0])               # inside red base, looking at the flag
    assert int(h.flag_cls[0, MID]) == OWN
    assert int(h.base_cls[0, MID]) == NONE


def test_dead_agent_invisible():
    h = cast(place(a0=[30.0, 30.0], a2=[30.0, 35.0]), [90.0, 0, 0, 0], alive=[True, True, False, True])
    assert int(h.solid_cls[0, MID]) == WALL


def test_carrier_bit():
    h = cast(place(a0=[30.0, 30.0], a2=[30.0, 35.0]), [90.0, 0, 0, 0], carrying=[False, False, True, False])
    assert bool(h.carrier[0, MID]) and not bool(h.carrier[0, 0])


def test_features_shape_and_range():
    p = CFG.agent.perception
    pos = jnp.tile(jnp.array(place(a0=[30.0, 30.0]), jnp.float32), (8, 1, 1))
    f = jax.jit(jax.vmap(lambda ps: ray_features(
        cast_rays(ps, jnp.zeros(4), jnp.ones(4, bool), jnp.zeros(4, bool), TEAM,
                  ARENA.flag_stand, jnp.ones(2, bool), ARENA, R, p), p.max_range_m)))
    feats = f(pos)
    assert feats.shape == (8, 4, p.ray_count, RAY_FEATURES)
    assert float(feats.min()) >= 0.0 and float(feats.max()) <= 1.0
