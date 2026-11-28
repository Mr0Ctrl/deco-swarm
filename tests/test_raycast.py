from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.config.schema import PerceptionConfig
from deco_swarm.sim.raycast import ALLY, ENEMY, NONE, OWN_FLAG, WALL, cast_rays

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
ARENA = compile_arena(CONFIGS / "arenas" / "ctf_open.yaml", CFG.simulation.physics)
R = CFG.agent.body.radius_m
P3 = PerceptionConfig(ray_count=3, fov_deg=180.0, max_range_m=25.0)   # rays at -90, 0, +90
TEAM = jnp.array([0, 0, 1, 1])                                      # red_0, red_1, blue_0, blue_1
FAR = [[10.0, 5.0], [20.0, 5.0], [40.0, 5.0], [50.0, 5.0]]          # out of the way


def cast(pos, heading, alive=None, carrying=None, perception=P3):
    pos = jnp.array(pos, jnp.float32)
    alive = jnp.ones(4, bool) if alive is None else jnp.array(alive)
    carrying = jnp.zeros(4, bool) if carrying is None else jnp.array(carrying)
    return cast_rays(pos, jnp.array(heading, jnp.float32), alive, carrying, TEAM,
                     ARENA.flag_stand, jnp.ones(2, bool), ARENA, R, perception)


def place(**where):
    """place(a0=[x, y], a2=[x, y]) -> agent positions; others stay at FAR."""
    pos = [p[:] for p in FAR]
    for i, p in where.items():
        pos[int(i[1:])] = p
    return pos


def test_wall_ahead_and_nothing_sideways():
    dist, cls, _ = cast(place(a0=[30.0, 30.0]), [90.0, 0, 0, 0])    # red_0 looks up
    assert int(cls[0, 1]) == WALL
    np.testing.assert_allclose(dist[0, 1], 10.0, atol=1e-4)          # top wall at y = 40
    assert int(cls[0, 0]) == NONE and int(cls[0, 2]) == NONE         # 30 m to the sides > 25 m
    np.testing.assert_allclose(dist[0, 0], 25.0)


def test_nearest_hit_wins_and_team_relative():
    pos = place(a0=[30.0, 30.0], a2=[30.0, 35.0])                     # blue_0 between red_0 and the wall
    dist, cls, _ = cast(pos, [90.0, 0, 270.0, 0])
    assert int(cls[0, 1]) == ENEMY
    np.testing.assert_allclose(dist[0, 1], 5.0 - R, atol=1e-4)
    assert int(cls[2, 1]) == ENEMY                                     # blue_0 looks down at red_0

    pos = place(a0=[30.0, 30.0], a1=[30.0, 35.0])
    _, cls, _ = cast(pos, [90.0, 0, 0, 0])
    assert int(cls[0, 1]) == ALLY


def test_inside_own_base_sees_flag_not_base():
    # red base center (6, 20), radius 2.5; flag radius 0.9
    _, cls, _ = cast(place(a0=[6.0, 21.5]), [270.0, 0, 0, 0])
    assert int(cls[0, 1]) == OWN_FLAG


def test_dead_agent_invisible():
    pos = place(a0=[30.0, 30.0], a2=[30.0, 35.0])
    dist, cls, _ = cast(pos, [90.0, 0, 0, 0], alive=[True, True, False, True])
    assert int(cls[0, 1]) == WALL
    np.testing.assert_allclose(dist[0, 1], 10.0, atol=1e-4)


def test_carrier_bit():
    pos = place(a0=[30.0, 30.0], a2=[30.0, 35.0])
    _, _, carrier = cast(pos, [90.0, 0, 0, 0], carrying=[False, False, True, False])
    assert bool(carrier[0, 1]) and not bool(carrier[0, 0])


def test_shapes_and_vmap():
    p = CFG.agent.perception
    pos = jnp.tile(jnp.array(place(a0=[30.0, 30.0]), jnp.float32), (8, 1, 1))
    heading = jnp.zeros((8, 4))
    f = jax.jit(jax.vmap(lambda ps, h: cast_rays(ps, h, jnp.ones(4, bool), jnp.zeros(4, bool), TEAM,
                                                 ARENA.flag_stand, jnp.ones(2, bool), ARENA, R, p)))
    dist, cls, carrier = f(pos, heading)
    assert dist.shape == cls.shape == carrier.shape == (8, 4, p.ray_count)
    assert float(dist.max()) <= p.max_range_m and float(dist.min()) > 0.0