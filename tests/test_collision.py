from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.sim.collision import _closest_points, collide, push_out_of_walls, separate_agents
from deco_swarm.sim.kinematics import integrate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
BODY = CFG.agent.body
PHYS = CFG.simulation.physics
R = BODY.radius_m
DT = 1.0 / PHYS.tick_hz

FLOOR = jnp.array([[0.0, 0.0, 10.0, 0.0]])        # wall along y = 0
WALL_X0 = jnp.array([[0.0, 0.0, 0.0, 10.0]])      # wall along x = 0
ON = jnp.array([True])


def one(p):
    return jnp.array([p], jnp.float32), jnp.array([True])


def test_perpendicular_stops_at_radius():
    pos, alive = one([5.0, 0.3])
    out = push_out_of_walls(pos, alive, R, FLOOR, ON)
    np.testing.assert_allclose(out[0], [5.0, R], atol=1e-6)


def test_slides_along_wall():
    pos, alive = one([5.0, R])
    heading = jnp.array([315.0])                    # down-right, into the floor
    act = jnp.array([[1, 0, 0, 0]])
    for _ in range(30):
        pos, heading = integrate(pos, heading, act, alive, BODY, DT)
        pos = push_out_of_walls(pos, alive, R, FLOOR, ON)
    assert float(pos[0, 0]) > 5.0 + 1.0             # moved along x
    np.testing.assert_allclose(pos[0, 1], R, atol=1e-5)


def test_inner_corner():
    segs = jnp.concatenate([FLOOR, WALL_X0])
    pos, alive = one([0.3, 0.2])
    out = collide(pos, alive, R, segs, jnp.array([True, True]), PHYS.collision_iterations)
    assert float(out[0, 0]) >= R - 1e-5 and float(out[0, 1]) >= R - 1e-5


def test_masked_segment_ignored():
    pos, alive = one([5.0, 0.3])
    out = push_out_of_walls(pos, alive, R, FLOOR, jnp.array([False]))
    np.testing.assert_allclose(out, pos)


def test_two_agents_separate_symmetric():
    pos = jnp.array([[5.0, 5.0], [5.4, 5.0]])
    out = separate_agents(pos, jnp.array([True, True]), R)
    np.testing.assert_allclose(jnp.linalg.norm(out[0] - out[1]), 2 * R, atol=1e-5)
    np.testing.assert_allclose(out.mean(0), pos.mean(0), atol=1e-6)


def test_coincident_agents_separate():
    pos = jnp.array([[5.0, 5.0], [5.0, 5.0]])
    out = separate_agents(pos, jnp.array([True, True]), R)
    np.testing.assert_allclose(jnp.linalg.norm(out[0] - out[1]), 2 * R, atol=1e-5)


def test_dead_agent_ignored():
    pos = jnp.array([[5.0, 5.0], [5.4, 5.0]])
    alive = jnp.array([True, False])
    out = separate_agents(pos, alive, R)
    np.testing.assert_allclose(out, pos)
    out = push_out_of_walls(jnp.array([[5.0, 0.3]]), jnp.array([False]), R, FLOOR, ON)
    np.testing.assert_allclose(out, [[5.0, 0.3]])


def test_random_rollout_stays_valid():
    arena = compile_arena(CONFIGS / "arenas" / "ctf_pillars.yaml", PHYS)
    n = 4
    pos = jnp.concatenate([arena.spawns[0, :2, :2], arena.spawns[1, :2, :2]])
    heading = jnp.concatenate([arena.spawns[0, :2, 2], arena.spawns[1, :2, 2]])
    alive = jnp.ones(n, bool)

    def tick(carry, key):
        pos, heading = carry
        act = jax.random.randint(key, (n, 4), 0, jnp.array([3, 3, 3, 2]))
        pos, heading = integrate(pos, heading, act, alive, BODY, DT)
        pos = collide(pos, alive, R, arena.segments, arena.segment_mask, PHYS.collision_iterations)
        return (pos, heading), pos

    keys = jax.random.split(jax.random.key(0), 1000)
    _, traj = jax.jit(lambda c, k: jax.lax.scan(tick, c, k))((pos, heading), keys)

    W, H = arena.size
    assert float(traj[..., 0].min()) >= R - 1e-3 and float(traj[..., 0].max()) <= W - R + 1e-3
    assert float(traj[..., 1].min()) >= R - 1e-3 and float(traj[..., 1].max()) <= H - R + 1e-3
    flat = traj.reshape(-1, 2)
    d = jnp.linalg.norm(flat[:, None, :] - _closest_points(flat, arena.segments), axis=-1)
    d = jnp.where(arena.segment_mask[None, :], d, jnp.inf)
    assert float(d.min()) >= R - 1e-3


def test_no_tunneling():
    assert BODY.linear_speed_mps / PHYS.tick_hz < R