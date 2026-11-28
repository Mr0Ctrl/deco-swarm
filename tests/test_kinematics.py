from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import load_all
from deco_swarm.sim.kinematics import integrate

CFG = load_all(Path(__file__).resolve().parents[1] / "configs")
BODY = CFG.agent.body
DT = 1.0 / CFG.simulation.physics.tick_hz
V = BODY.linear_speed_mps


def step(heading, action, alive=True):
    """Single agent at the origin, one tick."""
    pos = jnp.zeros((1, 2))
    return integrate(pos, jnp.array([heading]), jnp.array([action]), jnp.array([alive]), BODY, DT)


def test_forward_at_0_moves_plus_x():
    pos, _ = step(0.0, [1, 0, 0, 0])
    np.testing.assert_allclose(pos[0], [V * DT, 0.0], atol=1e-6)


def test_forward_at_90_moves_plus_y():
    pos, _ = step(90.0, [1, 0, 0, 0])
    np.testing.assert_allclose(pos[0], [0.0, V * DT], atol=1e-6)


def test_strafe_left_at_0_moves_plus_y():
    pos, _ = step(0.0, [0, 1, 0, 0])
    np.testing.assert_allclose(pos[0], [0.0, V * DT], atol=1e-6)


def test_diagonal_not_faster():
    pos, _ = step(0.0, [1, 1, 0, 0])
    np.testing.assert_allclose(jnp.linalg.norm(pos[0]), V * DT, rtol=1e-5)


def test_turn_left_is_ccw_and_wraps():
    _, h = step(359.0, [0, 0, 1, 0])
    np.testing.assert_allclose(h[0], (359.0 + BODY.angular_speed_dps * DT) % 360.0, atol=1e-4)


def test_dead_agent_does_not_move():
    pos, h = step(0.0, [1, 0, 1, 0], alive=False)
    np.testing.assert_allclose(pos[0], [0.0, 0.0])
    assert float(h[0]) == 0.0


def test_jit_and_vmap():
    B, N = 8, 4                                  # 8 envs, 4 agents
    pos = jnp.zeros((B, N, 2))
    heading = jnp.zeros((B, N))
    actions = jnp.ones((B, N, 4), jnp.int32)
    alive = jnp.ones((B, N), bool)
    f = jax.jit(jax.vmap(lambda p, h, a, al: integrate(p, h, a, al, BODY, DT)))
    p, h = f(pos, heading, actions, alive)
    assert p.shape == (B, N, 2) and h.shape == (B, N)