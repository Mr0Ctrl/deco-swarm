from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.envs import CTFEnv

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
ENV = CTFEnv(CFG, compile_arena(CONFIGS / "arenas" / "ctf_pillars.yaml", CFG.simulation.physics))
B, T = 8, 200


@jax.jit
def rollout(key):
    k_reset, k_run = jax.random.split(key)
    _, state = jax.vmap(ENV.reset)(jax.random.split(k_reset, B))
    dims = jnp.array(ENV.action_dims)

    def one_step(state, k):
        k_act, k_step = jax.random.split(k)
        actions = jax.random.randint(k_act, (B, ENV.num_agents, 4), 0, dims)
        obs, state, reward, *_ = jax.vmap(ENV.step)(jax.random.split(k_step, B), state, actions)
        return state, (obs, reward)

    return jax.lax.scan(one_step, state, jax.random.split(k_run, T))


def leaves(tree):
    return [np.asarray(x) for x in jax.tree.leaves(tree)]


def test_same_seed_same_result():
    a, b = rollout(jax.random.key(42)), rollout(jax.random.key(42))
    for x, y in zip(leaves(a), leaves(b)):
        np.testing.assert_array_equal(x, y)              # bit-for-bit, not allclose


def test_different_seed_different_result():
    a, b = rollout(jax.random.key(1)), rollout(jax.random.key(2))
    assert not np.array_equal(np.asarray(a[0].pos), np.asarray(b[0].pos))