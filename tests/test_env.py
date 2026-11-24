"""Behavioural tests of the JAX environment that need no legacy dependencies."""
import jax
import jax.numpy as jnp
import numpy as np

from deco.config import load_configs
from deco.env import MapPool, NavEnv
from deco.maps import MapBank

SMALL_MAP = "configs/maps/faz_1v0.1.yaml"   # 4 segments
BIG_MAP = "configs/maps/faz_1v2.6.yaml"     # 18 segments


def _env(paths):
    bank = MapBank(paths)
    env = NavEnv(load_configs(), bank)
    pool = MapPool(*map(jnp.asarray, bank.pool(paths)))
    return env, pool


def _rollout(env, pool, key, num_steps):
    obs, state = env.reset(key, pool)
    step = jax.jit(env.step)
    rewards = []
    for t in range(num_steps):
        k_act, k_env = jax.random.split(jax.random.fold_in(key, t))
        action = jax.random.randint(k_act, (2,), 0, jnp.array([5, 3]))
        obs, state, reward, done, _ = step(k_env, state, action, pool)
        rewards.append(float(reward))
    return np.asarray(obs), np.asarray(rewards)


def test_observation_shape_and_range():
    env, pool = _env([BIG_MAP])
    obs, _ = env.reset(jax.random.PRNGKey(0), pool)
    assert obs.shape == (640,)
    assert obs.dtype == jnp.float32
    assert float(jnp.min(obs)) >= -1.0 and float(jnp.max(obs)) <= 1.0


def test_same_seed_same_trajectory():
    env, pool = _env([SMALL_MAP, BIG_MAP])
    a = _rollout(env, pool, jax.random.PRNGKey(7), 40)
    b = _rollout(env, pool, jax.random.PRNGKey(7), 40)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_segment_padding_does_not_change_physics():
    """The small map padded to the big map's segment count must behave exactly like it does alone."""
    alone_env, alone_pool = _env([SMALL_MAP])
    bank_env, _ = _env([SMALL_MAP, BIG_MAP])
    small_only = MapPool(jnp.array([0, 0], dtype=jnp.int32), jnp.array(1, dtype=jnp.int32))
    a = _rollout(alone_env, alone_pool, jax.random.PRNGKey(3), 60)
    b = _rollout(bank_env, small_only, jax.random.PRNGKey(3), 60)
    np.testing.assert_allclose(a[0], b[0], atol=1e-5)
    np.testing.assert_allclose(a[1], b[1], atol=1e-5)


def test_truncation_and_auto_reset():
    env, pool = _env([BIG_MAP])
    obs0, state = env.reset(jax.random.PRNGKey(0), pool)
    step = jax.jit(env.step)
    idle = jnp.array([0, 0], dtype=jnp.int32)  # standing still never reaches the goal
    for t in range(env.cfg.max_steps):
        obs, state, reward, done, info = step(jax.random.PRNGKey(t), state, idle, pool)
        assert bool(done) == (t == env.cfg.max_steps - 1)
    assert bool(info["truncated"]) and not bool(info["is_success"])
    assert float(info["reward_terminal"]) == env.rew.timeout_penalty
    assert int(state.step_count) == 0  # auto-reset happened
    np.testing.assert_allclose(obs, obs0, atol=1e-5)  # jit vs eager float32 rounding


def test_vmap_over_envs():
    env, pool = _env([SMALL_MAP, BIG_MAP])
    keys = jax.random.split(jax.random.PRNGKey(0), 16)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(keys, pool)
    actions = jnp.tile(jnp.array([1, 2], dtype=jnp.int32), (16, 1))
    obs, state, reward, done, info = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))(keys, state, actions, pool)
    assert obs.shape == (16, 640) and reward.shape == (16,) and done.shape == (16,)
