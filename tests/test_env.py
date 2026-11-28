from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.envs import CTFEnv
from deco_swarm.sim.state import BLUE, CARRIED

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
ARENA = compile_arena(CONFIGS / "arenas" / "ctf_pillars.yaml", CFG.simulation.physics)
ENV = CTFEnv(CFG, ARENA)
N = ENV.num_agents
NOOP = jnp.zeros((N, 4), jnp.int32)


def random_actions(key):
    return jax.random.randint(key, (N, 4), 0, jnp.array(ENV.action_dims))


def test_shapes():
    obs, state = ENV.reset(jax.random.key(0))
    assert obs.shape == (N, ENV.obs_size) == (N, 916)
    obs, state, reward, term, trunc, info = ENV.step(jax.random.key(1), state, NOOP)
    assert obs.shape == (N, 916) and reward.shape == (N,)
    assert term.shape == () and trunc.shape == ()


def test_obs_range_after_random_play():
    step = jax.jit(ENV.step)
    _, state = ENV.reset(jax.random.key(0))
    for k in jax.random.split(jax.random.key(1), 50):
        k1, k2 = jax.random.split(k)
        obs, state, *_ = step(k1, state, random_actions(k2))
    assert bool(jnp.all(jnp.isfinite(obs)))
    assert float(obs.min()) >= -1.0 - 1e-5 and float(obs.max()) <= 1.0 + 1e-5


def test_symmetric_start_gives_identical_obs():
    _, state = ENV.reset(jax.random.key(0))
    W, H = ARENA.size
    red_pos = ARENA.spawns[0, :2, :2]
    red_h = ARENA.spawns[0, :2, 2]
    blue_pos = jnp.stack([W - red_pos[:, 0], H - red_pos[:, 1]], axis=-1)     # rotate_180
    blue_h = jnp.mod(red_h + 180.0, 360.0)
    state = state._replace(pos=jnp.concatenate([red_pos, blue_pos]),
                           heading_deg=jnp.concatenate([red_h, blue_h]))
    obs = ENV.get_obs(state)
    np.testing.assert_allclose(obs[0], obs[2], atol=1e-4)     # red_0 vs blue_0
    np.testing.assert_allclose(obs[1], obs[3], atol=1e-4)


def test_time_limit_terminates_and_resets():
    _, state = ENV.reset(jax.random.key(0))
    state = state._replace(step=jnp.array(CFG.environment.episode.max_steps - 1),
                           score=jnp.array([2, 1]))
    _, new_state, reward, term, _, info = ENV.step(jax.random.key(1), state, NOOP)
    assert bool(term)
    assert int(new_state.step) == 0 and new_state.score.tolist() == [0, 0]
    assert info["score"].tolist() == [2, 1]
    assert float(reward[0]) > 0 > float(reward[2])             # red won


def test_capture_reward_is_zero_sum():
    _, state = ENV.reset(jax.random.key(0))
    pos = state.pos.at[0].set(ARENA.base[0, :2] + jnp.array([0.5, 0.0]))   # red_0 in red base
    state = state._replace(pos=pos,
                           flag_state=state.flag_state.at[BLUE].set(CARRIED),
                           flag_carrier=state.flag_carrier.at[BLUE].set(0))
    _, new_state, reward, term, _, info = ENV.step(jax.random.key(1), state, NOOP)
    assert info["score"].tolist() == [1, 0]
    assert bool(info["events"].captured[0])
    red, blue = reward[:N // 2], reward[N // 2:]
    assert bool(jnp.all(red > 0)) and bool(jnp.all(blue < 0))


def test_jit_vmap_many_envs():
    B = 64
    keys = jax.random.split(jax.random.key(0), B)
    obs, state = jax.vmap(ENV.reset)(keys)
    actions = jax.vmap(random_actions)(keys)
    obs, state, reward, term, trunc, info = jax.jit(jax.vmap(ENV.step))(keys, state, actions)
    assert obs.shape == (B, N, 916) and reward.shape == (B, N) and term.shape == (B,)