"""
Step-by-step comparison of the JAX environment against the legacy Python environment.

Both run the same action sequence; positions, observations, rewards and episode ends must match.
Runs in float64 (jax x64) so the comparison is exact up to rounding, not float32 noise.
Needs the legacy dependencies: gymnasium, pettingzoo, pygame.
"""
import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("pygame")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

try:  # moved out of jax.experimental in newer JAX versions
    from jax import enable_x64
except ImportError:
    from jax.experimental import enable_x64

from conftest import add_legacy_to_path  # noqa: E402
from deco.config import load_configs  # noqa: E402
from deco.env import NavEnv  # noqa: E402
from deco.maps import MapBank  # noqa: E402

MAPS = [
    "configs/maps/faz_1v0.1.yaml",  # open field: the greedy policy reaches the goal (termination + goal reward)
    "configs/maps/faz_1v1.3.yaml",  # walls + box: collisions and sliding
    "configs/maps/faz_1v2.6.yaml",  # maze: many wall contacts, usually times out (truncation)
]


def _legacy_env(map_path):
    add_legacy_to_path()
    from engine.map_loader import MapLoader
    from envs.single_agent_env import SingleAgentMazeEnv

    env = SingleAgentMazeEnv()
    env.current_map_path = map_path
    env.world.load_map(MapLoader.load(map_path))
    return env


def _choose_action(rng, polar_angle):
    """Mostly steer toward the goal (to exercise goal reaching), sometimes act randomly."""
    if rng.random() < 0.3:
        return np.array([rng.integers(5), rng.integers(3)])
    rot = 2 if polar_angle > 0.03 else (1 if polar_angle < -0.03 else 0)
    return np.array([1, rot])


@pytest.mark.parametrize("map_path", MAPS)
def test_episode_matches_legacy(map_path):
    legacy = _legacy_env(map_path)
    legacy_obs = legacy.reset()[0]["agent_0"]

    with enable_x64():
        env = NavEnv(load_configs(), MapBank([map_path]))
        obs, state = env.reset_to_map(0)
        step = jax.jit(env.step_no_reset)
        np.testing.assert_allclose(np.asarray(obs), legacy_obs, atol=1e-6)

        rng = np.random.default_rng(0)
        for t in range(env.cfg.max_steps):
            action = _choose_action(rng, legacy_obs[631])
            legacy_obs, l_rew, l_term, l_trunc, l_info = legacy.step({"agent_0": action})
            legacy_obs, l_rew = legacy_obs["agent_0"], l_rew["agent_0"]
            obs, state, reward, term, trunc, info = step(state, jnp.asarray(action))

            np.testing.assert_allclose(
                np.asarray(state.pos), legacy.world.agents["agent_0"].position, atol=1e-6, err_msg=f"pos @ step {t}"
            )
            np.testing.assert_allclose(np.asarray(obs), legacy_obs, atol=1e-6, err_msg=f"obs @ step {t}")
            assert abs(float(reward) - l_rew) < 1e-6, f"reward @ step {t}: {float(reward)} vs {l_rew}"
            assert bool(term) == l_term["agent_0"] and bool(trunc) == l_trunc["agent_0"], f"done @ step {t}"
            assert bool(info["collided"]) == l_info["agent_0"]["collided"], f"collision @ step {t}"
            if l_term["agent_0"] or l_trunc["agent_0"]:
                break
