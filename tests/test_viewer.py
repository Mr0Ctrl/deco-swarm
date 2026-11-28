import os

os.environ["SDL_VIDEODRIVER"] = "dummy"          # no window needed

from pathlib import Path

import jax

from deco_swarm.config import compile_arena, load_all
from deco_swarm.envs import CTFEnv
from deco_swarm.rendering.viewer import Viewer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_draws_without_error():
    cfg = load_all(CONFIGS)
    env = CTFEnv(cfg, compile_arena(CONFIGS / "arenas" / "ctf_pillars.yaml", cfg.simulation.physics))
    viewer = Viewer(env, cfg.render)
    _, state = env.reset(jax.random.key(0))
    viewer.draw(state, selected=0, show_rays=True)
    viewer.draw(state, selected=3, show_rays=False)
    viewer.close()


def test_y_axis_points_up():
    cfg = load_all(CONFIGS)
    env = CTFEnv(cfg, compile_arena(CONFIGS / "arenas" / "ctf_open.yaml", cfg.simulation.physics))
    viewer = Viewer(env, cfg.render)
    _, y_low = viewer.to_screen((30.0, 10.0))
    _, y_high = viewer.to_screen((30.0, 30.0))
    assert y_high < y_low                       # larger world y -> higher on screen
    viewer.close()