from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from deco_swarm.config import compile_arena, load_all
from deco_swarm.sim.rules import apply_rules
from deco_swarm.sim.state import BLUE, CARRIED, HOME, RED, STRAY, initial_state

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CFG = load_all(CONFIGS)
ARENA = compile_arena(CONFIGS / "arenas" / "ctf_open.yaml", CFG.simulation.physics)
RULES = CFG.environment.rules
R = CFG.agent.body.radius_m
DT = 1.0 / CFG.simulation.physics.tick_hz
FAR = [[20.0, 5.0], [25.0, 5.0], [35.0, 5.0], [40.0, 5.0]]   # red_0, red_1, blue_0, blue_1
RED_STAND, BLUE_STAND = [6.0, 20.0], [54.0, 20.0]


def make(**where):
    """make(a0=[x, y]) -> initial state; other agents stay at FAR."""
    pos = [p[:] for p in FAR]
    for k, p in where.items():
        pos[int(k[1:])] = p
    return initial_state(jnp.array(pos), jnp.zeros(4), ARENA.flag_stand)


def tick(state):
    return apply_rules(state, ARENA, R, RULES, DT)


def carried_by(state, flag, agent):
    return state._replace(
        flag_state=state.flag_state.at[flag].set(CARRIED),
        flag_carrier=state.flag_carrier.at[flag].set(agent),
    )


def stray_blue(state, at):
    return state._replace(
        flag_state=state.flag_state.at[BLUE].set(STRAY),
        flag_pos=state.flag_pos.at[BLUE].set(jnp.array(at)),
    )


def test_pickup_enemy_flag_not_own():
    s, ev = tick(make(a0=[54.3, 20.0], a2=[54.0, 21.0]))      # red_0 and blue_0 at the blue flag
    assert int(s.flag_state[BLUE]) == CARRIED and int(s.flag_carrier[BLUE]) == 0
    assert ev.picked_up.tolist() == [True, False, False, False]


def test_closest_candidate_wins():
    s, _ = tick(make(a0=[54.8, 20.0], a1=[54.3, 20.0]))
    assert int(s.flag_carrier[BLUE]) == 1


def test_carried_flag_follows_carrier():
    s = carried_by(make(a0=[30.0, 30.0]), BLUE, 0)
    s, _ = tick(s)
    np.testing.assert_allclose(s.flag_pos[BLUE], [30.0, 30.0])


def test_capture_scores_and_resets_flag():
    s = carried_by(make(a0=[6.5, 20.0]), BLUE, 0)              # red_0 in red base with the blue flag
    s, ev = tick(s)
    assert s.score.tolist() == [1, 0]
    assert int(s.flag_state[BLUE]) == HOME and int(s.flag_carrier[BLUE]) == -1
    np.testing.assert_allclose(s.flag_pos[BLUE], BLUE_STAND)
    assert ev.captured.tolist() == [True, False, False, False]


def test_no_capture_when_own_flag_away():
    s = carried_by(make(a0=[6.5, 20.0], a2=[30.0, 30.0]), BLUE, 0)
    s = carried_by(s, RED, 2)                                    # blue_0 holds the red flag
    s, ev = tick(s)
    assert s.score.tolist() == [0, 0] and not bool(ev.captured.any())


def test_owner_returns_stray_flag():
    s, ev = tick(stray_blue(make(a2=[30.0, 20.3]), [30.0, 20.0]))
    assert int(s.flag_state[BLUE]) == HOME
    np.testing.assert_allclose(s.flag_pos[BLUE], BLUE_STAND)
    assert ev.returned.tolist() == [False, False, True, False]


def test_enemy_picks_up_stray_flag():
    s, _ = tick(stray_blue(make(a0=[30.0, 20.3]), [30.0, 20.0]))
    assert int(s.flag_state[BLUE]) == CARRIED and int(s.flag_carrier[BLUE]) == 0


def test_stray_auto_return():
    s = stray_blue(make(), [30.0, 20.0])
    s = s._replace(stray_timer=s.stray_timer.at[BLUE].set(RULES.flag_auto_return_s - DT / 2))
    s, _ = tick(s)
    assert int(s.flag_state[BLUE]) == HOME


def test_jit():
    f = jax.jit(lambda s: apply_rules(s, ARENA, R, RULES, DT))
    s, ev = f(make(a0=[54.3, 20.0]))
    assert int(s.flag_carrier[BLUE]) == 0