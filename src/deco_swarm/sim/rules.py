import jax.numpy as jnp

from deco_swarm.config.arena import Arena
from deco_swarm.config.schema import RulesConfig
from .state import CARRIED, HOME, STRAY, EnvState, Events, agent_team, carrying_mask


def _closest(mask, dist):
    """mask, dist (2, N) -> (any (2,), index of the closest True (2,))."""
    return jnp.any(mask, axis=1), jnp.argmin(jnp.where(mask, dist, jnp.inf), axis=1)


def _who(flag_event, agent_idx, n):
    """flag-level event (2,) + responsible agent (2,) -> per-agent bool (N,)."""
    return jnp.any(flag_event[:, None] & (agent_idx[:, None] == jnp.arange(n)[None, :]), axis=0)


def apply_rules(state: EnvState, arena: Arena, radius, rules: RulesConfig, dt):
    """One physics tick of CTF rules. Both flags are updated together (no team goes first).

    Flag f belongs to team f. Order: follow carrier -> return -> pickup -> auto return -> capture.
    """
    n = state.pos.shape[0]
    team = agent_team(n)
    flags = jnp.arange(2)
    other = 1 - flags # team that steals flag f
    owner = team[None, :] == flags[:, None] # (2, N) agent i owns flag f
    carrying = carrying_mask(state.flag_state, state.flag_carrier, n)
    flag_state, flag_carrier = state.flag_state, state.flag_carrier

    # 1) carried flags follow their carrier
    carried = flag_state == CARRIED
    flag_pos = jnp.where(carried[:, None], state.pos[jnp.maximum(flag_carrier, 0)], state.flag_pos)

    dist = jnp.linalg.norm(state.pos[None, :, :] - flag_pos[:, None, :], axis=-1)   # (2, N)
    touch = (dist <= radius + arena.flag_radius) & state.alive[None, :]

    # 2) owner touches its stray flag -> back home (wins over an enemy pickup in the same tick)
    ret, returner = _closest(touch & owner & (flag_state == STRAY)[:, None], dist)
    flag_state = jnp.where(ret, HOME, flag_state)
    flag_carrier = jnp.where(ret, -1, flag_carrier)
    flag_pos = jnp.where(ret[:, None], arena.flag_stand, flag_pos)

    # 3) enemy without a flag touches a home/stray flag -> picks it up (closest wins)
    can_pick = touch & ~owner & ~carrying[None, :] & (flag_state != CARRIED)[:, None] & ~ret[:, None]
    pick, picker = _closest(can_pick, dist)
    flag_state = jnp.where(pick, CARRIED, flag_state)
    flag_carrier = jnp.where(pick, picker, flag_carrier)
    flag_pos = jnp.where(pick[:, None], state.pos[picker], flag_pos)

    # 4) stray too long -> back home
    stray = flag_state == STRAY
    stray_timer = jnp.where(stray, state.stray_timer + dt, 0.0)
    auto = stray & (stray_timer >= rules.flag_auto_return_s)
    flag_state = jnp.where(auto, HOME, flag_state)
    flag_pos = jnp.where(auto[:, None], arena.flag_stand, flag_pos)
    stray_timer = jnp.where(auto, 0.0, stray_timer)

    # 5) capture: carrier of flag f is inside its own base (team other[f]) and its own flag is home
    cidx = jnp.maximum(flag_carrier, 0)
    base = arena.base[other] # (2, 3)
    in_base = jnp.linalg.norm(state.pos[cidx] - base[:, :2], axis=-1) <= base[:, 2]
    own_home = flag_state[other] == HOME
    if rules.requires_own_flag_at_home: # static config -> plain Python if is fine
        cap = (flag_state == CARRIED) & in_base & own_home
    else:
        cap = (flag_state == CARRIED) & in_base
    events = Events(
        picked_up=_who(pick, picker, n),
        returned=_who(ret, returner, n),
        captured=_who(cap, cidx, n),
    )
    flag_state = jnp.where(cap, HOME, flag_state)
    flag_carrier = jnp.where(cap, -1, flag_carrier)
    flag_pos = jnp.where(cap[:, None], arena.flag_stand, flag_pos)
    score = state.score + cap[other].astype(jnp.int32)  # team t scores when flag other[t] is captured

    new_state = state._replace(flag_pos=flag_pos, flag_state=flag_state, flag_carrier=flag_carrier, stray_timer=stray_timer, score=score)
    return new_state, events