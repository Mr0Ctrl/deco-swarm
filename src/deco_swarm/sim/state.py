from typing import NamedTuple

import jax
import jax.numpy as jnp

HOME, CARRIED, STRAY = 0, 1, 2 # flag_state values
RED, BLUE = 0, 1

class EnvState(NamedTuple):
    pos: jax.Array # (N, 2)
    heading_deg: jax.Array # (N,)
    alive: jax.Array # (N,) bool
    respawn_timer: jax.Array # (N,)   Phase 3
    tag_cooldown: jax.Array # (N,)   Phase 3
    flag_pos: jax.Array # (2, 2) [team] -> xy
    flag_state: jax.Array # (2,) int  HOME / CARRIED / STRAY
    flag_carrier: jax.Array # (2,) int  agent index, -1 = none
    stray_timer: jax.Array # (2,)   seconds on the ground
    score: jax.Array # (2,) int
    step: jax.Array # () int  decision steps
    prev_action: jax.Array # (N, 4) int  last action (part of the observation)

class Events(NamedTuple):
    picked_up: jax.Array # (N,) bool
    returned: jax.Array # (N,) bool
    captured: jax.Array # (N,) bool

def agent_team(n):
    """[red_0..red_k, blue_0..blue_k] -> [0, .., 0, 1, .., 1]."""
    return (jnp.arange(n) >= n // 2).astype(jnp.int32)

def carrying_mask(flag_state, flag_carrier, n):
    """(N,) bool: which agents carry a flag."""
    idx = jnp.arange(n)
    return jnp.any((flag_state[:, None] == CARRIED) & (flag_carrier[:, None] == idx[None, :]), axis=0)

def initial_state(pos, heading_deg, flag_stand):
    n = pos.shape[0]
    return EnvState(
        pos=pos.astype(jnp.float32),
        heading_deg=heading_deg.astype(jnp.float32),
        alive=jnp.ones(n, bool),
        respawn_timer=jnp.zeros(n),
        tag_cooldown=jnp.zeros(n),
        flag_pos=flag_stand,
        flag_state=jnp.full(2, HOME, jnp.int32),
        flag_carrier=jnp.full(2, -1, jnp.int32),
        stray_timer=jnp.zeros(2),
        score=jnp.zeros(2, jnp.int32),
        step=jnp.array(0, jnp.int32),
        prev_action=jnp.zeros((n, 4), jnp.int32),
    )