"""Unit tests for pure-JAX CTF rules, laser tagging, flag mechanics, and throughput."""

import time

import jax
import jax.numpy as jnp

from src.configs import load_arena, load_project_config
from src.sim import reset_world, step_world


def _step_jit(cfg):
    return jax.jit(
        lambda st, ar, act: step_world(st, ar, act, cfg.agent, cfg.physics, cfg.episode)
    )


def test_reset_and_laser_tag_respawn():
    cfg = load_project_config("configs")
    arena = load_arena("configs/maps/ctf_open.yaml")
    state = reset_world(arena, team_size=2)

    # Place Red[0] at (400, 400) facing 0 deg, Blue[0] at (550, 400) in front of it
    pos = state.pos.at[0].set(jnp.asarray([400.0, 400.0])).at[2].set(jnp.asarray([550.0, 400.0]))
    heading = state.heading_deg.at[0].set(0.0)
    state = state._replace(pos=pos, heading_deg=heading)

    actions = jnp.zeros((4, 3), dtype=jnp.int32).at[0, 2].set(1)  # Red[0] fires laser
    step_fn = _step_jit(cfg)
    next_st, (rays, ego), events, term, trunc = step_fn(state, arena, actions)

    assert rays.shape == (4, 90, 12)
    assert ego.shape == (4, 9)
    assert int(events.tagged_count[0]) == 1
    assert not bool(next_st.alive[2])
    assert float(next_st.respawn_timer[2]) > 0.0
    assert not bool(term) and not bool(trunc)

    # Advance 2.5 seconds (25 macro-steps) -> Blue[0] respawns at its base spawn
    noop = jnp.zeros((4, 3), dtype=jnp.int32)
    for _ in range(26):
        next_st, _, _, _, _ = step_fn(next_st, arena, noop)
    assert bool(next_st.alive[2])
    assert jnp.allclose(next_st.pos[2], arena.blue_spawns[0, :2], atol=1e-4)


def test_flag_pickup_drop_touch_return_and_scoring():
    cfg = load_project_config("configs")
    arena = load_arena("configs/maps/ctf_open.yaml")
    state = reset_world(arena, team_size=2)
    step_fn = _step_jit(cfg)
    noop = jnp.zeros((4, 3), dtype=jnp.int32)

    # 1. Red[0] stands on Blue flag -> picks it up
    state = state._replace(pos=state.pos.at[0].set(arena.blue_flag_pos))
    state, _, events, _, _ = step_fn(state, arena, noop)
    assert bool(events.picked_up[0])
    assert bool(state.carrying_flag[0])
    assert int(state.blue_flag_carrier) == 0
    assert not bool(state.blue_flag_home)

    # 2. Blue[0] tags Red[0] -> Red[0] drops Blue flag on the spot
    pos = state.pos.at[0].set(jnp.asarray([600.0, 400.0])).at[2].set(jnp.asarray([700.0, 400.0]))
    heading = state.heading_deg.at[2].set(180.0)
    state = state._replace(pos=pos, heading_deg=heading)
    fire_b0 = noop.at[2, 2].set(1)
    state, _, events, _, _ = step_fn(state, arena, fire_b0)
    assert bool(events.tagged_carrier[2])
    assert not bool(state.carrying_flag[0])
    assert int(state.blue_flag_carrier) == -1
    assert jnp.allclose(state.blue_flag_pos, jnp.asarray([600.0, 400.0]), atol=1e-4)

    # 3. Blue[1] touches dropped Blue flag -> instant return to Blue base
    state = state._replace(pos=state.pos.at[3].set(jnp.asarray([600.0, 400.0])))
    state, _, events, _, _ = step_fn(state, arena, noop)
    assert bool(events.returned[3])
    assert bool(state.blue_flag_home)
    assert jnp.allclose(state.blue_flag_pos, arena.blue_flag_pos, atol=1e-4)

    # 4. Red[1] picks up Blue flag and stands in Red base while Red flag is home -> scores!
    state = state._replace(pos=state.pos.at[1].set(arena.blue_flag_pos))
    state, _, _, _, _ = step_fn(state, arena, noop)
    assert int(state.blue_flag_carrier) == 1
    state = state._replace(pos=state.pos.at[1].set(arena.red_base[:2]))
    state, _, events, _, _ = step_fn(state, arena, noop)
    assert int(events.captured[0]) == 1
    assert int(state.scores[0]) == 1
    assert bool(state.blue_flag_home)
    assert not bool(state.carrying_flag[1])


def test_batched_vmap_gpu_throughput():
    cfg = load_project_config("configs")
    arena = load_arena("configs/maps/ctf_pillars.yaml")
    num_envs = 256
    team_size = 2

    states = jax.vmap(lambda _: reset_world(arena, team_size))(jnp.arange(num_envs))
    actions = jnp.ones((num_envs, 2 * team_size, 3), dtype=jnp.int32)

    batched_step = jax.jit(
        jax.vmap(
            lambda st, act: step_world(st, arena, act, cfg.agent, cfg.physics, cfg.episode),
            in_axes=(0, 0),
        )
    )
    # Warmup JIT compilation
    st, _, _, _, _ = batched_step(states, actions)
    jax.block_until_ready(st.pos)

    iters = 50
    t0 = time.perf_counter()
    for _ in range(iters):
        st, _, _, _, _ = batched_step(st, actions)
    jax.block_until_ready(st.pos)
    elapsed = time.perf_counter() - t0
    sps = (num_envs * iters) / elapsed
    print(f"\n[CTF JAX vmap] {num_envs} envs x {iters} steps = {sps:,.0f} RL steps/sec")
    assert sps >= 10_000.0
