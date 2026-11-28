import jax
import jax.numpy as jnp

from deco_swarm.config import ConfigError
from deco_swarm.config.arena import Arena
from deco_swarm.config.schema import Configs
from deco_swarm.sim.collision import collide
from deco_swarm.sim.kinematics import decode_actions, integrate
from deco_swarm.sim.raycast import RAY_FEATURES, cast_rays, ray_features
from deco_swarm.sim.rules import apply_rules
from deco_swarm.sim.state import CARRIED, EnvState, Events, agent_team, carrying_mask, initial_state

NUM_EGO_TARGETS = 2 # own_base, enemy_flag
NUM_SELF = 10
NUM_PREV_ACTION = 4

class CTFEnv:
    """2-team CTF on one arena. Pure functions of (key, state, actions): jit/vmap from outside."""
    def __init__(self, cfg: Configs, arena: Arena):
        self.cfg = cfg
        self.arena = arena
        self.team_size = cfg.environment.episode.team_size
        self.num_agents = 2 * self.team_size
        self.agents = [f"red_{i}" for i in range(self.team_size)] + [f"blue_{i}" for i in range(self.team_size)]
        self.team = agent_team(self.num_agents)
        self.dt = 1.0 / cfg.simulation.physics.tick_hz
        p = cfg.agent.perception
        self.obs_size = p.ray_count * RAY_FEATURES + 3 * NUM_EGO_TARGETS + NUM_SELF + NUM_PREV_ACTION
        self.action_dims = (3, 3, 3, 2)
        if int(arena.spawn_mask.sum(axis=1).min()) < self.team_size:
            raise ConfigError(f"arena has fewer spawns than team_size={self.team_size}")

    # reset
    def reset(self, key):
        pos, heading = [], []
        for t, k in enumerate(jax.random.split(key, 2)):
            mask = self.arena.spawn_mask[t]
            score = jnp.where(mask, jax.random.uniform(k, mask.shape), 2.0)   # invalid spawns sort last
            idx = jnp.argsort(score)[: self.team_size]
            pos.append(self.arena.spawns[t, idx, :2])
            heading.append(self.arena.spawns[t, idx, 2])
        state = initial_state(jnp.concatenate(pos), jnp.concatenate(heading), self.arena.flag_stand)
        return self.get_obs(state), state
    
    # step
    def _tick(self, carry, _):
        state, actions, ev = carry
        body = self.cfg.agent.body
        pos, heading = integrate(state.pos, state.heading_deg, actions, state.alive, body, self.dt)
        pos = collide(pos, state.alive, body.radius_m, self.arena.segments, self.arena.segment_mask,
                      self.cfg.simulation.physics.collision_iterations)
        state = state._replace(pos=pos, heading_deg=heading)
        state, new_ev = apply_rules(state, self.arena, body.radius_m, self.cfg.environment.rules, self.dt)
        ev = jax.tree.map(jnp.logical_or, ev, new_ev) # an event in any tick counts
        return (state, actions, ev), None
    
    # step
    def step(self, key, state: EnvState, actions):
        """actions (N, 4) int -> obs, state, reward (N,), terminated (), truncated (), info."""
        n = self.num_agents
        no_ev = Events(*(jnp.zeros(n, bool) for _ in Events._fields))
        (state, _, ev), _ = jax.lax.scan(self._tick, (state, actions, no_ev), None, length=self.cfg.environment.episode.action_repeat)
        state = state._replace(step=state.step + 1, prev_action=actions.astype(jnp.int32))

        terminated = self._is_over(state)
        reward = self._reward(state, ev, terminated)
        info = {"events": ev, "score": state.score} # values before auto-reset

        # auto-reset: finished episodes continue from a fresh state
        _, reset_state = self.reset(key)
        state = jax.tree.map(lambda r, s: jnp.where(terminated, r, s), reset_state, state)
        truncated = jnp.array(False)
        return self.get_obs(state), state, reward, terminated, truncated, info
    
    def _is_over(self, state):
        ep = self.cfg.environment.episode
        over = state.step >= ep.max_steps
        if ep.score_limit is not None: # static config
            over = over | (jnp.max(state.score) >= ep.score_limit)
        return over
    
    def _reward(self, state, ev, terminated):
        rw = self.cfg.environment.rewards
        team, other = self.team, 1 - self.team
        team_cap = jnp.stack([jnp.any(ev.captured & (self.team == t)) for t in (0, 1)]).astype(jnp.float32)
        r = rw.capture_reward * (team_cap[team] - team_cap[other]) # zero-sum team reward
        r = r + rw.pickup_reward * ev.picked_up + rw.return_reward * ev.returned
        lead = jnp.sign(state.score[team] - state.score[other]).astype(jnp.float32)
        r = r + jnp.where(terminated, rw.win_reward * lead, 0.0) # win +, loss -, draw 0
        return r.astype(jnp.float32)
    
    # observation (SIM_CONTRACT v0)
    def get_obs(self, state: EnvState):
        cfg, p = self.cfg, self.cfg.agent.perception
        n, team, other = self.num_agents, self.team, 1 - self.team
        f32 = jnp.float32
        carrying = carrying_mask(state.flag_state, state.flag_carrier, n)

        # 1) rays: 3 groups (solid / flag / base), RAY_FEATURES values per ray
        hits = cast_rays(state.pos, state.heading_deg, state.alive, carrying, team, state.flag_pos, state.flag_state != CARRIED, self.arena, cfg.agent.body.radius_m, p)
        rays = ray_features(hits, p.max_range_m).reshape(n, -1)

        # 2) ego bearings to own base and enemy flag: [cos, sin, dist / diagonal], heading-relative
        targets = jnp.stack([self.arena.base[team, :2], state.flag_pos[other]], axis=1) # (N, 2, 2)
        rel = targets - state.pos[:, None, :]
        ang = jnp.arctan2(rel[..., 1], rel[..., 0]) - jnp.deg2rad(state.heading_deg)[:, None]
        d = jnp.linalg.norm(rel, axis=-1) / jnp.linalg.norm(self.arena.size)
        ego = jnp.stack([jnp.cos(ang), jnp.sin(ang), d], axis=-1).reshape(n, -1)

        # 3) self state
        ep = cfg.environment.episode
        selfs = jnp.concatenate([
            carrying[:, None].astype(f32),
            jax.nn.one_hot(state.flag_state[team], 3), # own flag: home / carried / stray
            jax.nn.one_hot(state.flag_state[other], 3), # enemy flag
            (~state.alive)[:, None].astype(f32), # respawning
            (state.tag_cooldown / cfg.agent.tagger.cooldown_s)[:, None],
            jnp.full((n, 1), 1.0 - state.step / ep.max_steps), # time left
        ], axis=-1)

        fwd, left, turn, fire = decode_actions(state.prev_action)
        prev = jnp.stack([fwd, left, turn, fire.astype(f32)], axis=-1)
        return jnp.concatenate([rays, ego, selfs, prev], axis=-1).astype(f32)