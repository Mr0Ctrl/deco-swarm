# deco/ppo.py
"""
Recurrent PPO in pure JAX (PureJaxRL style) - port of legacy training/ppo.py.

Same algorithm as the original trainer:
  - GAE(gamma, lambda) advantages, normalized over the whole rollout
  - GRU unrolled over sequence chunks of `batch_seq_len`, hidden state reset after episode ends
  - Clipped surrogate policy loss, clipped value loss, entropy bonus
  - Adam (eps=1e-5) with global gradient-norm clipping

Difference: experience is collected from `num_envs` environments in parallel, and both the rollout
and the update are single jitted functions (no Python in the inner loops).
With num_envs=1 and num_minibatches = rollout_steps / batch_seq_len it matches the original update schedule.
"""
from dataclasses import dataclass
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax

from deco.env import MapPool, NavEnv
from deco.model import RecurrentActorCritic, log_prob_entropy, sample_action


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 64
    rollout_steps: int = 256        # decision steps per environment per update
    batch_seq_len: int = 32         # GRU unroll length (truncated BPTT)
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4


class Transition(NamedTuple):
    obs: jax.Array       # (T, N, obs_dim)
    action: jax.Array    # (T, N, 2)
    reward: jax.Array    # (T, N)
    done: jax.Array      # (T, N)
    value: jax.Array     # (T, N)
    log_prob: jax.Array  # (T, N)
    hidden: jax.Array    # (T, N, H) recurrent state BEFORE the step


class Rollout(NamedTuple):
    transitions: Transition
    advantages: jax.Array
    returns: jax.Array


class EpisodeStats(NamedTuple):
    """Per-step episode-end markers, used on the host for the curriculum."""
    done: jax.Array
    success: jax.Array
    steps: jax.Array
    map_idx: jax.Array


class RunnerState(NamedTuple):
    params: Dict
    opt_state: optax.OptState
    env_state: object
    obs: jax.Array
    hidden: jax.Array
    key: jax.Array


def make_optimizer(lr: float, max_grad_norm: float) -> optax.GradientTransformation:
    return optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=1e-5),
    )


def get_learning_rate(opt_state) -> float:
    return float(opt_state[1].hyperparams["learning_rate"])


def set_learning_rate(opt_state, lr: float):
    old = opt_state[1].hyperparams["learning_rate"]
    opt_state[1].hyperparams["learning_rate"] = jnp.asarray(lr, dtype=old.dtype)
    return opt_state


class PPOTrainer:
    def __init__(self, env: NavEnv, model: RecurrentActorCritic, cfg: PPOConfig):
        if cfg.rollout_steps < cfg.batch_seq_len:
            raise ValueError("rollout_steps must be >= batch_seq_len")
        self.env = env
        self.model = model
        self.cfg = cfg

        self.num_chunks = cfg.rollout_steps // cfg.batch_seq_len
        num_seqs = self.num_chunks * cfg.num_envs
        self.num_minibatches = max(1, min(cfg.num_minibatches, num_seqs))
        self.minibatch_seqs = num_seqs // self.num_minibatches
        self.optimizer = make_optimizer(cfg.lr, cfg.max_grad_norm)

        self.collect = jax.jit(self._collect)
        self.train = jax.jit(self._train)
        self.reset_envs = jax.jit(self._reset_envs)

    # --- Setup ---
    def init_params(self, key):
        dummy_obs = jnp.zeros((1, self.env.obs_dim), dtype=jnp.float32)
        return self.model.init(key, self.model.initial_hidden(1), dummy_obs)

    def init_runner(self, key, params, pool: MapPool) -> RunnerState:
        k_reset, k_run = jax.random.split(key)
        env_state, obs, hidden = self.reset_envs(k_reset, pool)
        return RunnerState(params, self.optimizer.init(params), env_state, obs, hidden, k_run)

    def _reset_envs(self, key, pool: MapPool):
        keys = jax.random.split(key, self.cfg.num_envs)
        obs, env_state = jax.vmap(self.env.reset, in_axes=(0, None))(keys, pool)
        return env_state, obs, self.model.initial_hidden(self.cfg.num_envs)

    # --- Experience collection ---
    def _collect(self, runner: RunnerState, pool: MapPool) -> Tuple[RunnerState, Rollout, EpisodeStats]:
        cfg = self.cfg
        env_step = jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

        def step_fn(carry, _):
            env_state, obs, hidden, key = carry
            key, k_act, k_env = jax.random.split(key, 3)
            next_hidden, move_logits, rot_logits, value = self.model.apply(runner.params, hidden, obs)
            action, log_prob = sample_action(k_act, move_logits, rot_logits)
            next_obs, env_state, reward, done, info = env_step(
                jax.random.split(k_env, cfg.num_envs), env_state, action, pool
            )
            transition = Transition(obs, action, reward.astype(jnp.float32), done, value, log_prob, hidden)
            stats = EpisodeStats(done, info["is_success"], info["step_count"], info["map_idx"])
            next_hidden = jnp.where(done[:, None], 0.0, next_hidden)  # new episode -> fresh memory
            return (env_state, next_obs, next_hidden, key), (transition, stats)

        carry = (runner.env_state, runner.obs, runner.hidden, runner.key)
        (env_state, obs, hidden, key), (traj, stats) = jax.lax.scan(step_fn, carry, None, length=cfg.rollout_steps)

        _, _, _, last_value = self.model.apply(runner.params, hidden, obs)
        advantages = self._gae(traj.reward, traj.done, traj.value, last_value)
        rollout = Rollout(traj, advantages, advantages + traj.value)
        return runner._replace(env_state=env_state, obs=obs, hidden=hidden, key=key), rollout, stats

    def _gae(self, rewards, dones, values, last_value):
        gamma, lam = self.cfg.gamma, self.cfg.gae_lambda

        def body(carry, xs):
            gae, next_value = carry
            reward, done, value = xs
            next_non_terminal = 1.0 - done.astype(jnp.float32)
            delta = reward + gamma * next_value * next_non_terminal - value
            gae = delta + gamma * lam * next_non_terminal * gae
            return (gae, value), gae

        init = (jnp.zeros_like(last_value), last_value)
        _, advantages = jax.lax.scan(body, init, (rewards, dones, values), reverse=True)
        return advantages

    # --- Policy update ---
    def _loss(self, params, mb: Dict[str, jax.Array], init_hidden):
        cfg = self.cfg
        seq_len, batch = mb["obs"].shape[:2]

        # Encoders run on all frames in one batched pass; only the GRU is unrolled over time
        features = self.model.apply(
            params, mb["obs"].reshape(seq_len * batch, -1), method=RecurrentActorCritic.encode
        ).reshape(seq_len, batch, -1)
        done_prev = jnp.concatenate([jnp.zeros((1, batch)), mb["dones"][:-1].astype(jnp.float32)], axis=0)

        def unroll(hidden, xs):
            feat, prev_done = xs
            hidden = hidden * (1.0 - prev_done)[:, None]  # episode ended at t-1 -> reset memory
            hidden, move_logits, rot_logits, value = self.model.apply(
                params, hidden, feat, method=RecurrentActorCritic.core
            )
            return hidden, (move_logits, rot_logits, value)

        _, (move_logits, rot_logits, values) = jax.lax.scan(unroll, init_hidden, (features, done_prev))
        log_probs, entropy = log_prob_entropy(move_logits, rot_logits, mb["actions"])

        # Policy loss (PPO clipped objective)
        ratio = jnp.exp(log_probs - mb["old_log_probs"])
        adv = mb["advantages"]
        pg_loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv).mean()

        # Value loss (clipped value objective)
        v_clipped = mb["old_values"] + jnp.clip(values - mb["old_values"], -cfg.clip_coef, cfg.clip_coef)
        vf_loss = 0.5 * jnp.maximum((values - mb["returns"]) ** 2, (v_clipped - mb["returns"]) ** 2).mean()

        ent = entropy.mean()
        total = pg_loss + cfg.vf_coef * vf_loss - cfg.ent_coef * ent
        return total, {"policy_loss": pg_loss, "value_loss": vf_loss, "entropy": ent}

    def _train(self, runner: RunnerState, rollout: Rollout) -> Tuple[RunnerState, Dict[str, jax.Array]]:
        cfg = self.cfg
        seq_len, chunks, n = cfg.batch_seq_len, self.num_chunks, cfg.num_envs
        used = chunks * seq_len
        tr = rollout.transitions

        adv = rollout.advantages
        adv = (adv - adv.mean()) / (adv.std(ddof=1) + 1e-8)  # torch.std is unbiased

        def to_sequences(x):  # (T, N, ...) -> (seq_len, chunks * N, ...)
            x = x[:used].reshape((chunks, seq_len, n) + x.shape[2:])
            return jnp.swapaxes(x, 0, 1).reshape((seq_len, chunks * n) + x.shape[3:])

        batch = {
            "obs": to_sequences(tr.obs),
            "actions": to_sequences(tr.action),
            "dones": to_sequences(tr.done),
            "old_log_probs": to_sequences(tr.log_prob),
            "old_values": to_sequences(tr.value),
            "advantages": to_sequences(adv),
            "returns": to_sequences(rollout.returns),
        }
        # Each sequence starts from the hidden state stored at its first step
        init_hidden = tr.hidden[:used].reshape(chunks, seq_len, n, -1)[:, 0].reshape(chunks * n, -1)

        grad_fn = jax.value_and_grad(self._loss, has_aux=True)

        def epoch(carry, key):
            perm = jax.random.permutation(key, chunks * n)[: self.num_minibatches * self.minibatch_seqs]

            def minibatch(carry, idx):
                params, opt_state = carry
                mb = jax.tree.map(lambda x: x[:, idx], batch)
                (_, metrics), grads = grad_fn(params, mb, init_hidden[idx])
                updates, opt_state = self.optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), metrics

            return jax.lax.scan(minibatch, carry, perm.reshape(self.num_minibatches, self.minibatch_seqs))

        key, sub = jax.random.split(runner.key)
        (params, opt_state), metrics = jax.lax.scan(
            epoch, (runner.params, runner.opt_state), jax.random.split(sub, cfg.update_epochs)
        )
        metrics = jax.tree.map(jnp.mean, metrics)
        return runner._replace(params=params, opt_state=opt_state, key=key), metrics
