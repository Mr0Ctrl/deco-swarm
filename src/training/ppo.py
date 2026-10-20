# src/training/ppo.py
import os
import time
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from models.actor_critic import RecurrentActorCritic
from envs.single_agent_env import SingleAgentMazeEnv
from envs.vec_env import VectorMazeEnv
from envs.curriculum import MapCurriculum

# Disable distribution validation assertions for high-throughput training
torch.distributions.Distribution.set_default_validate_args(False)


class RolloutBuffer:
    """
    Stores sequential trajectories for Recurrent PPO with GAE calculation.
    Supports batched environments (num_steps, num_envs, ...).
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.reset()

    def reset(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs, self.obs_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs, self.act_dim), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.hiddens = torch.zeros((self.num_steps, self.num_envs, self.hidden_dim), dtype=torch.float32, device=self.device)

        self.advantages = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((self.num_steps, self.num_envs), dtype=torch.float32, device=self.device)

        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,       # (num_envs, obs_dim)
        action: torch.Tensor,    # (num_envs, act_dim)
        reward: torch.Tensor,    # (num_envs,)
        done: torch.Tensor,      # (num_envs,)
        value: torch.Tensor,     # (num_envs,)
        log_prob: torch.Tensor,  # (num_envs,)
        hidden: torch.Tensor,    # (num_envs, hidden_dim)
    ):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.hiddens[self.ptr] = hidden

        self.ptr += 1

    def compute_gae(self, last_val: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Computes Generalized Advantage Estimation (GAE) across all parallel environments."""
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values


class PPOTrainer:
    """
    Recurrent PPO Trainer supporting both SingleAgentMazeEnv and VectorMazeEnv (parallel maps).
    """

    def __init__(
        self,
        env: Union[SingleAgentMazeEnv, VectorMazeEnv],
        model: RecurrentActorCritic,
        device: torch.device,
        lr: float = 3e-4,
        rollout_steps: int = 256,
        batch_seq_len: int = 32,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
    ):
        self.env = env
        self.model = model
        self.device = device

        self.is_vector_env = isinstance(env, VectorMazeEnv)
        self.num_envs = env.num_envs if self.is_vector_env else 1

        # rollout_steps is the total steps per environment per update
        self.num_steps = rollout_steps
        self.batch_seq_len = batch_seq_len
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.optimizer = Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            num_steps=self.num_steps,
            num_envs=self.num_envs,
            obs_dim=640,
            act_dim=2,
            hidden_dim=model.hidden_dim,
            device=device,
        )

        # Episode logging
        self.global_step = 0
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []

    def collect_rollouts(
        self,
        current_obs: torch.Tensor,     # (num_envs, 640)
        current_hidden: torch.Tensor,  # (num_envs, hidden_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes policy across all N environments in parallel and stores experiences.
        """
        self.buffer.reset()
        obs = current_obs
        hidden = current_hidden

        for _ in range(self.num_steps):
            self.global_step += self.num_envs

            # Batched action selection on GPU
            with torch.no_grad():
                action, log_prob, val, next_hidden = self.model.get_action(obs, hidden)

            action_np = action.cpu().numpy()
            val_tensor = val.squeeze(-1)  # (num_envs,)

            if self.is_vector_env:
                next_obs_np, rewards_np, dones_np, _ = self.env.step(action_np)

                reward_tensor = torch.tensor(rewards_np, dtype=torch.float32, device=self.device)
                done_tensor = torch.tensor(dones_np, dtype=torch.float32, device=self.device)
                next_obs_tensor = torch.tensor(next_obs_np, dtype=torch.float32, device=self.device)

                # Reset hidden state for sub-environments that finished
                if dones_np.any():
                    done_mask = (1.0 - done_tensor).unsqueeze(-1)
                    hidden = next_hidden * done_mask
                else:
                    hidden = next_hidden

                next_obs = next_obs_tensor
            else:
                # Single environment fallback
                next_obs_dict, rewards_dict, terms_dict, truncs_dict, _ = self.env.step({"agent_0": action_np[0]})
                reward = rewards_dict["agent_0"]
                done = terms_dict["agent_0"] or truncs_dict["agent_0"]

                reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
                done_tensor = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

                if done:
                    reset_obs_dict, _ = self.env.reset()
                    next_obs = torch.tensor(reset_obs_dict["agent_0"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    hidden = self.model.initial_hidden_state(batch_size=1, device=self.device)
                else:
                    next_obs = torch.tensor(next_obs_dict["agent_0"], dtype=torch.float32, device=self.device).unsqueeze(0)
                    hidden = next_hidden

            # Store in rollout buffer
            self.buffer.add(
                obs=obs,
                action=action,
                reward=reward_tensor,
                done=done_tensor,
                value=val_tensor,
                log_prob=log_prob,
                hidden=hidden,
            )

            obs = next_obs

        # Bootstrap value for final step to compute GAE
        with torch.no_grad():
            _, _, last_val, _ = self.model.forward_step(obs, hidden)
            last_val_tensor = last_val.squeeze(-1)

        self.buffer.compute_gae(last_val=last_val_tensor, gamma=self.gamma, gae_lambda=self.gae_lambda)

        return obs, hidden

    def train_epoch(self) -> Dict[str, float]:
        """
        Executes PPO updates unrolling through sequence chunks across all parallel environments.
        """
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Chunks: (num_steps // batch_seq_len, num_envs)
        num_seqs_per_env = self.num_steps // self.batch_seq_len
        total_chunks = num_seqs_per_env * self.num_envs

        policy_losses, value_losses, entropy_losses = [], [], []

        for _ in range(self.update_epochs):
            # Create all (start_step, env_idx) chunk pairs and shuffle
            chunk_pairs = [
                (s * self.batch_seq_len, e)
                for s in range(num_seqs_per_env)
                for e in range(self.num_envs)
            ]
            np.random.shuffle(chunk_pairs)

            # Mini-batch chunk iteration
            for start_t, env_idx in chunk_pairs:
                end_t = start_t + self.batch_seq_len

                chunk_obs = self.buffer.obs[start_t:end_t, env_idx].unsqueeze(1)  # (seq_len, 1, 640)
                chunk_actions = self.buffer.actions[start_t:end_t, env_idx].unsqueeze(1)  # (seq_len, 1, 2)
                chunk_dones = self.buffer.dones[start_t:end_t, env_idx].unsqueeze(1)  # (seq_len, 1)
                chunk_old_log_probs = self.buffer.log_probs[start_t:end_t, env_idx]  # (seq_len,)
                chunk_old_values = self.buffer.values[start_t:end_t, env_idx]  # (seq_len,)
                chunk_advs = advantages[start_t:end_t, env_idx]  # (seq_len,)
                chunk_returns = self.buffer.returns[start_t:end_t, env_idx]  # (seq_len,)

                initial_hidden = self.buffer.hiddens[start_t, env_idx].unsqueeze(0)  # (1, hidden_dim)

                # Unroll model over sequence chunk
                new_log_probs, entropy, new_values = self.model.evaluate_actions(
                    chunk_obs, initial_hidden, chunk_actions, chunk_dones
                )

                new_log_probs = new_log_probs.squeeze(1)
                new_values = new_values.squeeze(1).squeeze(-1)
                entropy = entropy.squeeze(1)

                # Ratio: r(theta) = exp(new_log_p - old_log_p)
                log_ratio = new_log_probs - chunk_old_log_probs
                ratio = torch.exp(log_ratio)

                # Policy Loss (PPO Clipped Objective)
                surr1 = ratio * chunk_advs
                surr2 = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * chunk_advs
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value Loss (Clipped Value Objective)
                v_clipped = chunk_old_values + torch.clamp(
                    new_values - chunk_old_values, -self.clip_coef, self.clip_coef
                )
                vf_loss_unclipped = (new_values - chunk_returns) ** 2
                vf_loss_clipped = (v_clipped - chunk_returns) ** 2
                vf_loss = 0.5 * torch.max(vf_loss_unclipped, vf_loss_clipped).mean()

                # Entropy Loss
                ent_loss = entropy.mean()

                total_loss = pg_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss

                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(pg_loss.item())
                value_losses.append(vf_loss.item())
                entropy_losses.append(ent_loss.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_losses)),
        }
