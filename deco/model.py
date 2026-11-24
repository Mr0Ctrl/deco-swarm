# deco/model.py
"""
Late-fusion recurrent actor-critic in Flax (port of legacy RecurrentActorCritic).

Structure (identical to the PyTorch model, weights are convertible 1:1):
  1. Spatial ray encoder (1D-CNN)  -> 128-d
  2. Ego scalar encoder (MLP)      -> 32-d
  3. Late fusion concatenation     -> 160-d
  4. Recurrent core (GRU cell)     -> 128-d
  5. Actor heads: movement logits (5), rotation logits (3)
  6. Critic head: state value V(s)

Layers use PyTorch's default initialization (U(-1/sqrt(fan_in), 1/sqrt(fan_in))) so training from scratch
starts from the same weight distribution as the original code.
"""
import math
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from deco.config import AgentConfig
from deco.env import NUM_MOVE_ACTIONS, NUM_ROTATE_ACTIONS


def torch_uniform(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)

    return init


def torch_dense(features: int, fan_in: int, name: str = None) -> nn.Dense:
    bound = 1.0 / math.sqrt(fan_in)
    return nn.Dense(features, kernel_init=torch_uniform(bound), bias_init=torch_uniform(bound), name=name)


def torch_conv1d(features: int, in_channels: int, kernel: int, stride: int, padding: int, name: str) -> nn.Conv:
    bound = 1.0 / math.sqrt(in_channels * kernel)
    return nn.Conv(
        features,
        kernel_size=(kernel,),
        strides=(stride,),
        padding=[(padding, padding)],
        kernel_init=torch_uniform(bound),
        bias_init=torch_uniform(bound),
        name=name,
    )


def adaptive_avg_pool1d(x: jax.Array, out_len: int) -> jax.Array:
    """Same bin edges as torch.nn.AdaptiveAvgPool1d. x: (B, L, C) -> (B, out_len, C)."""
    length = x.shape[1]
    bins = [x[:, (i * length) // out_len: -(-((i + 1) * length) // out_len)].mean(axis=1) for i in range(out_len)]
    return jnp.stack(bins, axis=1)


class SpatialRayEncoder(nn.Module):
    """1D-CNN over the panoramic rays, treated as a 1D image with 7 channels."""
    ray_channels: int = 7
    out_dim: int = 128

    @nn.compact
    def __call__(self, rays: jax.Array) -> jax.Array:
        # rays: (B, num_rays, channels)
        x = nn.relu(torch_conv1d(32, self.ray_channels, kernel=5, stride=2, padding=2, name="conv1")(rays))
        x = nn.relu(torch_conv1d(64, 32, kernel=3, stride=2, padding=1, name="conv2")(x))
        x = adaptive_avg_pool1d(x, 8)  # (B, 8, 64)
        x = jnp.swapaxes(x, 1, 2).reshape(x.shape[0], -1)  # channel-major flatten, like torch's Flatten
        return nn.relu(torch_dense(self.out_dim, 64 * 8, name="fc")(x))


class EgoScalarEncoder(nn.Module):
    """MLP over the 10 ego scalars (polar target, velocity, previous action, health, ammo)."""
    in_dim: int = 10
    out_dim: int = 32

    @nn.compact
    def __call__(self, ego: jax.Array) -> jax.Array:
        x = nn.relu(torch_dense(self.out_dim, self.in_dim, name="fc1")(ego))
        return nn.relu(torch_dense(self.out_dim, self.out_dim, name="fc2")(x))


class TorchGRUCell(nn.Module):
    """GRU cell with exactly torch.nn.GRUCell's equations and parameter layout (gates ordered r, z, n)."""
    in_dim: int
    features: int

    @nn.compact
    def __call__(self, h: jax.Array, x: jax.Array) -> jax.Array:
        bound = 1.0 / math.sqrt(self.features)
        dense = lambda name: nn.Dense(  # noqa: E731
            3 * self.features, kernel_init=torch_uniform(bound), bias_init=torch_uniform(bound), name=name
        )
        i_r, i_z, i_n = jnp.split(dense("input")(x), 3, axis=-1)
        h_r, h_z, h_n = jnp.split(dense("hidden")(h), 3, axis=-1)
        r = nn.sigmoid(i_r + h_r)
        z = nn.sigmoid(i_z + h_z)
        n = jnp.tanh(i_n + r * h_n)
        return (1.0 - z) * n + z * h


class RecurrentActorCritic(nn.Module):
    num_rays: int = 90
    ray_channels: int = 7
    ego_dim: int = 10
    spatial_emb_dim: int = 128
    ego_emb_dim: int = 32
    hidden_dim: int = 128

    @classmethod
    def from_agent_config(cls, agent: AgentConfig) -> "RecurrentActorCritic":
        return cls(num_rays=agent.ray_count)

    def setup(self):
        self.spatial_encoder = SpatialRayEncoder(ray_channels=self.ray_channels, out_dim=self.spatial_emb_dim)
        self.ego_encoder = EgoScalarEncoder(in_dim=self.ego_dim, out_dim=self.ego_emb_dim)
        self.gru = TorchGRUCell(in_dim=self.spatial_emb_dim + self.ego_emb_dim, features=self.hidden_dim)
        self.actor_fc = torch_dense(64, self.hidden_dim)
        self.move_head = torch_dense(NUM_MOVE_ACTIONS, 64)
        self.rot_head = torch_dense(NUM_ROTATE_ACTIONS, 64)
        self.critic_fc = torch_dense(64, self.hidden_dim)
        self.critic_out = torch_dense(1, 64)

    def encode(self, obs: jax.Array) -> jax.Array:
        """Splits the flat observation into rays + ego scalars, encodes both and concatenates (late fusion)."""
        n = self.num_rays * self.ray_channels
        rays = obs[:, :n].reshape(obs.shape[0], self.num_rays, self.ray_channels)
        ego = obs[:, n: n + self.ego_dim]
        return jnp.concatenate([self.spatial_encoder(rays), self.ego_encoder(ego)], axis=-1)

    def core(self, hidden: jax.Array, features: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Recurrent step + heads. Returns (next_hidden, move_logits, rot_logits, value)."""
        next_hidden = self.gru(hidden, features)
        actor = nn.relu(self.actor_fc(next_hidden))
        value = self.critic_out(nn.relu(self.critic_fc(next_hidden)))[..., 0]
        return next_hidden, self.move_head(actor), self.rot_head(actor), value

    def __call__(self, hidden: jax.Array, obs: jax.Array):
        return self.core(hidden, self.encode(obs))

    def initial_hidden(self, batch_size: int) -> jax.Array:
        return jnp.zeros((batch_size, self.hidden_dim), dtype=jnp.float32)


# ==============================================================================
# MULTI-DISCRETE ACTION DISTRIBUTION (movement x rotation)
# ==============================================================================

def sample_action(key, move_logits, rot_logits, deterministic: bool = False):
    """Returns (action (B, 2) int32, log_prob (B,))."""
    if deterministic:
        move = jnp.argmax(move_logits, axis=-1)
        rot = jnp.argmax(rot_logits, axis=-1)
    else:
        k_move, k_rot = jax.random.split(key)
        move = jax.random.categorical(k_move, move_logits)
        rot = jax.random.categorical(k_rot, rot_logits)
    action = jnp.stack([move, rot], axis=-1).astype(jnp.int32)
    log_prob, _ = log_prob_entropy(move_logits, rot_logits, action)
    return action, log_prob


def log_prob_entropy(move_logits, rot_logits, action):
    """Joint log-probability and entropy of the two independent categorical heads."""
    move_logp = jax.nn.log_softmax(move_logits)
    rot_logp = jax.nn.log_softmax(rot_logits)
    log_prob = (
        jnp.take_along_axis(move_logp, action[..., 0:1], axis=-1)[..., 0]
        + jnp.take_along_axis(rot_logp, action[..., 1:2], axis=-1)[..., 0]
    )
    entropy = -(jnp.exp(move_logp) * move_logp).sum(-1) - (jnp.exp(rot_logp) * rot_logp).sum(-1)
    return log_prob, entropy
