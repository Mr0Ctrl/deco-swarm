# src/models/actor_critic.py
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class SpatialRayEncoder(nn.Module):
    """
    1D-CNN feature extractor for the 90 raycasts (each with 7 channels).
    Treats the 180° panoramic rays as a 1D spatial image.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv_net = nn.Sequential(
            # Input: (batch, 7 channels, 90 rays)
            nn.Conv1d(in_channels=7, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # (batch, 32, 45)
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (batch, 64, 23)
            nn.AdaptiveAvgPool1d(8),
            # (batch, 64, 8) -> 512 flattened
            nn.Flatten(),
            nn.Linear(64 * 8, out_dim),
            nn.ReLU(),
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        # rays shape: (batch, 7, 90)
        return self.conv_net(rays)


class EgoScalarEncoder(nn.Module):
    """
    MLP encoder for the 10 internal ego scalars:
      - Polar target vector (2)
      - Odometry & velocities (3)
      - Previous action a_{t-1} (3)
      - Health & Ammo (2)
    """

    def __init__(self, in_dim: int = 10, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, ego: torch.Tensor) -> torch.Tensor:
        return self.mlp(ego)


class RecurrentActorCritic(nn.Module):
    """
    Late-Fusion Recurrent Actor-Critic Model for POMDP Navigation.
    
    Structure:
      1. Spatial Ray Encoder (1D-CNN) -> 128-d
      2. Ego Scalar Encoder (MLP)     -> 32-d
      3. Late Fusion Concatenation   -> 160-d
      4. Recurrent Core (GRUCell)    -> hidden_dim (128-d)
      5. Actor Head:
         - Movement logits (5: idle, fwd, bwd, left, right)
         - Rotation logits (3: hold, left, right)
      6. Critic Head:
         - State value V(s) (1)
    """

    def __init__(
        self,
        ray_channels: int = 7,
        num_rays: int = 90,
        ego_dim: int = 10,
        spatial_emb_dim: int = 128,
        ego_emb_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_rays = num_rays
        self.ray_channels = ray_channels
        self.ego_dim = ego_dim
        self.hidden_dim = hidden_dim

        # 1. Encoders
        self.spatial_encoder = SpatialRayEncoder(out_dim=spatial_emb_dim)
        self.ego_encoder = EgoScalarEncoder(in_dim=ego_dim, out_dim=ego_emb_dim)

        fusion_dim = spatial_emb_dim + ego_emb_dim

        # 2. Recurrent Memory Core (GRU Cell)
        self.gru_cell = nn.GRUCell(input_size=fusion_dim, hidden_size=hidden_dim)

        # 3. Policy (Actor) Heads
        self.actor_backbone = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
        )
        self.move_head = nn.Linear(64, 5)  # 5 movement choices
        self.rot_head = nn.Linear(64, 3)   # 3 rotation choices

        # 4. Value (Critic) Head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def initial_hidden_state(self, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns zero-initialized recurrent hidden state."""
        device = device or next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Splits raw 640-float observation into rays (630) and ego scalars (10),
        encodes both, and concatenates them.
        """
        batch_size = obs.shape[0]

        # Rays: first 630 floats -> reshape to (batch, 90 rays, 7 channels) -> transpose to (batch, 7, 90)
        rays_flat = obs[:, : self.num_rays * self.ray_channels]
        rays = rays_flat.view(batch_size, self.num_rays, self.ray_channels).transpose(1, 2)

        # Ego scalars: remaining 10 floats
        ego = obs[:, self.num_rays * self.ray_channels : self.num_rays * self.ray_channels + self.ego_dim]

        spatial_feat = self.spatial_encoder(rays)
        ego_feat = self.ego_encoder(ego)

        # Late fusion concatenation
        return torch.cat([spatial_feat, ego_feat], dim=-1)

    def forward_step(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes one recurrent step.
        Returns:
          move_logits, rot_logits, state_value, next_hidden
        """
        features = self.extract_features(obs)
        next_hidden = self.gru_cell(features, hidden)

        # Actor heads
        actor_feat = self.actor_backbone(next_hidden)
        move_logits = self.move_head(actor_feat)
        rot_logits = self.rot_head(actor_feat)

        # Critic head
        value = self.critic_head(next_hidden)

        return move_logits, rot_logits, value, next_hidden

    def get_action(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy for rollout execution.
        Returns:
          action: (batch, 2) [move_idx, rot_idx]
          log_prob: (batch,) total log probability
          value: (batch, 1) estimated state value
          next_hidden: (batch, hidden_dim) updated memory state
        """
        move_logits, rot_logits, value, next_hidden = self.forward_step(obs, hidden)

        move_dist = Categorical(logits=move_logits)
        rot_dist = Categorical(logits=rot_logits)

        if deterministic:
            move_act = torch.argmax(move_logits, dim=-1)
            rot_act = torch.argmax(rot_logits, dim=-1)
        else:
            move_act = move_dist.sample()
            rot_act = rot_dist.sample()

        log_prob = move_dist.log_prob(move_act) + rot_dist.log_prob(rot_act)
        action = torch.stack([move_act, rot_act], dim=-1)

        return action, log_prob, value, next_hidden

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        initial_hidden: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates a sequence of actions during PPO updates.
        Unrolls the GRU over time steps while respecting episode boundaries (dones).
        
        Args:
          obs_seq: (seq_len, batch_size, 640)
          initial_hidden: (batch_size, hidden_dim)
          actions_seq: (seq_len, batch_size, 2)
          dones_seq: (seq_len, batch_size) 1.0 if done, 0.0 otherwise
        
        Returns:
          log_probs: (seq_len, batch_size)
          entropy: (seq_len, batch_size)
          values: (seq_len, batch_size, 1)
        """
        seq_len, batch_size, _ = obs_seq.shape
        log_probs = []
        entropies = []
        values = []

        hidden = initial_hidden

        for t in range(seq_len):
            # If an episode terminated at t-1, reset hidden state for that batch element
            if t > 0:
                mask = (1.0 - dones_seq[t - 1]).unsqueeze(-1)
                hidden = hidden * mask

            obs_t = obs_seq[t]
            move_logits, rot_logits, val_t, hidden = self.forward_step(obs_t, hidden)

            move_dist = Categorical(logits=move_logits)
            rot_dist = Categorical(logits=rot_logits)

            move_act = actions_seq[t, :, 0]
            rot_act = actions_seq[t, :, 1]

            log_p = move_dist.log_prob(move_act) + rot_dist.log_prob(rot_act)
            ent = move_dist.entropy() + rot_dist.entropy()

            log_probs.append(log_p)
            entropies.append(ent)
            values.append(val_t)

        return torch.stack(log_probs), torch.stack(entropies), torch.stack(values)
