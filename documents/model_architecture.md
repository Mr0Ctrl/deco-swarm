# DECO-SWARM: Neural Network Architecture & Model Exploration Guide

## 1. Overview & Architectural Motivation

In DECO-SWARM Phase 1, our agents must navigate 2D continuous environments under **Partial Observability (POMDP)**. The observation vector published by the environment contains **640 floating-point numbers**:
- **630 floats (Spatial Perception):** 90 raycasts spanning a 180° field-of-view, with 7 features per ray (`norm_distance`, `is_wall`, `is_friend`, `is_enemy`, `is_objective`, `rel_cos`, `rel_sin`).
- **10 floats (Internal / Ego State):** Relative target polar vector ($r, \Delta\theta$), linear and angular velocities ($v_x, v_y, \omega$), previous action vector $a_{t-1}$, and normalized health & ammo.

Because the input mixes ordered spatial geometry (adjacent rays) with non-spatial scalar state variables, feeding this vector into a monolithic flat MLP forces the network to learn both spatial proximity and scalar physics simultaneously.

To address this, we adopt a **Modular Multi-Stream (Late-Fusion)** design.

---

## 2. Current Architecture: Late-Fusion 1D-CNN + GRU (`RecurrentActorCritic`)

The current baseline is implemented in [`src/models/actor_critic.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/models/actor_critic.py).

```text
                  Observation Vector (640 floats)
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
 Rays (630: 90 rays x 7 channels)             Ego Scalars (10 floats)
           │                                           │
  Reshape to (Batch, 7, 90)                            │
           │                                           │
  Conv1D(7 -> 32, k=5, s=2)                            │
           │                                           │
  Conv1D(32 -> 64, k=3, s=2)                  Linear(10 -> 32) -> ReLU
           │                                           │
  AdaptiveAvgPool1D(8)                        Linear(32 -> 32) -> ReLU
           │                                           │
  Linear(512 -> 128) -> ReLU                           │
           │                                           │
    Spatial Embedding [128-d]                  Ego Embedding [32-d]
           │                                           │
           └─────────────────────┬─────────────────────┘
                                 ▼
                     Concatenation [160-d]
                                 │
                                 ▼
                    GRU Cell (hidden_dim = 128)
                    [POMDP Belief State h_t]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
      Actor Backbone                              Critic Backbone
  Linear(128 -> 64) -> ReLU                   Linear(128 -> 64) -> ReLU
           │                                           │
  ┌────────┴────────┐                         Linear(64 -> 1)
  ▼                 ▼                                  │
Move Head       Rotate Head                      State Value V(s)
(5 logits)      (3 logits)
```

### Module Breakdown & Rationale

| Module | Design Choice | Rationale & Trade-offs |
| :--- | :--- | :--- |
| **Spatial Stream** | `1D-CNN` ($k=5, k=3$) | Treats the 90 rays as a 1D panoramic scan. Exploits translation equivariance (detecting an opening or obstacle regardless of whether it appears on the left, center, or right) with very few trainable parameters. |
| **Ego Stream** | 2-Layer MLP | Directly projects non-spatial kinematic coordinates into a compatible latent subspace without convolutional assumptions. |
| **Fusion Layer** | Late-Fusion Concatenation | Preserves distinct feature representations until high-level geometric and target properties have been extracted. |
| **Recurrent Core** | `nn.GRUCell` ($128$-d) | Maintains temporal history ($h_t$). Gated Recurrent Units require fewer parameters and faster rollout updates than LSTMs, which is ideal for consumer laptop training. |
| **Policy Heads** | Multi-Discrete Heads | Decouples longitudinal/lateral translation ($5$ choices) from angular rotation ($3$ choices), allowing the agent to combine forward movement with turning smoothly. |
| **Critic Head** | Scalar Value Output | Provides baseline state-value estimates $V(h_t)$ needed for Temporal Difference (TD) error calculation and Generalized Advantage Estimation (GAE). |

---

## 3. Alternative Architectures to Explore in Future Experiments

Because RL performance is sensitive to model inductive biases, our codebase is structured to allow dropping in alternative backbones:

### Alternative A: Pure Spatial MLP (The Flat Baseline)
- **Concept:** Replace the Conv1D layers with dense linear layers: `Linear(630 -> 256) -> ReLU -> Linear(256 -> 128)`.
- **When to try:** To benchmark whether 1D convolutions provide a tangible sample-efficiency boost over dense linear representations on simple geometric maps (Tier 0).

### Alternative B: Long Short-Term Memory (LSTM Core)
- **Concept:** Replace `GRUCell` with `LSTMCell(160, 128)`.
- **When to try:** In Tier 2 mazes if the agent suffers from long-horizon forgetting in deep dead ends. LSTMs maintain a separate cell state $c_t$ with linear error carousel dynamics that can store memories across longer horizons.

### Alternative C: Spatial Self-Attention / Transformer Perceiver
- **Concept:** Treat the 90 rays as tokens: $(90, 7) \rightarrow$ Multi-Head Self-Attention block.
- **When to try:** If raycast resolutions increase or if we add dynamic entity tokens (friends, enemies, missiles) in Phases 3 and 4. Attention provides permutation invariance and dynamic relationship modeling across observed entities.

### Alternative D: Dueling Q-Network (Off-Policy TD Exploration)
- **Concept:** Instead of an on-policy Actor-Critic (PPO), use a Branching Dueling Q-Network (BDQ) where each action branch estimates:
  $$Q_i(s, a_i) = V(s) + \left( A_i(s, a_i) - \frac{1}{|A_i|} \sum_{a'} A_i(s, a') \right)$$
- **When to try:** When maximum sample efficiency is needed, allowing replay buffers to reuse old experience transitions.
