# DECO-SWARM: Quickstart & Command Guide

A comprehensive, practical guide for training, evaluating, and customizing the DECO-SWARM reinforcement learning system.

---

## 1. Quick Environment Setup

Activate the Python virtual environment containing PyTorch (with CUDA support), PettingZoo, and Pygame:

```bash
# In the repository root:
source venv/bin/activate
```

Verify GPU acceleration (NVIDIA RTX 3070 Laptop GPU supported):
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 2. Testing & Visualizing (Play Mode)

[`play.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/play.py) is the interactive testbed. It supports two modes: **Human (Manual)** and **AI Watch**.

### A. Human Manual Control (`--mode human`)
Test map layouts, collision detection, and raycast sensors yourself using keyboard controls:

```bash
# Play on any map:
python3 play.py --mode human --map configs/maps/faz_1v0.1.yaml

# Test a complex Tier 2 maze:
python3 play.py --mode human --map configs/maps/faz_1v2.6.yaml
```

**Keyboard & Mouse Controls:**
- <kbd>W</kbd> / <kbd>S</kbd>: Move Forward / Backward
- <kbd>A</kbd> / <kbd>D</kbd>: Strafe Left / Right
- <kbd>Q</kbd> / <kbd>E</kbd> (or Left/Right Arrows): Rotate Heading
- <kbd>Space</kbd>: Fire trigger
- <kbd>C</kbd>: Toggle camera follow agent on/off
- <kbd>F1</kbd>: Toggle raycast beams visibility
- <kbd>F2</kbd>: Reset agent spawn position
- <kbd>Esc</kbd>: Exit
- **Right-Click + Drag**: Pan camera
- **Mouse Wheel**: Zoom in / out

---

### B. AI Policy Visualization (`--mode watch`)
Watch a trained neural network checkpoint navigate the arena in real time:

```bash
# Watch the latest trained model:
python3 play.py --mode watch --model-path checkpoints/model_latest.pt --map configs/maps/faz_1v0.1.yaml

# Test generalization on another map:
python3 play.py --mode watch --model-path checkpoints/model_latest.pt --map configs/maps/faz_1v1.1.yaml
```

**Parameters for `play.py`:**
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `human` | Choose `human` for keyboard play or `watch` for AI policy inference. |
| `--model-path` | `checkpoints/model_final.pt` | Path to the trained PyTorch checkpoint file (`.pt`). |
| `--map` | `configs/maps/faz_1v0.1.yaml` | Map YAML configuration file to load. |
| `--device` | `auto` | Device for inference: `auto`, `cuda`, or `cpu`. |

---

## 3. Training & Fine-Tuning (`train.py`)

[`train.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/train.py) runs high-throughput headless reinforcement learning using Recurrent PPO with Generalized Advantage Estimation (GAE).

### Scenario 1: Train a New Model from Scratch
```bash
python3 train.py \
  --total-timesteps 50000 \
  --rollout-steps 256 \
  --batch-seq-len 32 \
  --lr 0.0003 \
  --ent-coef 0.01 \
  --save-interval 5000 \
  --save-dir checkpoints
```

### Scenario 2: Fine-Tune an Existing Model
Resume training from an existing checkpoint to continue improving policy weights:
```bash
python3 train.py \
  --resume checkpoints/model_latest.pt \
  --total-timesteps 30000 \
  --lr 0.0001
```

### Scenario 3: Enforce and Lock a Specific Tier (e.g. `faz_1v0.*`)
Train exclusively on **Tier 0** (`faz_1v0.*`) maps with within-tier domain randomization, preventing premature promotion:
```bash
python3 train.py \
  --resume checkpoints/model_latest.pt \
  --tier 0 \
  --lock-tier \
  --total-timesteps 30000
```

### Scenario 4: Train on Tier 1 (Obstacles & Dead-Ends)
```bash
python3 train.py \
  --resume checkpoints/model_latest.pt \
  --tier 1 \
  --total-timesteps 50000
```

---

## 4. Training Parameters Reference (`train.py`)

| Parameter | Default | Explanation & Practical Advice |
| :--- | :--- | :--- |
| `--total-timesteps` | `50000` | Total environment interaction steps. 30k takes ~4 minutes on an RTX 3070. |
| `--resume` | `""` | Path to `.pt` checkpoint to load weights from. Omit to start fresh. |
| `--tier` | `-1` | Explicit curriculum tier: `0` (v0 open), `1` (v1 obstacles), `2` (v2 mazes). `-1` uses checkpoint tier. |
| `--lock-tier` | `False` | Disables auto-promotion/demotion; forces agent to stay on the chosen tier. |
| `--lr` | `3e-4` | Learning rate. Use `3e-4` for scratch, `1e-4` to `5e-5` for gentle fine-tuning. |
| `--rollout-steps` | `256` | Environment steps collected before each PPO update. |
| `--batch-seq-len` | `32` | Sequence length unrolled through the GRU recurrent core. |
| `--ent-coef` | `0.01` | Policy entropy bonus weight. Increase to `0.02` for more exploration; lower to `0.005` to stabilize. |
| `--clip-coef` | `0.2` | PPO surrogate clipping threshold ($\epsilon$). |
| `--gamma` | `0.99` | Future reward discount factor ($0.99 \approx 100$ decision steps horizon). |
| `--gae-lambda` | `0.95` | GAE bias-variance balance factor. |
| `--save-interval` | `10000` | Environment steps between periodic checkpoint snapshots. |
| `--save-dir` | `checkpoints` | Directory where `.pt` weights are stored. |

---

## 5. How to Tune Rewards & Goal Mechanics

To change how the agent is evaluated, open [`src/envs/single_agent_env.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/single_agent_env.py):

- **Progress Weight (Line ~202):** `dist_progress = (self.prev_dist_to_goal - curr_dist) * 0.05`
  - Increase `0.05` $\to$ `0.10` to make the agent prioritize distance closing more aggressively.
- **Urgency / Time Penalty (Line ~206):** `time_penalty = -0.01`
  - Make more negative (e.g., `-0.02`) to penalize long circuitous paths.
- **Collision Penalty (Line ~209):** `collision_penalty = -0.05 if collided_during_action else 0.0`
  - Increase (e.g., `-0.15`) if the agent is scraping walls too much.
- **Exploration Novelty Bonus (Line ~217):** `exploration_reward = 0.02`
  - Awarded when the agent enters a new $40\times 40$ px grid cell during the episode.
- **Terminal Goal Reward (Line ~223):** `terminal_reward = 10.0 if goal_reached else 0.0`
- **Episode Timeout (Line ~42):** `max_steps = 500` ($50$ seconds in simulation).
