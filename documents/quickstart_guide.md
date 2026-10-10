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

## 5. How to Tune Rewards & Environment Hyperparameters

You can customize all reward weights, timeout limits, and grid sizes in **two easy ways**:

### Method A: Edit `configs/environment.yaml` (Recommended - No Code Changes Needed)
Simply open [`configs/environment.yaml`](file:///home/mrctrl/Workspace/projects/deco-swarm/configs/environment.yaml):

```yaml
episode:
  max_steps: 100             # Maximum decision steps before timeout (10 Hz)
  action_repeat: 6           # Physics substeps per agent decision (60 Hz / 6 = 10 Hz)

rewards:
  progress_weight: 0.01      # Reward per unit of distance closed towards goal
  time_penalty: -1.1         # Step urgency penalty
  collision_penalty: -0.5    # Penalty when bumping into walls
  exploration_weight: 0.1    # Bonus for discovering a new spatial cell
  cell_size: 120.0           # Grid cell size in pixels for spatial visit tracking
  goal_reward: 100.0         # Terminal reward for reaching the goal zone
  timeout_penalty: -1.0      # Penalty applied if episode times out
```

### Method B: In Python Code (`env.rewards` / `RewardConfig`)
In [`src/envs/single_agent_env.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/single_agent_env.py), all parameters are grouped in typed dataclasses at the top of the file:
```python
# Access directly on the environment instance:
env.rewards.goal_reward = 150.0
env.rewards.time_penalty = -0.5
env.max_steps = 200
```
Every step also returns a detailed dictionary breakdown in `info["agent_0"]["reward_breakdown"]` (`progress`, `time`, `collision`, `exploration`, `terminal`).

---

## 6. Catastrophic Failure Protection & Model Saving Rules

Deep reinforcement learning policies can occasionally experience sudden degradation or policy collapse due to gradient spikes or local minima. To ensure you never lose high-performing weights, [`train.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/train.py) implements three automatic protection rules:

```text
                                  TRAINING LOOP
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   RULE A: PEAK GUARD           RULE B: AUTO-ROLLBACK        RULE C: SAFE FINAL
  checkpoints/model_best.pt     Detects Sudden Collapse     checkpoints/model_final.pt
  Saved ONLY when a new         Restores peak weights &     Guaranteed to NEVER save a
  record success rate or        halves learning rate to     degraded policy over good
  fastest path is achieved!     escape loss cliff.          weights.
```

### Rule A: Peak Performance Guard (`model_best.pt` & `model_best_tier_{id}.pt`)
- The trainer continuously tracks historical peak success rates and fastest completion steps.
- Whenever the agent sets a new high-water mark, it saves:
  - `checkpoints/model_best.pt` (Global peak model)
  - `checkpoints/model_best_tier_0.pt` (Best model for Tier 0)
  - `checkpoints/model_best_tier_1.pt` (Best model for Tier 1)
- **Result:** Your peak weights are never overwritten by subsequent plateauing.

### Rule B: Automatic Collapse Detection & Rollback
- If an agent has demonstrated proficiency ($\ge 50\%$ success rate) but suddenly drops to $\le 15\%$ over a rolling window:
  1. Detects collapse: `[ALERT: CATASTROPHIC COLLAPSE DETECTED]`.
  2. Automatically **rolls back** neural network weights to `model_best.pt`.
  3. **Decays the learning rate by 50%** ($\text{lr} \gets \text{lr} \times 0.5$) to prevent destabilizing updates.
  4. Resets the corrupted sliding window so the agent recovers cleanly from its peak state.

### Rule C: Safe Final Model Guard (`model_final.pt`)
- When a training run completes, it evaluates whether the last step degraded compared to the peak:
  - If the last step degraded by $> 25\%$, the system **protects `model_final.pt` by saving peak weights instead**:
    ```text
    [SAFE FINAL RULE] Final step policy is degraded compared to peak!
    >>> Protecting model_final.pt: Saving peak weights as model_final.pt.
    ```
  - `model_latest.pt` always captures the raw final step regardless.

### Summary of Saved Model Files:
| File | Behavior | Purpose |
| :--- | :--- | :--- |
| `checkpoints/model_best.pt` | **Only updates when performance improves** | Guaranteed peak policy across the entire training history. |
| `checkpoints/model_final.pt` | **Protected final checkpoint** | Best policy ready for deployment/watching without risk of degradation. |
| `checkpoints/model_latest.pt` | **Latest raw frame** | Exact current state, useful if you want to inspect unconstrained progress. |

