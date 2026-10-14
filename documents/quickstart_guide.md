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

## 2. High-Performance C++ Core Setup (`deco_core_cpp`)

The computationally intensive spatial mathematics—including multi-segment raycasting, continuous sliding collisions, and ego-polar tracking—are compiled as a native C++17 shared module via `pybind11` and `Eigen3`.

### Compiling the C++ Core:
```bash
# From the repository root with venv activated:
mkdir -p cpp_core/build
cd cpp_core/build
cmake -DPython3_ROOT_DIR=$(readlink -f ../../venv) -DPython3_FIND_STRATEGY=LOCATION ..
make -j$(nproc)
cd ../..

# Link the compiled shared library so Python discovers it:
ln -sf $(readlink -f cpp_core/build/deco_core_cpp.*.so) src/deco_core_cpp.so
```

### Verifying C++ Module:
```bash
PYTHONPATH=src python3 -c "import deco_core_cpp; print('deco_core_cpp loaded successfully!')"
```
*(If the C++ module is not compiled, the engine gracefully falls back to the pure-Python implementation).*

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

## 6. Promotion-Based Checkpoints & Tier-Scoped Protection

To prevent policy regression and ensure peak performance is captured without noisy disk churn, [`train.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/train.py) implements promotion-based graduation snapshots and tier-isolated safeguards:

```text
                                  CURRICULUM FLOW
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
   1. TIER GRADUATION           2. TIER MILESTONES          3. ISOLATED ROLLBACK
   `model_graduated_tier_X.pt`  `model_best_tier_X.pt`      Only monitors intra-tier
   Saved immediately when a     Saved only on genuine       drops after proficiency;
   tier is mastered and the     high milestones (>=70% &    never triggers on fresh
   agent is promoted!           +10% improvements).         tier exploration!
```

### 1. Graduation Checkpoints on Promotion (`model_graduated_tier_{id}.pt`)
- When the agent masters a tier (e.g. achieves $\ge 85\%$ on Tier 0) and is promoted to Tier 1, the trainer immediately preserves:
  - `checkpoints/model_graduated_tier_0.pt`
  - `checkpoints/model_best_tier_0.pt`
  - `checkpoints/model_best.pt`
- **Result:** You have an immutable, permanent record of the exact model that conquered Tier 0, even as it moves on to learn complex obstacles.

### 2. Tier-Scoped Milestone Tracking (`model_best_tier_{id}.pt`)
- Each tier maintains its **own independent peak record**.
- To prevent saving on trivial micro-fluctuations, a new tier milestone is only saved when:
  - The agent has completed at least the evaluation window on that tier.
  - Success rate reaches **$\ge 70\%$**.
  - Success rate improves by at least **$+10\%$** over the previous tier peak.

### 3. Safe Intra-Tier Collapse Protection
- When entering a brand-new, harder tier (like Tier 1), a success rate of $0\%$ is normal during early exploration. The system **never** compares performance across different tiers.
- A rollback only triggers if:
  1. The agent is on the **same tier**.
  2. It had already proven $\ge 70\%$ proficiency on that specific tier.
  3. It has spent $\ge 40$ episodes on that tier, but performance suddenly plunged to $\le 20\%$.
  - In that rare failure event, it restores `model_best_tier_{curr_tier}.pt` and anneals the learning rate.

### Summary of Checkpoint Files:
| File | Behavior | Purpose |
| :--- | :--- | :--- |
| `checkpoints/model_graduated_tier_0.pt` | **Saved upon Tier 0 mastery** | Permanent snapshot of the model that conquered Tier 0 navigation. |
| `checkpoints/model_graduated_tier_1.pt` | **Saved upon Tier 1 mastery** | Permanent snapshot of the model that conquered Tier 1 obstacle avoidance. |
| `checkpoints/model_best_tier_{id}.pt` | **Highest milestone on that tier** | Best policy for that specific curriculum level. |
| `checkpoints/model_best.pt` | **Global peak model** | Highest overall policy across training history. |
| `checkpoints/model_final.pt` | **Run completion snapshot** | Model at the end of the training execution. |
| `checkpoints/model_latest.pt` | **Rolling latest checkpoint** | Captured periodically (every `--save-interval`). |

