# DECO-SWARM

> **Decentralized Multi-Agent Reinforcement Learning (MARL) in 2D Continuous Tactical Arenas**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.6+CUDA](https://img.shields.io/badge/PyTorch-2.6%2BCUDA-EE4C2C.svg)](https://pytorch.org/)
[![PettingZoo](https://img.shields.io/badge/PettingZoo-1.27-brightgreen.svg)](https://pettingzoo.farama.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.3-orange.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DECO-SWARM** is an AI-first reinforcement learning platform engineered to train autonomous swarms of agents in complex 2D tactical shooter arenas on consumer-grade laptop hardware (e.g., NVIDIA RTX 3070). 

Agents operate in continuous kinematics without global coordinate awareness. Through ego-centric raycast perception and recurrent memory, they autonomously discover navigation and squad doctrines (**emergent behaviors** such as crossfire, tactical room clearance, and flanking).

---

## Key Features & Innovations

- **Decoupled Headless Simulation**: The physics engine ([`src/engine/`](src/engine/)) is 100% independent of graphics, sustaining high simulation throughput (~100-200 FPS on GPU) during headless training while supporting 60 FPS Pygame visualization on demand.
- **Continuous Kinematics & Collision**: Sub-pixel continuous motion with line-segment and box obstacle resolution, avoiding discrete grid artifacts.
- **Ego-Centric POMDP Observation (640 Floats)**:
  - **90 Raycasts $\times$ 7 Channels ($630$ floats)**: 180° field-of-view panoramic depth, obstacle types, and relative orientations.
  - **Ego-Centric Target ($2$ floats)**: Polar distance and heading delta to current objective.
  - **Odometry & Status ($8$ floats)**: Linear/angular velocities, previous action $a_{t-1}$, health, and ammunition.
- **Late-Fusion Recurrent Actor-Critic (`RecurrentActorCritic`)**:
  - **1D-CNN Spatial Encoder**: Extracts corridor and opening geometries from panoramic rays.
  - **MLP Ego Encoder**: Projects target vectors and kinematic vitals.
  - **GRU Cell Core (128-d)**: Maintains a persistent **belief state** ($h_t$) to solve Partial Observability (POMDP) and eliminate spinning deadlocks.
  - **Multi-Discrete Actor Head**: Decouples movement ($5$ choices) from rotation ($3$ choices).
- **Sub-Curriculum Progression**:
  - **Tier 0**: Open navigation and steering (`faz_1v0.*`).
  - **Tier 1**: Obstacle corners, corridors, and cul-de-sacs (`faz_1v1.*`).
  - **Tier 2**: Deep occluded mazes requiring temporal memory (`faz_1v2.*`).
- **Domain Randomization & Novelty Exploration**:
  - Randomly alternates maps within the active tier on every reset to prevent coordinate memorization.
  - Awards an episodic spatial cell-visit bonus to incentivize discovering unexplored map regions.

---

## Quickstart

### 1. Environment Setup

Activate the pre-configured virtual environment:

```bash
source venv/bin/activate
```

Verify PyTorch and CUDA availability:
```bash
python3 -c "import torch; print('CUDA Ready:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
```

---

### 2. Interactive Testing (`play.py`)

Test the environment yourself or watch a trained policy in a 60 FPS Pygame window.

#### A. Play as a Human (Keyboard Controls)
```bash
python3 play.py --mode human --map configs/maps/faz_1v0.1.yaml
```
- **Controls**: <kbd>W</kbd>/<kbd>S</kbd> (Forward/Backward), <kbd>A</kbd>/<kbd>D</kbd> (Strafe), <kbd>Q</kbd>/<kbd>E</kbd> or Arrows (Turn).
- **Shortcuts**: <kbd>C</kbd> (Follow camera), <kbd>F1</kbd> (Toggle raycasts), <kbd>F2</kbd> (Reset agent).

#### B. Watch a Trained AI Policy
```bash
python3 play.py --mode watch --model-path checkpoints/model_final.pt --map configs/maps/faz_1v0.3.yaml
```

---

### 3. Training & Fine-Tuning (`train.py`)

Run high-throughput recurrent PPO training on your GPU:

```bash
# 1. Train a new model from scratch:
python3 train.py --total-timesteps 50000 --save-interval 5000

# 2. Resume / fine-tune from an existing checkpoint:
python3 train.py --resume checkpoints/model_latest.pt --total-timesteps 30000

# 3. Force training on Tier 0 (faz_1v0.*) with tier lock:
python3 train.py --resume checkpoints/model_latest.pt --tier 0 --lock-tier --total-timesteps 30000

# 4. Train specifically on Tier 1 (Obstacles & Corridors):
python3 train.py --resume checkpoints/model_latest.pt --tier 1 --total-timesteps 50000
```

#### Key Command-Line Flags:
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--total-timesteps` | `50000` | Total environment frames to train. |
| `--resume` | `""` | Path to `.pt` checkpoint to resume weights from. |
| `--tier` | `-1` | Enforce curriculum tier (`0`, `1`, or `2`). `-1` uses saved checkpoint tier. |
| `--lock-tier` | `False` | Disables auto-promotion; locks the agent onto the chosen tier. |
| `--lr` | `3e-4` | Learning rate for Adam optimizer. |
| `--ent-coef` | `0.01` | Policy entropy bonus weight for exploratory actions. |
| `--rollout-steps` | `256` | Experience buffer size per PPO update. |
| `--batch-seq-len` | `32` | Sequence unroll length for the recurrent GRU core. |

---

## Repository Structure

```text
deco-swarm/
├── configs/                      # YAML configurations
│   ├── agent.yaml                # Kinematics, speeds, 90 rays, 180° FOV
│   ├── simulation.yaml           # 60 Hz physics tick & collision iterations
│   ├── render.yaml               # Pygame display resolution, camera zoom
│   └── maps/                     # Curriculum maps (faz_1v0.*, faz_1v1.*, faz_1v2.*)
│
├── documents/                    # Comprehensive system documentation
│   ├── quickstart_guide.md       # Friendly training & testing reference guide
│   ├── scripts_overview.md       # Purpose of every script across the codebase
│   ├── model_architecture.md     # 1D-CNN + GRU late-fusion design & alternatives
│   ├── environment_rules.md      # Reward math, goal rules, and curriculum criteria
│   ├── project_overview.md       # Scope, MARL doctrine, and kinematics
│   ├── architecture.md           # Clean architecture layers & decoupling
│   ├── roadmap.md                # 28-week development plan & milestones
│   └── risks.md                  # RL pathologies, retraining risks, and solutions
│
├── src/                          # Main source package
│   ├── engine/                   # Pure headless physics core (NO Pygame)
│   │   ├── core/                 # Level 0: Geometry and circle-segment collision
│   │   ├── entities/             # Level 1: Agent (640-float obs) and GameMap
│   │   ├── services/             # Level 2: RaycastService (90 rays parametric)
│   │   ├── controllers.py        # Input abstraction (Keyboard, ActionDecoder)
│   │   ├── map_loader.py         # YAML map parser
│   │   └── world.py              # Level 3: Simulation step orchestrator
│   ├── envs/                     # PettingZoo / Gymnasium RL wrappers
│   │   ├── single_agent_env.py   # ParallelEnv wrapper with Action Repeat (N=6)
│   │   └── curriculum.py         # Sub-curriculum manager & domain randomization
│   ├── models/                   # Neural network architectures
│   │   └── actor_critic.py       # RecurrentActorCritic (1D-CNN + MLP + GRU)
│   ├── rendering/                # Pygame visualization layer
│   │   ├── camera.py             # Viewport transform, pan, zoom, follow
│   │   ├── drawers/              # Specialized drawers (agent, goal, rays, map)
│   │   └── pygame_renderer.py    # Pygame canvas and HUD telemetry
│   └── training/                 # RL training pipelines
│       └── ppo.py                # Recurrent PPO with GAE & sequence unrolling
│
├── checkpoints/                  # Saved PyTorch model weights (.pt)
├── play.py                       # Interactive visual player / model inspector
├── train.py                      # Headless training entrypoint
└── README.md
```

---

## Documentation Index

Detailed specifications and architectural deep dives can be found in [`documents/`](documents/):
- **[Quickstart Guide](documents/quickstart_guide.md)**: Friendly guide with all commands, flags, and reward tuning tips.
- **[Scripts Overview](documents/scripts_overview.md)**: Explanation of every file and module across the codebase.
- **[Model Architecture Guide](documents/model_architecture.md)**: Deep dive into the 1D-CNN, GRU, and alternative models.
- **[Environment Rules & Rewards](documents/environment_rules.md)**: Exact mathematical formulations for rewards and curriculum.
- **[Project Overview](documents/project_overview.md)**: Project scope, emergent squad tactics, and MARL goals.
- **[Architecture & Clean Layering](documents/architecture.md)**: Principles of clean layering and decoupling.
- **[28-Week Roadmap](documents/roadmap.md)**: Chronological project timeline and phase progression.
- **[Technical Risks & Pathologies](documents/risks.md)**: Retraining risks, POMDP pitfalls, and mitigation strategies.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.