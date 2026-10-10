# DECO-SWARM: Script & Module Purpose Reference

This document provides a concise reference explaining the exact responsibility of every source file in the repository.

---

## 1. Top-Level Entrypoint Scripts

| Script | Responsibility |
| :--- | :--- |
| [`play.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/play.py) | **Interactive Visual Testbed**: Launches a 60 FPS Pygame window. In `--mode human`, lets a developer test maps, physics, and raycasts via keyboard. In `--mode watch`, executes a trained neural network checkpoint live with smooth camera tracking and HUD telemetry. |
| [`train.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/train.py) | **High-Throughput Headless Trainer**: The main training loop. Runs headless (no GUI overhead) on GPU, manages rollout collection, executes recurrent PPO updates, coordinates sub-curriculum promotions, and saves model checkpoints. |

---

## 2. Environment & Curriculum (`src/envs/`)

| Script | Responsibility |
| :--- | :--- |
| [`src/envs/single_agent_env.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/single_agent_env.py) | **PettingZoo Environment Wrapper**: Adapts the continuous physics world to standard RL MDP methods (`reset`, `step`). Downsamples 60 Hz physics to 10 Hz decisions via **Action Repeat ($N=6$)**, calculates potential-based distance rewards, step penalties, collision penalties, and episodic spatial cell-visit exploration bonuses. |
| [`src/envs/curriculum.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/curriculum.py) | **Sub-Curriculum Manager**: Organizes map tiers (`Tier 0` open, `Tier 1` obstacles, `Tier 2` mazes). Tracks sliding-window success rates, handles within-tier domain randomization (randomly alternating maps per episode), and manages promotion/demotion triggers. |

---

## 3. Neural Network Models (`src/models/`)

| Script | Responsibility |
| :--- | :--- |
| [`src/models/actor_critic.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/models/actor_critic.py) | **Recurrent Actor-Critic Model (`RecurrentActorCritic`)**: Implements the Late-Fusion network for POMDP navigation. Uses a **1D-CNN** to encode 90 raycasts (630 floats), an **MLP** to encode 10 ego scalars, a **GRU Cell** (128-d) to maintain temporal memory across frames, a **Multi-Discrete Actor Head** for movement and rotation, and a **Critic Head** for scalar state-value $V(s)$. |

---

## 4. Reinforcement Learning Training (`src/training/`)

| Script | Responsibility |
| :--- | :--- |
| [`src/training/ppo.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/training/ppo.py) | **Recurrent PPO Algorithm & Rollout Buffer**: Manages the trajectory storage buffer, computes Generalized Advantage Estimation (GAE) and TD targets, and executes mini-batch sequence unrolling with clipped surrogate loss and entropy regularization. |

---

## 5. Simulation & Physics Engine (`src/engine/`)

The engine is **strictly headless** (zero graphical or Pygame dependencies), enabling ultra-fast physics stepping during training.

| Script / Directory | Responsibility |
| :--- | :--- |
| [`src/engine/world.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/world.py) | **Simulation Orchestrator**: Manages active agents, coordinates kinematic updates, runs iterative circle-segment collision resolution, updates raycasts and polar goals, and checks win conditions. Supports hot-reloading maps via `load_map()`. |
| [`src/engine/map_loader.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/map_loader.py) | **Map Parser**: Reads YAML map configurations, extracts arena dimensions, outer boundary walls, internal line segments, box obstacles, agent spawn positions, and target goal coordinates into typed `MapConfig` objects. |
| [`src/engine/controllers.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/controllers.py) | **Input Abstraction**: Defines `AgentAction` dataclass, `AgentController` interface, `KeyboardController` for human input, and `ActionDecoder` which translates discrete neural network indices `(move_idx, rot_idx)` into physical motor velocities `(v_fwd, v_strafe, omega)`. |
| [`src/engine/core/geometry.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/core/geometry.py) | **Pure Mathematical Primitives (Level 0)**: Point2D, Segment2D, distance calculations, angle wrapping ($[-180^\circ, 180^\circ]$), vector projections, and ego-centric polar target calculations. Zero dependency on game entities. |
| [`src/engine/core/collision.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/core/collision.py) | **Collision Resolution (Level 0)**: Implements continuous circle-segment collision resolution (`resolve_circle_segment`), sliding agents along obstacle surfaces and returning contact detection flags. |
| [`src/engine/entities/agent.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/entities/agent.py) | **Agent Kinematics & Observation Publishing**: Manages position, heading, velocity, health, ammo, and ray hit buffers. Publishes the formalized **640-float ego-centric observation tensor** via `get_observation()`. |
| [`src/engine/entities/map.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/entities/map.py) | **Map Entity**: Holds wall segments, boundaries, and spatial properties of the current map. |
| [`src/engine/services/raycast.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/services/raycast.py) | **Perception Service**: Computes parametric ray-segment intersection tests (`cast_rays`) for 90 rays across a 180° field-of-view, producing normalized distances and hit metadata. |

---

## 6. Rendering & Visualization (`src/rendering/`)

| Script / Directory | Responsibility |
| :--- | :--- |
| [`src/rendering/pygame_renderer.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/rendering/pygame_renderer.py) | **Pygame Canvas & UI**: Manages the Pygame display surface, renders background, invokes dedicated drawers, and draws on-screen HUD telemetry. |
| [`src/rendering/camera.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/rendering/camera.py) | **Virtual 2D Camera**: Handles world-to-screen coordinate transformations, zoom scaling, mouse-drag panning, and smooth target tracking (`camera.follow`). |
| [`src/rendering/drawers/`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/rendering/drawers/) | **Modular Renderers**: Cleanly partitioned drawing components: `agent_drawer.py` (agent circle & heading), `goal_drawer.py` (pulsing goal perimeter), `map_drawer.py` (walls & boxes), and `ray_drawer.py` (color-coded raycast cones). |

---

## 7. Configurations (`configs/`)

| Path | Purpose |
| :--- | :--- |
| `configs/simulation.yaml` | Physics tick frequency (60 Hz) and collision relaxation iterations. |
| `configs/agent.yaml` | Agent radius, linear speed, angular speed, FOV angle (180°), and ray count (90). |
| `configs/render.yaml` | Pygame window dimensions, target FPS, and camera defaults. |
| `configs/maps/` | Curriculum map definitions categorized by tier (`faz_1v0.*`, `faz_1v1.*`, `faz_1v2.*`). |
