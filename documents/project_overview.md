# DECO-SWARM: Project Overview & Scope

## 1. Project Goal & Scope

The primary objective of **DECO-SWARM** is to develop a Decentralized Multi-Agent Reinforcement Learning (MARL) system capable of training on consumer-grade laptop hardware within a simplified 2D tactical shooter arena. 

Instead of agents (autonomous infantry or drone swarms) moving in random or independent trajectories, the goal is for agents to discover military squad doctrines autonomously (**emergent behaviors**) through restricted tactical information sharing. Target emergent behaviors include:
- Crossfire setup and angle coverage
- Flanking maneuvers and ambushes
- Tactical room entry / clearing
- Rear guard coverage and perimeter security

The project follows a 28-week development roadmap, culminating in competitive 3v3 and 5v5 tactical missions (bomb defusal / area defense).

---

## 2. Technology Stack & Hybrid Architecture

To maximize training throughput while supporting visual debugging, the system adopts a hybrid architecture:

- **Simulation Core (C++ Migration Target):** High-CPU loops—including geometry processing, continuous movement physics, line-segment raycasting, and bullet tunneling prevention—will reside in C++.
- **Machine Learning (Python / PyTorch):** Neural network architectures (MLP / 1D-CNN + GRU), reward computation, MAPPO algorithm pipelines, and training execution reside in Python.
- **Bridge (pybind11 & Zero-Copy NumPy):** High-throughput data exchange between C++ and Python avoids deep-copy serialization overhead by utilizing shared zero-copy NumPy memory buffers.
- **Environment API Standard:** Python environments conform to **Gymnasium** (single-agent 1v0) and **PettingZoo** (multi-agent) interfaces for broad compatibility with MARL frameworks.
- **Visualization:** During training, the simulation runs **headless** at tens of thousands of steps per second. For debugging and demonstrations, rendering is supported via **Pygame** (current prototype) and later a lightweight embedded C++ renderer (**Raylib / SFML**) at 60 FPS.

---

## 3. Simulation & Continuous Kinematics

- **Physics Model:** Instead of a discrete grid, agents operate in continuous 2D space ($x, y$) governed by continuous kinematic motion.
- **Map Geometry:** Obstacles and walls are modeled as arbitrary line segments (supporting angled corners, narrow corridors, and rooms) rather than axis-aligned bounding boxes (AABBs).
- **Time Scale & Decision Frequency:**
  - **Physics Engine:** Runs at **60 Hz** ($\Delta t \approx 0.0167$ s) to prevent tunneling and provide smooth collision resolution.
  - **Decision Frequency (Action Repeat):** Agents make decisions at **10 Hz** (every 6 physics frames, `action_repeat = 6`). Movement actions persist across 6 frames, while trigger actions are gated by weapon cooldowns.

---

## 4. Observation Space (Ego-Centric Formulation)

Agents have no global awareness of absolute map coordinates; observations are strictly ego-centric (relative to the agent's body position and orientation).

The formalized observation tensor produces a **640-float vector** (`np.float32`):

| Block | Shape | Channels / Features | Description |
| :--- | :--- | :--- | :--- |
| **1. Raycast Perception** | 90 rays $\times$ 7 | $630$ floats | Normalized distance `[0.0, 1.0]`, wall flag, friend flag, enemy flag, objective flag, relative cosine, relative sine of viewed entity |
| **2. Ego Polar Target** | 2 floats | $2$ floats | Relative distance ($r$) and relative heading ($\Delta\theta$) to current tactical objective |
| **3. Ego Odometry & Velocities** | 3 floats | $3$ floats | Normalized linear velocities and angular velocity ($v_x, v_y, \omega$) |
| **4. Previous Action ($a_{t-1}$)** | 3 floats | $3$ floats | Previous step's applied movement vector $(u_{\text{fwd}}, u_{\text{strafe}}, u_{\text{rot}})$ |
| **5. Self Status** | 2 floats | $2$ floats | Normalized health (`[0.0, 1.0]`) and normalized ammo remaining (`[0.0, 1.0]`) |
| **Total** | | **640 floats** | Zero-padding reserved at the tail for future tactical communication tokens |

### Memory & Recurrence
Due to Partial Observability (POMDP), agents incorporate a **Recurrent Neural Network (GRU cell)** rather than frame stacking. This maintains memory of occluded enemies, targets outside the 180° field-of-view, and avoids spinning loops in dead ends.

---

## 5. Action Space (Multi-Discrete)

Controls map to a multi-discrete structure reflecting tactical inputs:
1. **Movement Branch (5 choices):** `0: Idle, 1: Forward, 2: Backward, 3: Strafe Left, 4: Strafe Right`
2. **Rotation Branch (3 choices):** `0: Hold Heading, 1: Turn Left, 2: Turn Right`
3. **Trigger Branch (2 choices):** `0: Hold Fire, 1: Fire`

The `ActionDecoder` in `src/engine/controllers.py` translates these discrete indices into continuous motor velocities `(move_fwd, strafe, rotate_dir)` for the physics engine.

---

## 6. Communication & Decision Framework

- **CTDE (Centralized Training with Decentralized Execution):**
  - **Critic (Training only):** Observes global map state, team positions, and opponent ground truth.
  - **Actor (Execution / Deployment):** Operates purely on local ego-centric observations and received radio messages.
- **Structured Radio Protocol:** Rather than unconstrained high-entropy float vectors (which suffer from communication collapse), agents communicate structured tactical tokens (e.g., target coordinates, distress beacons, room status).

---

## 7. Curriculum Learning Strategy

To train efficiently on modest hardware without suffering from sparse-reward bottlenecks:

1. **Phase 1 (1v0 Navigation - Current Active Milestone):** Solo agent navigation in mazes. Raycast navigation and GRU memory without enemies or teammates.
2. **Phase 2 (1v1 Static Target Practice):** Stationary targets; learning raycast target acquisition, aiming, and ammunition economy.
3. **Phase 3 (1v1 Dynamic Combat):** Mobile adversary; cover utilization and moving firefights.
4. **Phase 4 (3v3 / 5v5 Swarm & Self-Play):** Teammates introduced, tactical radio active, MAPPO league self-play enabled.
