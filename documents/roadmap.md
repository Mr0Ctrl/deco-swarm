# DECO-SWARM: 28-Week Master Roadmap

## Phase 1: Foundations & Proof of Concept (Weeks 1 - 4)

*Primary Objective: Verify RL environment mechanics and validate that a single agent (1v0) learns obstacle avoidance and navigation without looping in dead ends.*

### Weeks 1 - 2: 1v0 Maze Environment & First Training (Python)
- [x] Pure Python continuous kinematic engine and raycast mathematics.
- [x] Map configuration loader and multi-difficulty maze topologies (`configs/maps/faz_1v*.yaml`).
- [x] Standardized 640-float ego-centric observation tensor (`Agent.get_observation()`).
- [x] Multi-discrete action decoder (`ActionDecoder`).
- [ ] Gymnasium single-agent environment wrapper (`src/envs/single_agent_env.py`) with Action Repeat ($N=6$).
- [ ] Reward shaping function (progress toward goal, obstacle avoidance penalty, time decay).
- [ ] PPO training loop with GRU memory cells for recurrent state handling.

### Weeks 3 - 4: Action Space Verification & Static Target Practice (1v1 Static)
- Verify maze navigation convergence without local minima (eliminating spinning behavior).
- Introduce multi-discrete trigger actions (movement + rotation + fire).
- Place stationary targets to train raycast aiming reflexes and ammunition economy.

---

## Phase 2: Performance Optimization & Multi-Agent Transition (Weeks 5 - 9)

*Primary Objective: Overcome Python CPU bottlenecks using C++ acceleration and transition to PettingZoo MARL APIs.*

### Weeks 5 - 7: C++ Acceleration Bridge (pybind11)
- Profile computational bottlenecks (Raycast intersection and continuous collision detection).
- Port geometry math and raycast loops to high-performance C++.
- Establish zero-copy shared NumPy memory buffers via `pybind11` to exceed **10,000+ steps/second**.

### Weeks 8 - 9: Transition to Multi-Agent Environment (PettingZoo)
- Transition from Gymnasium to PettingZoo Parallel API (`ParallelEnv`).
- Partition observation tensors (local actor observations vs. global critic state for CTDE).

---

## Phase 3: Tactical Combat & Midterm Milestone (Weeks 10 - 14)

*Primary Objective: Achieve dynamic 1v1 engagements, establish baseline communication channels, and deliver the midterm evaluation prototype.*

### Weeks 10 - 12: Dynamic 1v1 & Communication Channel
- Dynamic 1v1 combat training (mobile opponent, evasive maneuvers, cover utilization).
- Integrate tactical radio communication channel (sharing detected target coordinates).

### Weeks 13 - 14: 3v3 Swarm Prototype & Midterm Evaluation
- Permutation-invariant team pooling (Deep Sets / MaxPooling across teammates).
- **Midterm Deliverable:** High-speed C++ accelerated engine showcasing solved 1v0 navigation, effective 1v1 combat, and initial 3v3 cooperative maneuvers.

---

## Phase 4: Advanced MARL & Self-Play (Weeks 15 - 20)

*Primary Objective: Induce military squad behaviors (crossfire, ambushes, room clearing) via self-play.*

### Weeks 15 - 17: Adversarial Training (Self-Play / League Play)
- Continuous league self-play (T vs. CT factions) against historical policy checkpoints to prevent overfitting.
- Advanced reward shaping: penalizing individual greed, friendly-fire constraints, rewarding sacrificial play and tactical positioning.

### Weeks 18 - 20: Heterogeneous Roles & Policy Fine-Tuning
- Introduce distinct weapon profiles (Rifle, Shotgun, DMR) via one-hot vector conditioning.
- Learning rate annealing and fine-tuning to incorporate role specialization.

---

## Phase 5: Human-AI Interaction & High-Performance UI (Weeks 21 - 25)

*Primary Objective: Build an intuitive, playable visual client for demonstration and human vs. AI benchmarking.*

### Weeks 21 - 23: C++ Graphical Client (Raylib / SFML)
- Lightweight 60 FPS graphical renderer embedded in C++.
- Real-time visualization of raycast cones, bullets, health bars, and tactical communication links.

### Weeks 24 - 25: Interactive Socket / Playable Client
- Socket-based keyboard/mouse input bridging into the engine.
- Human player competing alongside or against trained swarm models.

---

## Phase 6: Validation, Metrics & Final Delivery (Weeks 26 - 28)

*Primary Objective: Statistically validate squad doctrine emergence against military benchmarks and finalize project submission.*

### Weeks 26 - 27: Empirical Analysis & Baseline Comparison
- Quantitative comparison: Decentralized tactical radio vs. omniscient centralized baseline (Oracle).
- Metrics extraction: dispersion distance, crossfire angles, hit accuracy, survivability rates.

### Week 28: Final Delivery
- Comprehensive technical report, presentation materials, and final project defense.
