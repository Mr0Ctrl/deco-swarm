# DECO-SWARM

> **Decentralized Multi-Agent Reinforcement Learning for 2D Capture the Flag with Learned Inter-Agent Communication (Pure JAX)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**DECO-SWARM** is an end-to-end Multi-Agent Reinforcement Learning (MARL) research platform built to train autonomous agent teams in continuous 2D Capture the Flag (CTF) arenas on a single consumer GPU.

Agents operate under strict **partial observability**: they perceive the world through ego-centric panoramic semantic raycasts, local proprioception, recurrent memory (`GRU`), and a learned inter-agent communication channel aggregated via permutation-invariant attention/pooling.

---

## Architecture & Design Documentation

All system specifications, physics math, reward shaping, neural design, and safety rules live in [`documents/`](documents/README.md):

- **[System Overview & Index](documents/README.md)** — Core architectural pillars, end-to-end flow, and v1-to-v2 evolution guide.
- **[01. Configuration & Arenas](documents/01_configuration_and_arenas.md)** — Parameter schemas and unified 2D line-segment arena compilation.
- **[02. Simulation & Perception](documents/02_simulation_and_perception.md)** — Headless 60 Hz physics / 10 Hz decisions, sliding collisions, semantic raycasting, and ego-polar bearings.
- **[03. Environment, Rewards & Curriculum](documents/03_environment_rewards_and_curriculum.md)** — Episodic macro-steps, termination vs. truncation, shaped rewards, CTF rules, and progressive curriculum.
- **[04. Agent Architecture & Training](documents/04_agent_architecture_and_training.md)** — Modular late-fusion + GRU design, communication & CTDE extensions, recurrent PPO, and the 5-rule checkpoint/recovery system.
- **[05. Visualization & Interaction](documents/05_visualization_and_interaction.md)** — Read-only 2D viewer, camera transforms, layered rendering, and human vs. AI watch modes.
- **[06. v2 Development Roadmap](documents/06_development_roadmap.md)** — Step-by-step build phases, throughput/test gates, and evaluation plan for pure-JAX CTF.

---

## Repository Status

- **`v2/jax-ctf`** *(active branch)*: Clean starting point for the pure-JAX CTF + communication implementation.
- **`archive/v1-pytorch-cpp`** *(tag `v1.0-nav-archive`)*: Archived v1 single-agent PyTorch + C++ navigation prototype.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
