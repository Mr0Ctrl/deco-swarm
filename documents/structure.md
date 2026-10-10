# DECO-SWARM: Repository Structure & Component Organization

```text
deco-swarm/
├── documents/                      # System documentation & specifications (English)
│   ├── project_overview.md         # Scope, technical stack, kinematics, observation & action spaces
│   ├── architecture.md             # Clean architecture layers, decoupling, and interfaces
│   ├── roadmap.md                  # 28-week development timeline & milestone status
│   ├── development.md              # Frameworks, neural network architecture, performance targets
│   ├── risks.md                    # Retraining risks, training pathologies, and mitigation
│   └── structure.md                # Repository layout and module responsibilities
│
├── configs/                        # YAML configuration files
│   ├── agent.yaml                  # Body dimensions, speeds, and perception parameters (90 rays, 180° FOV)
│   ├── render.yaml                 # Screen resolution, FPS, camera defaults
│   ├── simulation.yaml             # Physics tick rate (60 Hz) and collision iterations
│   └── maps/                       # Curriculum map definitions
│       ├── faz_1v0.1.yaml ...      # Incremental difficulty navigation mazes
│       └── haritalar.md            # Map design notes and layout descriptions
│
├── src/                            # Core Python package
│   ├── __init__.py
│   │
│   ├── engine/                     # Headless physics & simulation core (NO Pygame dependency)
│   │   ├── __init__.py
│   │   ├── core/                   # Level 0: Pure math & geometry
│   │   │   ├── geometry.py         # Point2D, distance, angle wrapping, ego polar computation
│   │   │   └── collision.py        # Circle-segment collision resolution
│   │   ├── entities/               # Level 1: Domain entities
│   │   │   ├── agent.py            # Agent kinematics, state, and 640-float observation tensor
│   │   │   └── map.py              # GameMap and line segment collections
│   │   ├── services/               # Level 2: Algorithmic services
│   │   │   └── raycast.py          # Line-ray intersection and batch raycasting
│   │   ├── controllers.py          # Input abstraction (Keyboard, Random, ActionDecoder)
│   │   ├── map_loader.py           # YAML map loader and dataclass parser
│   │   └── world.py                # Level 3: Simulation orchestrator (step loop, physics tick)
│   │
│   ├── rendering/                  # Level 4: Visualization layer (Pygame)
│   │   ├── __init__.py
│   │   ├── camera.py               # Viewport transform, pan, zoom, agent-follow
│   │   ├── drawers/                # Dedicated drawer modules
│   │   │   ├── agent_drawer.py     # Agent circle, heading indicator, health
│   │   │   ├── goal_drawer.py      # Objective markers
│   │   │   ├── map_drawer.py       # Wall segments and obstacle rendering
│   │   │   └── ray_drawer.py       # Raycast beam visualization
│   │   └── pygame_renderer.py      # Window initialization, event loop, HUD overlay
│   │
│   ├── envs/                       # Level 4: RL environment wrappers (Gymnasium / PettingZoo)
│   │   ├── __init__.py             # [Planned]
│   │   ├── single_agent_env.py     # [Planned] Gymnasium wrapper for Phase 1 (1v0) with Action Repeat
│   │   └── swarm_env.py            # [Planned] PettingZoo ParallelEnv wrapper for Phases 3-4
│   │
│   ├── models/                     # PyTorch neural network architectures
│   │   ├── __init__.py             # [Planned]
│   │   └── actor_critic.py         # [Planned] Multi-modal feature extractor + GRU + Actor-Critic heads
│   │
│   └── training/                   # RL training pipelines
│       ├── __init__.py             # [Planned]
│       └── ppo.py                  # [Planned] PPO implementation (cleanrl-style) with GAE & GRU rollouts
│
├── cpp/                            # [Phase 2] C++ high-performance acceleration core
│   ├── CMakeLists.txt              # Build configuration
│   └── ...                         # Fast raycast & collision routines via pybind11
│
├── play.py                         # Interactive visualization entry point (Pygame GUI testbed)
├── train.py                        # [Planned] High-throughput headless training entry point
│
├── .gitignore
└── README.md
```
