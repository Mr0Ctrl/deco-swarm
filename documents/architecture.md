# DECO-SWARM: System Architecture

## 1. Architectural Principles & Clean Layering

The codebase enforces a strict **Clean Architecture** with inward-only dependencies. Lower layers have zero knowledge of higher layers:

```text
[LEVEL 4: CONSUMERS]          rendering/ (Pygame)    │   envs/ (Gymnasium / PettingZoo)
                                       ▲              │              ▲
───────────────────────────────────────┼──────────────┼──────────────┼───────
[LEVEL 3: ORCHESTRATOR]                └─────────── world.py ────────┘
                                                         ▲
─────────────────────────────────────────────────────────┼───────────────────
[LEVEL 2: SERVICES]                              services/ (Raycast)
                                                         ▲
─────────────────────────────────────────────────────────┼───────────────────
[LEVEL 1: ENTITIES]                              entities/ (Agent, Map)
                                                         ▲
─────────────────────────────────────────────────────────┼───────────────────
[LEVEL 0: PURE CORE]                             core/ (Geometry, Collision)
```

### Layer Responsibilities
- **`core/` (Level 0):** Pure spatial math and algorithms (`geometry.py`, `collision.py`). Operates strictly on points, vectors, line segments, angles, and circle-segment collision resolutions. No entity awareness.
- **`entities/` (Level 1):** Domain entities holding internal state (`agent.py`, `map.py`). The `Agent` manages its kinematics, health, ammo, ray perception buffers, and generates the formalized observation tensor via `get_observation()`.
- **`services/` (Level 2):** Algorithmic processors (`raycast.py`). Resolves multi-ray intersection math against line segments using core mathematical primitives.
- **`world.py` (Level 3):** Simulation orchestrator. Loads map configurations, manages the tick step loop, updates agent positions, resolves collisions iteratively, and triggers perception updates.
- **`rendering/` & `envs/` (Level 4):** Consumers. 
  - `rendering/` visualizes simulation state via Pygame without polluting the physics core.
  - `envs/` wraps `World` into standard Gymnasium / PettingZoo MDP interfaces for RL training.

---

## 2. Decoupled Headless Engine Design

A core requirement is training throughput:
- The engine (`src/engine/`) has **zero graphical dependencies** (no Pygame imports inside physics, entities, or services).
- Headless execution allows stepping through thousands of environment steps per second.
- Visual debugging (`play.py`, `src/rendering/`) attaches non-intrusively to read `World` state and render frames at 60 FPS on demand.

---

## 3. Observation Publishing (`Agent.get_observation`)

Sensors are formally published through a standardized interface on the `Agent` entity:
- **Tensor Format:** 1D `np.ndarray` of dtype `np.float32`, total dimension: **640**.
- **Normalization:** All values are bounded to $[0.0, 1.0]$ or $[-1.0, 1.0]$.
- **Usage:**
  - RL environments pass this tensor directly to the policy network.
  - HUD and renderers read from this standardized tensor to display telemetry.
  - Future C++ bindings will map this tensor directly into zero-copy shared memory.

---

## 4. Action Decoding & Control Abstraction

Located in `src/engine/controllers.py`:
- **`AgentAction`:** Unified dataclass holding `(move_fwd, strafe, rotate_dir, fire)`.
- **`AgentController` (Abstract Base Class):** Common interface supporting interchangeable controllers:
  - `KeyboardController`: Human testing via WASD / Arrows.
  - `RandomController`: Fuzz testing and mock baselines.
  - `ActionDecoder`: Translates discrete action tuple `(move_idx, rotate_idx)` from neural network policies into continuous movement signals.

---

## 5. Timing & Action Repeat

To prevent bullet tunneling, corner clipping, and instability in continuous kinematics:
- **Physics Tick:** Runs at **60 Hz** ($\Delta t = 1/60$ s) with iterative collision relaxation (`collision_iterations: 2`).
- **Policy Decision Rate:** Agents evaluate policies at **10 Hz** ($\Delta t = 0.1$ s).
- **Action Repeat ($N=6$):** The RL environment step function executes 6 physics substeps per RL action before computing rewards and next observations.