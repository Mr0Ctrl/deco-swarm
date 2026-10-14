# Autonomous Tactical Coordination in Decentralized Swarms via Multi-Agent Reinforcement Learning (DECO-SWARM)

## Academic Abstract & Research Problem Statement

Cooperative decision-making under partial observability, communication constraints, and dynamic competitive pressure remains a central challenge in autonomous multi-agent robotics and defense simulation systems. In complex spatial domains, conventional heuristic planners (e.g., behavior trees, finite-state machines, or static potential fields) fail to generalize to adversarial tactics, while fully centralized planners suffer from exponential state-space explosion and vulnerability to communication link degradation.

**DECO-SWARM** investigates the emergence of decentralized tactical cooperation using **Multi-Agent Deep Reinforcement Learning (MADRL)** formulated under a **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**. The project examines how autonomous agents—operating strictly on localized ego-centric perception and low-bandwidth communication channels—can synthesize foundational tactical doctrines (such as sector coverage, dynamic crossfire positioning, bounding maneuvers, and perimeter security) in continuous 2D spatial environments without explicit rule-based choreography.

---

## 1. Research Objectives & Key Questions

1. **Autonomous Tactical Synthesis (Emergent Behaviors):**
   Can decentralized recurrent policies discover military/tactical squad doctrines (e.g., crossfire setups, flanking, corner-clearing, rear-guard defense) purely through reward incentives and competitive multi-agent pressure?
2. **Robustness Under Bandwidth-Constrained Communication:**
   How do low-bandwidth, discrete tactical communication tokens compare against unconstrained continuous latent communication channels in preventing message entropy collapse in CTDE frameworks?
3. **Hardware-Efficient Recurrent Policy Training:**
   How effectively can a hybrid compute pipeline (C++ simulation core with zero-copy shared memory paired with PyTorch GAE-PPO/MAPPO) train recurrent multi-agent policies on consumer-grade workstation/laptop hardware?
4. **Generalization Across Topological Curricula:**
   Can a multi-tier curriculum mitigate sample complexity and catastrophic forgetting when navigating from open Euclidean topologies to occluded, dead-end maze topologies?

---

## 2. Theoretical Formulation & Mathematical Framework

### 2.1 Dec-POMDP Formalization
The cooperative multi-agent environment is formalized as a Dec-POMDP defined by the tuple:
$$\mathcal{M} = \langle \mathcal{N}, \mathcal{S}, \{\mathcal{A}_i\}, \mathcal{P}, \{R_i\}, \{\Omega_i\}, \mathcal{O}, \gamma \rangle$$

- $\mathcal{N} = \{1, \dots, N\}$: Set of autonomous agents.
- $\mathcal{S}$: Global environment state space (positions, orientations, continuous velocities, wall segments, tactical objectives).
- $\mathcal{A}_i$: Action space for agent $i$ (multi-discrete motor commands and discrete triggering).
- $\mathcal{P}(s' \mid s, \mathbf{a})$: Continuous kinematic state transition probability function.
- $R_i(s, \mathbf{a}, s')$: Shared/individual reward signals balancing task progress, survivability, and spatial coordination.
- $\Omega_i$: Local observation space for agent $i$ (ego-centric raycasts and odometry).
- $\mathcal{O}(o_i \mid s, a)$: Observation emission probability function modeling sensor occlusions and limited field-of-view (FOV).
- $\gamma \in [0, 1)$: Temporal discount factor ($\gamma = 0.99$).

### 2.2 CTDE Paradigm (Centralized Training with Decentralized Execution)
To handle non-stationarity in multi-agent environments while guaranteeing decentralized operational autonomy:
- **Centralized Critic ($V_\phi(s)$):** Conditioned on the global environment state $s \in \mathcal{S}$ during training, stabilizing value estimation and variance in Generalized Advantage Estimation (GAE).
- **Decentralized Actor ($\pi_\theta(a_i \mid o_i, h_i)$):** Evaluated strictly on the local ego-centric observation history $h_i$ produced by recurrent memory units (GRU), allowing autonomous deployment without external state knowledge.

```mermaid
flowchart LR
    subgraph Centralized Training [Training Phase (CTDE)]
        GlobalState[Global State s] --> Critic[Centralized Critic V_phi]
        Buffer[PPO / MAPPO Replay & GAE] --> Critic
        Buffer --> ActorUpdate[Policy Gradient Updates]
    end

    subgraph Decentralized Execution [Deployment / Inference]
        Obs[Local Ego Observation o_i] --> RayCNN[1D-CNN Spatial Encoder]
        Odom[Ego Odometry & Status] --> MLP[Ego MLP Encoder]
        RayCNN --> GRU[Recurrent Memory (GRUCell)]
        MLP --> GRU
        GRU --> Policy[Decentralized Policy Head pi_theta]
        Policy --> Action[Continuous Kinematic Action a_i]
    end
```

---

## 3. System Architecture & High-Throughput Engineering

To ensure continuous multi-agent training is feasible on modest consumer workstation hardware without requiring high-performance computing (HPC) clusters, DECO-SWARM employs a dual-language, decoupled architecture:

```
+-------------------------------------------------------------------------+
|                              PYTHON LAYER                               |
|  - Multi-Agent PPO / MAPPO Optimization Pipeline                        |
|  - Recurrent Actor-Critic Models (1D-CNN + GRUCell + Multi-Discrete)    |
|  - Gymnasium / PettingZoo AEC Environment Compliant API                 |
+------------------------------------+------------------------------------+
                                     |  pybind11 & Zero-Copy NumPy
                                     v
+------------------------------------+------------------------------------+
|                         C++ SIMULATION CORE                             |
|  - Continuous Kinematics & Sub-stepping (60 Hz Integration)             |
|  - Analytical 2D Line-Segment Raycasting (Ray-Wall Intersections)       |
|  - Spatial Partitioning (Grid/BVH) & Continuous Collision Detection     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                         VISUALIZATION & TELEMETRY                       |
|  - Headless Execution (>1,000 FPS for asynchronous training)            |
|  - Optional Interactive Telemetry (Pygame / Raylib 60 FPS viewer)       |
+-------------------------------------------------------------------------+
```

### Continuous Kinematics & Temporal Discretization
- **Kinematic Substepping:** Physics integration runs at **60 Hz** ($\Delta t \approx 16.67\text{ ms}$) to prevent spatial tunneling and ensure continuous collision resolution along arbitrary line segments.
- **Decision Resolution (Macro-actions):** Policy inference executes at **10 Hz** via temporal action repetition ($N = 6$ physics steps per agent decision), reflecting realistic robotic actuation latencies.

---

## 4. Perception & Observation Formulation

Observations are strictly ego-centric, avoiding reliance on global coordinate frames to guarantee invariance to spatial translation and rotation:

$$\mathbf{o}_t \in \mathbb{R}^{640}$$

| Feature Stream | Dimensionality | Representation | Sensor Meaning |
| :--- | :---: | :---: | :--- |
| **Spatial Raycast Array** | $90 \text{ rays} \times 7$ | $630$ floats | Depth range $[0, 1]$, surface hit flags (wall, ally, adversary, objective), relative incident geometry ($\cos \Delta\theta, \sin \Delta\theta$) |
| **Ego-Polar Objective Vector** | $2$ | $2$ floats | Distance $\| \mathbf{d} \|_2$ and heading angle relative to current waypoint |
| **Internal Odometry** | $3$ | $3$ floats | Body-frame linear velocities $(v_x, v_y)$ and rotational rate $\omega$ |
| **Actuator Feedback ($a_{t-1}$)** | $3$ | $3$ floats | Preceding motor control signals $(u_{\text{fwd}}, u_{\text{strafe}}, u_{\text{rot}})$ |
| **Agent State Metrics** | $2$ | $2$ floats | Normalized structural health and remaining ammunition |
| **Tactical Comms Buffer** | Tail Padding | Reserved | Structured inter-agent communication channels |

### Recurrent Temporal Memory
To address state aliasing under severe occlusion (walls, corners, blind spots), the network integrates a **Gated Recurrent Unit (GRU)**. The recurrent hidden state $h_t \in \mathbb{R}^{128}$ enables the agent to maintain track of targets beyond its field of view and avoid dead-end cyclic loops.

---

## 5. Staged Pedagogical Curriculum & Research Milestones

Training progresses along a 4-stage empirical curriculum designed to isolate kinematic mastery from tactical coordination:

```
[Phase 1: Ego-Kinematics & Spatial Exploration]
     │  - Single-Agent Navigation (1v0)
     │  - 1D-CNN + GRU Validation in Multi-Tier Mazes
     ▼
[Phase 2: Ballistic Engagement & Target Tracking]
     │  - Stationary Adversary Target Practice (1v1)
     │  - Aiming Dynamics, Raycast Acquisition, Ammunition Economy
     ▼
[Phase 3: Adversarial Game-Theoretic Co-Evolution]
     │  - Dynamic 1v1 Combat & Self-Play
     │  - Dynamic Cover Utilization, Peeking, Continuous Firefights
     ▼
[Phase 4: Decentralized Swarm Coordination (Dec-POMDP)]
        - 3v3 / 5v5 Competitive Multi-Agent Scenarios
        - Bounding Overwatch, Angle Coverage, Discrete Radio Protocols
```

---

## 6. Target Academic Contributions

1. **Empirical Benchmarks for Consumer Hardware MARL:** Demonstrating that structured Dec-POMDP tactical emergent behaviors can be trained under strict compute budgets without high-performance clusters.
2. **Analysis of Topological Generalization:** Comprehensive evaluation of catastrophic forgetting and replay stabilization mechanisms across topological curricula.
3. **Structured vs. Latent Inter-Agent Communication:** Quantitative ablation measuring the emergence of cooperative tactics under structured discrete communication tokens versus continuous latent channels.
