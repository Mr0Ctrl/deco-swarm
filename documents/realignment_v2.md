# DECO-SWARM v2: Project Realignment, Architecture Trade-offs & Advisor Discussion Document

> **Purpose of this document:** A comprehensive reference for team alignment and thesis advisor review. It defines the repositioned project scope, explains the engineering trade-offs across simulation, perception, and MARL, details the required changes from the current baseline, and answers how to swap sensor input dimensions (1D Scan $\leftrightarrow$ $64 \times 64$ Depth Matrix) while preserving/freezing the recurrent (GRU) and policy (Dense) layers.

---

## 1. New Project Definition (For Thesis Proposal, TÜBİTAK & Industry Applications)

### 1.1 Title Options
- **Primary (Dual-Use / SAR & Robotics Focus):**
  > **DECO-SWARM: End-to-End Decentralized Multi-Agent Reinforcement Learning for Autonomous Search and Navigation in Occluded, Communication-Constrained Environments**
- **Alternative (Defense & Tactical Autonomy Focus):**
  > **DECO-SWARM: High-Throughput Multi-Agent Reinforcement Learning for Decentralized Swarm Coordination and Reconnaissance Using Depth Perception**

### 1.2 Executive Pitch (Replacing the "Tactical Shooter" Framing)
**Problem:** Autonomous robot swarms operating inside collapsed buildings (earthquake search and rescue), subterranean tunnels, or contested urban areas cannot rely on GPS, pre-built maps, or uninterrupted central communication. Classical modular pipelines (SLAM + global occupancy map + centralized task allocator + local planner) are computationally heavy on edge hardware, brittle under severe depth-sensor noise, and fail when communication links drop.

**Proposed Solution:** DECO-SWARM is a custom high-throughput C++/Python simulation engine and **end-to-end Multi-Agent Reinforcement Learning (MARL)** framework. Each autonomous agent maps local ego-centric depth perception, internal odometry, and low-bandwidth inter-agent radio tokens directly into continuous kinematic control commands via a recurrent deep neural network. Using **Centralized Training with Decentralized Execution (CTDE / MAPPO)**, agents autonomously learn to:
1. Navigate complex static and dynamic obstacles (boxes, pipes, corridors, moving crowds),
2. Maintain temporal spatial memory (GRU belief state) to avoid cyclic deadlocks without a global coordinate oracle, and
3. Coordinate area search and target discovery under severe communication bandwidth and packet-loss constraints.

**Sim-to-Real Bridge:** By formulating perception around depth geometry (either 1D depth slices or $64 \times 64$ 2.5D depth matrices), policies trained at 50,000+ steps/second in our custom C++ engine can be fed directly from a low-cost monocular RGB camera using modern lightweight depth estimation (e.g., *Depth Anything V2 Small*) on edge hardware (Jetson Nano / laptop).

---

## 2. Answering Your Technical Question: Can We Change the Input Layer Shape Later While Freezing the GRU and Dense Layers?

**Short Answer: YES—and because you already built [`RecurrentActorCritic`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/models/actor_critic.py) with a Modular Late-Fusion architecture, your model is already 95% structured for this!**

However, there is a critical **representation alignment problem** you must understand so the frozen GRU doesn't receive "garbage" numbers when you swap the input encoder.

### 2.1 Why Your Current Architecture Makes This Structurally Possible
Look at how your [`RecurrentActorCritic`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/models/actor_critic.py#L46-L106) is currently wired:

```text
[Spatial Input] ──► Spatial Encoder (CNN) ──► z_spatial (64-d) ┐
                                                               ├──► concat (96-d) ──► GRUCell (128-d) ──► Actor/Critic Heads
[Ego Scalars]   ──► Ego Encoder (MLP)     ──► z_ego (32-d)     ┘
```

Notice that the `GRUCell(96, 128)`, `actor_backbone`, `move_head`, `rot_head`, and `critic_head` **never see the raw sensor shape!** They only see the **64-dimensional latent vector** $z_{\text{spatial}} \in \mathbb{R}^{64}$ output by `self.spatial_encoder`.

Therefore, you can replace:
- **Encoder A (1D Scan):** `Conv1d(7, 64) -> Linear -> (Batch, 64)`
- **Encoder B ($64 \times 64$ 3D Depth Matrix):** `Conv2d(1, 16) -> Conv2d(16, 32) -> Conv2d(32, 64) -> Linear -> (Batch, 64)`

The input to the GRU remains `(Batch, 64 + 32 = 96)`, so the shapes of `gru_cell` and all downstream dense layers **do not change at all**.

### 2.2 The Catch: Latent Space Alignment (Why Naive Freezing Fails & How to Solve It)
If you train the model with Encoder A (`1D-CNN`), freeze the GRU + Dense layers, and suddenly plug in a randomly initialized Encoder B (`2D-CNN` for $64 \times 64$), the new `2D-CNN` will initially output random vectors $z_{\text{spatial}}$.
- Even if you train with PPO while the GRU is frozen, RL policy gradients have high variance and are very slow at teaching a new CNN how to speak the exact "64-dimensional language" the frozen GRU expects.

**There are two proven ways to make swapping the input encoder work smoothly:**

#### Method 1: Cross-Modal Distillation / Feature Alignment (Fastest & Cleanest)
When you transition from the 1D simulator to the $64 \times 64$ 2.5D simulator, the simulator can momentarily generate **both** the 90-ray 1D observation ($o_{\text{1D}}$) and the $64 \times 64$ depth matrix ($o_{\text{2D}}$) for the exact same agent position:
1. Freeze the trained `1D-CNN` (Teacher), `GRU`, and `Dense` heads.
2. Initialize the new `2D-CNN` (Student).
3. Collect rollouts and minimize Mean Squared Error (MSE) between the two encoders' 64-d outputs:
   $$\mathcal{L}_{\text{align}} = \left\| \text{Encoder}_{\text{2D}}(o_{\text{2D}}) - \text{stop\_grad}\big(\text{Encoder}_{\text{1D}}(o_{\text{1D}})\big) \right\|_2^2$$
4. In just **5–10 minutes of supervised training**, the new $64 \times 64$ `2D-CNN` learns to map 2D depth images into the **exact same 64-d coordinate space** that the frozen GRU already understands!
5. After alignment, unfreeze everything with a small learning rate (`1e-4`) so the GRU can exploit new 3D vertical features (like crawling under pipes or stepping over low boxes).

#### Method 2: Two-Phase Freeze-Then-Unfreeze (If Teacher Input Is Not Available)
If you don't have paired inputs:
1. **Normalize the Bottleneck:** Add `nn.LayerNorm(64)` at the output of `spatial_encoder` *now* (before Phase 1 training). This forces $z_{\text{spatial}}$ to stay on a well-behaved unit distribution ($\mu=0, \sigma=1$).
2. **Stage 1 (Warmup):** Swap `spatial_encoder`, **freeze** `gru_cell` and `actor_backbone`, and train only `spatial_encoder` + `critic_head` with PPO for ~50k steps.
3. **Stage 2 (Joint Fine-tune):** Unfreeze `gru_cell` and `actor_backbone` at $0.1\times$ the learning rate.

> [!IMPORTANT]
> **Actionable Design Rule for Your Code:** If you want to swap input layers later, keep `spatial_encoder` as a swappable module that **always outputs `(Batch, 64)` followed by `nn.LayerNorm(64)`**, and keep `ego_encoder` separate.

---

## 3. Perception & Environment Trade-Offs (To Discuss With Your Advisor)

You are considering two physical demo environments:
- **Scenario 1:** A physical maze built from **pipes or boxes** (static 3D geometry with varying heights).
- **Scenario 2:** Navigating among a **crowd of people** (dynamic moving obstacles).

Here is how the three perception options compare across both scenarios on your **8 GB VRAM laptop**:

| Dimension | Option A: 1D Depth Slice (`1 × 90` rays) | Option B: 2.5D Multi-Layer (`3 × 90` rays) | Option C: Full Depth Matrix (`64 × 64` via 2.5D Column Projection) |
| :--- | :--- | :--- | :--- |
| **How it works in Sim** | 90 horizontal 2D rays | 90 horizontal 2D rays evaluated at 3 heights (floor, mid, high) | 64 horizontal 2D rays projected vertically into 64 pixels using wall `(z_min, z_max)` + floor/ceiling |
| **How it works on Real Camera** | Monocular depth $\rightarrow$ take `min()` of each vertical column in robot height band | Monocular depth $\rightarrow$ average/min across 3 horizontal bands | Monocular depth $\rightarrow$ resize directly to $64 \times 64$ |
| **Box Maze (Walls & Boxes)** | ✅ Works (if camera takes vertical column `min`) | ✅✅ Works + distinguishes low box vs full wall | ✅✅ Works + sees full box shape and floor plane |
| **Pipe Maze (Thin / Overhangs)** | ⚠️ Fails if robot must pass *under* a pipe vs *stop* at a pipe | ✅ Works (top band hits pipe, bottom bands clear) | ✅✅ Best (sees exact gap height) |
| **Crowd of People** | ✅ Works (people are moving cylinders in 2D) | ✅ Works | ✅✅ Works + distinguishes legs/torso vs floor |
| **VRAM Buffer (`256 envs, 128 steps, 4 agents`)** | ~330 MB (`float32`) | ~500 MB (`float32`) | **~550 MB if stored as `uint8` depth** (or ~2.2 GB as `float16`) |
| **Training Speed (8GB Laptop)** | **Fastest (1.0× baseline)** | **Very Fast (~0.85×)** | **Moderate (~0.25–0.4×)** due to `Conv2d` backprop |
| **Engineering Effort** | **Low** (Current engine) | **Low-Medium** (Add `z_min, z_max` to segments) | **Medium** (Add C++ column projection + `Conv2d` encoder) |

### Recommended Strategy for Advisor Discussion
Because you proved in Section 2 that **you can swap the Spatial Encoder later via Teacher-Student Distillation**:
1. **Start MARL development on Option A (`1 × 90` 1D scan)** so you can iterate 4× faster on multi-agent rewards, MAPPO, and communication.
2. **Add `(z_min, z_max)` height attributes** to map obstacles when building the C++ batch engine.
3. Once MARL coordination works in 2D, **distill/upgrade the encoder to Option C ($64 \times 64$ depth matrix)** for the camera-based pipe/box maze demo!

---

## 4. Required Changes Across the Project (What Stays vs. What Changes)

### 4.1 What Stays (85% Reuse)
- Clean architecture layering (`core` $\rightarrow$ `entities` $\rightarrow$ `services` $\rightarrow$ `world` $\rightarrow$ `envs`).
- Continuous 60 Hz kinematics + 10 Hz decision rate (`action_repeat = 6`).
- Late-fusion `RecurrentActorCritic` core (`GRUCell(96, 128)` + Multi-Discrete/Continuous actor heads).
- Tiered curriculum manager, milestone checkpointing, and collapse rollback in [`train.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/train.py).
- Pygame live telemetry viewer ([`play.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/play.py)) for debugging.

### 4.2 Required Fixes to Existing Code (Immediate — Week 1)
1. **Fix Recurrent PPO Hidden-State Off-by-One ([`src/training/ppo.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/training/ppo.py#L166-L211)):**
   - Store `hidden` *before* `model.get_action(obs, hidden)` in `RolloutBuffer`, not `next_hidden`.
2. **Fix Timeout Value Bootstrapping ([`src/training/ppo.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/training/ppo.py#L80-L91)):**
   - Separate `terminated` (goal/death $\rightarrow$ bootstrap $0$) from `truncated` (timeout $\rightarrow$ bootstrap $V(s_{t+1})$).
3. **Batch Sequence Chunks in `train_epoch()` ([`src/training/ppo.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/training/ppo.py#L248-L296)):**
   - Process $8\text{--}16$ sequence chunks per backward pass instead of `batch_size = 1` chunk at a time.
4. **Raycast Once Per Decision Step ([`src/engine/world.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/engine/world.py#L113)):**
   - Move `_update_perceptions()` out of the 60 Hz substep loop so rays are only cast on the final substep (10 Hz). Immediate $3\text{--}5\times$ speedup.
5. **Add `nn.LayerNorm(64)` to `spatial_encoder` ([`src/models/actor_critic.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/models/actor_critic.py#L50-L59)):**
   - Prepares the bottleneck so you can freeze the GRU and swap between 1D rays and $64 \times 64$ depth later.

### 4.3 Required New Modules for MARL & High-Performance Engine

| Area | Required Change | Trade-off |
| :--- | :--- | :--- |
| **1. Task & Environment** | Remove perfect `polar_target` goal oracle (or make it noisy/line-of-sight only). Add **Multiple Agents ($N=2\text{--}4$)**, **Targets/Victims** (`is_objective` channel), and **Dynamic Crowd Circles** (moving obstacles with simple social-force/waypoint motion). | Removing the goal oracle makes the problem harder initially, so Tier 0 should keep a short-range beacon and fade it out in Tier 1–2. |
| **2. C++ Batched Engine (`BatchWorld`)** | Move the environment step loop, agent arrays, wall segments, and multi-env stepping into C++ (`BatchWorld::step_batch`) using **Structure-of-Arrays (SoA)** and **OpenMP** parallelization across CPU cores. | Requires disciplined C++ memory design, but eliminates Python loop overhead and increases speed from ~1,000 steps/s to 50,000–100,000+ steps/s. |
| **3. MARL Algorithm (MAPPO)** | Upgrade `PPOTrainer` to **MAPPO**: decentralized actors share weights (`RecurrentActorCritic`), while the centralized critic takes global state (all agent positions + target states) during training. | Centralized critic only runs during training; deployed robots still run 100% decentralized. |
| **4. Inter-Agent Communication** | Replace the unused tail padding with a **Permutation-Invariant Attention Module** over received teammate messages (with configurable range, wall occlusion, and packet drop probability). | Slightly increases network size (~50K params), but provides your strongest academic/defense research contribution. |
| **5. Sim-to-Real Depth Noise** | Add domain randomization to simulated depth: multiplicative scale drift ($\times [0.8, 1.2]$), Gaussian noise, spatial smoothing, and random pixel/ray dropout. | Essential because monocular depth models (*Depth Anything V2*) produce smooth, scale-ambiguous depth maps rather than sharp laser hits. |

---

## 5. Questions to Ask Your Advisor at the Meeting

Bring these 4 concrete questions to your advisor to lock in the thesis rubric:

1. **Task Framing:**
   *"We want to reframe the tactical shooter arena into **Decentralized Multi-Robot Search, Navigation & Reconnaissance in Occluded/Dynamic Environments (Pipe/Box Mazes & Crowds)**. This keeps 90% of our engine and recurrent MARL architecture while making it applicable to Search & Rescue and Defense Autonomy. Do you approve this framing?"*
2. **End-of-Semester 1 Deliverable (In 8 Weeks):**
   *"For the end of this semester, we propose delivering: (a) our C++ Batched Multi-Agent Simulation Engine benchmarked at 50k+ steps/s, (b) solved single-agent and 3-agent cooperative navigation/search with MAPPO compared against a classical exploration baseline, and (c) a live camera-to-depth perception bridge (`Depth Anything V2` $\rightarrow$ policy input). Is this a strong Semester 1 milestone?"*
3. **2D vs. 2.5D ($64 \times 64$) Perception Progression:**
   *"We designed our neural network with a modular bottleneck so we can train MARL policies fast using 1D depth scans first, and then distill/swap the input layer to a $64 \times 64$ 2.5D depth matrix while freezing the recurrent tactical core. Do you like this two-stage experimental design as a thesis contribution?"*
4. **Physical Demo Scope for Semester 2:**
   *"In Semester 2, for physical validation in a box/pipe maze or crowd, would you prefer a single physical edge prototype (Jetson Nano + camera running the trained policy in a real box/pipe maze) paired with multi-agent simulation, or multiple simpler robots?"*

---

## 6. Two-Person Division of Labor (8-Week Plan to Semester 1 Presentation)

| Week | Partner 1: C++ Engine, Maps & Depth Perception Bridge | Partner 2: Neural Architecture, MAPPO & Evaluation |
| :--- | :--- | :--- |
| **Week 1** | Fix raycast-every-substep waste & config bug. Design C++ `BatchWorld` SoA memory layout. | Fix PPO hidden-state off-by-one bug, timeout bootstrap, and minibatching. Add `LayerNorm(64)` bottleneck. |
| **Week 2** | Implement C++ `BatchWorld::step()` for batched envs with OpenMP + zero-copy NumPy buffers. | Complete Phase 1 navigation on Tier 2 mazes (>85% on held-out maps). Build evaluation script (SPL, collision rate). |
| **Week 3** | Add multi-agent support (`is_friend`, agent-agent collisions) and dynamic crowd obstacles (moving circles) to C++ engine. | Upgrade PPO to **MAPPO** (shared decentralized actor + centralized global critic). |
| **Week 4** | Build procedural box/pipe/corridor map generator + target/victim spawning (`is_objective`). | Remove/weaken goal oracle; train 3-agent cooperative search & navigation. Implement classical Frontier baseline. |
| **Week 5** | Add `(z_min, z_max)` height spans to C++ engine to render $64 \times 64$ 2.5D depth matrices + depth noise randomization. | Implement **Teacher-Student Encoder Distillation**: freeze GRU/Dense, train `Conv2d` ($64 \times 64$) to match `Conv1d` latent space. |
| **Week 6** | Build the real camera pipeline: Webcam/IMX219 $\rightarrow$ *Depth Anything V2 Small* $\rightarrow$ $64 \times 64$ depth tensor. | Add **Attention-Based Teammate Communication** with packet loss / range constraints. |
| **Week 7** | Benchmark C++ engine throughput (1 env vs 64 envs, Python vs C++ OpenMP). Test real camera in a box/pipe setup. | Run ablation experiments (1D vs $64 \times 64$ frozen-GRU transfer, comms vs no-comms, crowd density sweep). |
| **Week 8** | **Semester 1 Presentation & Industry Mentorship Pitch Package** (2-page brief + benchmark plots + live camera/sim demo). | **Semester 1 Presentation & TÜBİTAK / Industry Mentorship Applications.** |
