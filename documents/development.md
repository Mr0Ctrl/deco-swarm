# DECO-SWARM: Development Workflow & Engineering Standards

## 1. Technology Stack & Framework Choices

- **Deep Learning Framework:** **PyTorch**. Chosen for modular network composition, seamless recurrent state (GRU) handling, and clean separation between perception and radio feature branches.
- **RL Algorithm Implementations:** Industry-standard modular implementations (e.g., **CleanRL** or **Ray RLlib**) to avoid reinventing foundational PPO / MAPPO math.
- **Environment API Standards:** Standardized wrappers adhering strictly to **Gymnasium** (single-agent) and **PettingZoo** (multi-agent parallel environments).
- **C++ Acceleration & Zero-Copy Bridge:** **pybind11** utilizing NumPy C-API buffers for direct pointer access (zero-copy memory sharing) between C++ physics and Python PyTorch tensors.
- **Rendering & Visualization:**
  - Prototyping / Interactive Testing: **Pygame** (current implementation).
  - Production Demonstration / High FPS: **Raylib** (lightweight C++ engine).

---

## 2. Modular Neural Network Architecture

To optimize gradient flow and representation learning across heterogeneous sensory modalities, input tensors are processed using **Late Fusion**:

1. **Spatial Perception Branch (Raycasts):**
   - 90 rays $\times$ 7 channels processed through a 1D Convolutional block or dense feature extractor to capture spatial depth and obstacle patterns.
2. **Tactical & Communication Branch:**
   - Ego target vector, odometry, internal health/ammo status, and received tactical radio tokens processed via an MLP branch.
3. **Fusion & Recurrent Core (GRU):**
   - Latent representations are concatenated and passed through a Gated Recurrent Unit (GRU) to handle Partial Observability (POMDP), maintain obstacle memory, and prevent looping/spinning.
4. **Actor & Critic Heads:**
   - **Actor Head:** Outputs categorical distributions across the multi-discrete action branches (movement, rotation, trigger).
   - **Critic Head:** Estimates state value $V(s)$. In CTDE, the critic receives augmented global state information during training.

---

## 3. Curriculum Prototyping & Iteration Strategy

Reinforcement learning development is inherently empirical. The workflow emphasizes rapid prototyping and disposable baseline models:

- **Stage 1 (Proof-of-Concept Baseline):** Single agent in isolated mazes (Phase 1, 1v0). Focus is exclusively on verifying reward shaping, hyperparameter stability (learning rate, entropy coefficient, clip range), and GRU temporal recurrence.
- **Stage 2 (Expansion with Padding):** Moving to combat and multi-agent setups expands input/output shapes. The 640-float observation space includes reserved padding at the tail to minimize structural retraining shocks.

---

## 4. Performance Roadmap & Throughput Targets

To enable feasible training iterations on standard laptop hardware, the simulation must sustain **10,000+ environment steps per second**:

```text
[Phase A: Pure Python] ──► [Phase B: Hybrid C++ Pybind11] ──► [Phase C: Full C++ Simulation]
• Logic & reward validation • Raycast & collision ported      • Full headless simulation in C++
• Current baseline          • Zero-copy shared memory          • Python handles purely PyTorch
• ~500 - 2,000 steps/sec     • 10,000+ steps/sec                • Maximum batch throughput
```
