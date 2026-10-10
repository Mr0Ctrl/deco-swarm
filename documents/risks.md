# DECO-SWARM: Technical Risks & Mitigation Strategies

## 1. Hard Reset Risks (Requiring Full Policy Retraining)

Changes to core simulation parameters invalidate pre-trained weights and force a cold restart:

- **Observation Shape Alterations:** The input layer weight matrix is tightly coupled to observation dimensionality. Adding or modifying sensor channels breaks model compatibility.
  - *Mitigation:* The observation vector is pre-dimensioned (640 floats) with reserved zero-padding channels for future sensors and communication tokens.
- **Action Space Restructuring:** Altering discrete action branches or velocity mappings breaks the actor head output layer.
  - *Mitigation:* The multi-discrete action space was defined upfront with independent movement, rotation, and firing branches.
- **Decision Frequency / Timestep ($\Delta t$) Shifts:** Transitioning a policy trained at 10 Hz to 60 Hz cannot be solved by fine-tuning; the temporal discounting factor ($\gamma^{\Delta t}$) and physical momentum change drastically, inducing policy chatter or instability.
  - *Mitigation:* The 60 Hz physics / 10 Hz decision cycle (Action Repeat = 6) is locked as an invariant rule.
- **Radical Reward Scaling Changes:** Shifting reward scales or reversing signs mid-training destabilizes the Critic value network and collapses the policy.
  - *Mitigation:* Normalize rewards and validate scale ranges during Phase 1 prior to long-term runs.

---

## 2. Training Pathologies & Local Minima

- **Reward Hacking & Passive Camping:** Agents optimize scalar returns rather than human intent. High death penalties cause agents to hide in spawn corners; excessive ammo penalties cause pacifism; naive Euclidean distance rewards lead to wall-facing deadlocks.
  - *Mitigation:* Potential-based reward shaping, step-time penalties, and waypoint/raycast-aligned progress bonuses.
- **Partial Observability (POMDP) & Spinning Behavior:** When targets exit the 180° field-of-view, memoryless policies rotate continuously in place to re-acquire the target.
  - *Mitigation:* Persistent relative polar target vector ($r, \Delta\theta$) combined with GRU recurrent memory cells.
- **Sparse Reward Bottlenecks:** In complex mazes, random exploratory actions have near-zero probability of reaching the objective, causing training stagnation for millions of steps.
  - *Mitigation:* Curriculum learning with incremental maze complexity (`faz_1v0.1` to `faz_1v2.6`).
- **Policy Collapse & Catastrophic Degeneration:** An agent may reach high proficiency and suddenly destabilize due to gradient spikes, overwriting checkpoint weights with a degraded policy.
  - *Mitigation:* The three-rule protection system in `train.py`: continuous peak performance tracking (`model_best.pt`), automatic rollback with learning rate annealing when severe degradation is detected, and protected final model preservation (`model_final.pt`).

---

## 3. Multi-Agent (MARL) & Communication Pitfalls

- **Communication Collapse:** When agents are given unconstrained float communication channels, early exploratory noise causes receiving networks to zero-out communication weights, ignoring teammates permanently.
  - *Mitigation:* Transmit structured, discrete tactical tokens (e.g., detected coordinates, status flags) rather than unconstrained continuous embeddings.
- **Agent Permutation Problem:** Presenting teammates as a rigid ordered list (`[teammate_1, teammate_2]`) fails when teammates die or list order shifts.
  - *Mitigation:* Permutation-invariant aggregation (Deep Sets, MaxPooling or Attention over teammate states).
- **Credit Assignment Dilemma:** In 3v3 team scenarios, an agent executing proper positioning may be penalized if a teammate blunders, incentivizing selfish behavior.
  - *Mitigation:* Centralized Critic in MAPPO and individualized reward decomposition (e.g., individual line-of-sight and territory control bonuses alongside team victory).
- **Rock-Paper-Scissors Cycles & Catastrophic Forgetting:** In self-play, teams may cycle through counter-strategies while forgetting foundational tactical competencies.
  - *Mitigation:* League training maintaining historical policy checkpoints in an opponent pool.

---

## 4. Hardware & Implementation Bottlenecks

- **pybind11 Deep-Copy Serialization Penalties:** Transferring observation tensors between C++ and Python via deep copies destroys simulation throughput.
  - *Mitigation:* Mandatory zero-copy memory mapping via shared NumPy buffer pointers.
- **Continuous Kinematic Raycast Load:** Computing 90 rays across multiple agents against arbitrary line segments scales heavily.
  - *Mitigation:* Implement spatial partitioning (grid hashing) to prune segment intersection candidates.
