# 03. Environment Loop, Reward Shaping & Curriculum Learning

**Navigation:** [Index](README.md) | **Previous:** [02. Simulation & Perception](02_simulation_and_perception.md) | **Next:** [04. Agent Architecture & Training](04_agent_architecture_and_training.md)

---

## 1. The Episodic Macro-Step Loop

The RL environment wraps the 60 Hz physics engine into standard episodic `reset` and `step` transitions at **10 Hz**:

1. **Reset:** Restores agent spawn poses, clears episodic tracking state (step counter, visited-cell grid, latched flags), optionally samples a fresh arena from the curriculum, and returns the initial ego-centric observation.
2. **Macro-Step (`action_repeat = 6`):**
   - Decodes the agent's discrete action into continuous velocities.
   - Advances the physics simulation for up to `6` substeps.
   - **Latches collisions** across all substeps (if any substep hits a wall, the macro-step counts as a collision).
   - **Exits early** if the terminal objective condition (e.g., reaching the goal zone) is met mid-step, so the agent receives credit immediately without overshooting.
   - Computes the step reward, updates curriculum statistics if the episode ended, and returns `(next_obs, reward, terminated, truncated, info)`.

---

## 2. Termination vs. Truncation

Distinguishing *why* an episode ended is essential for unbiased value estimation:
- **Terminated (`terminated = True`):** The episode ended naturally because the agent succeeded (or failed via a terminal game rule). There is no future return after this state, so Generalized Advantage Estimation (GAE) must mask future bootstrapped value to **`0`**.
- **Truncated (`truncated = True`):** The episode simply hit `max_steps` (the clock ran out). The world did not physically end; the critic's estimate $V(s_{t+1})$ should still be bootstrapped for TD targets (while still resetting the rollout trajectory and recurrent memory for the next episode).

> [!IMPORTANT]
> Confusing `terminated` with `truncated`—or masking out bootstrap values on timeouts—teaches the critic that approaching a goal near the end of the episode suddenly has zero value, distorting policies near the time horizon.

---

## 3. Reward Shaping Philosophy

Pure sparse rewards ("`+100` at goal, `0` everywhere else") fail in occluded 2D mazes because random exploration almost never stumbles across the objective. DECO-SWARM combines **dense behavioral shaping** with a **time-scaled terminal bonus**:

$$R_t = r_{\text{progress}} + r_{\text{exploration}} + r_{\text{collision}} + r_{\text{time}} + r_{\text{terminal}}$$

1. **Potential-Based Progress ($r_{\text{progress}}$):**
   $$r_{\text{progress}} = w_{\text{prog}} \cdot (d_{t-1} - d_t)$$
   Rewards reducing Euclidean distance to the objective and penalizes retreating. Because consecutive steps telescope ($\sum (d_{t-1} - d_t) = d_0 - d_T$), an agent cannot farm infinite reward by oscillating back and forth.
2. **Episodic Spatial Exploration ($r_{\text{exploration}}$):**
   The arena is partitioned into coarse grid cells $\big(\lfloor x / \text{cell\_size} \rfloor,\; \lfloor y / \text{cell\_size} \rfloor\big)$ (e.g., `80 × 80 px`), with the starting cell pre-marked at `reset`. Entering a cell for the first time in the current episode grants a small novelty bonus (`+0.10`). This breaks dead-end hesitation and encourages agents to sweep around obstacles without changing the optimal shortest path.
3. **Collision Penalty ($r_{\text{collision}}$):**
   Applies `-0.15` once per macro-step if wall contact occurred in any substep—discouraging wall-scraping without making the agent afraid to navigate narrow corridors.
4. **Avoiding Living-Penalty Bleed ($r_{\text{time}}$ & $r_{\text{terminal}}$):**
   Applying a large negative reward on every single step causes newly initialized agents to accumulate huge negative returns before ever seeing a goal, collapsing value predictions. Instead, `time_penalty` defaults to `0.0`, and speed is rewarded upon completion via a **time-efficiency scaled terminal bonus**:
   $$r_{\text{goal}} = R_{\text{peak}} \cdot \max\!\left(r_{\min},\; 1 - \frac{t}{T_{\max}} \cdot \alpha_{\text{decay}}\right)$$
   Fast completions earn near `100%` of `R_peak`, while slow completions still earn at least `r_min` (e.g., `20%`), clearly separating success from a `truncated` timeout penalty (`-10.0`).

### Episode Telemetry Contract
Every step exposes structured diagnostics in `info` for logging and curriculum tracking: `is_success`, `step_count`, `dist_to_goal`, `collided`, `explored_cells`, `map_name`, and a per-component `reward_breakdown` (`progress`, `time`, `collision`, `exploration`, `terminal`).

### Extending to v2 Capture the Flag (CTF)
In multi-agent CTF (2v2, 3v3, and 5v5), the same shaping philosophy governs team competition:
- **Core Game Rules:** Each team defends its home base and flag pedestal. An agent picks up the enemy flag by entering its radius, carries it back to its own home base to score, and can be **tagged** by opponents (dropping the flag and triggering a brief respawn cooldown).
- **Zero-Sum Team + Individual Shaping:** Sparse team rewards for **flag capture** and **match victory/defeat** are paired with lightweight individual shaping for **flag pickup**, **tagging an enemy flag-carrier**, and **spatial exploration**—so agents learn coordinated offense, defense, and callouts without selfish reward hacking.

---

## 4. Progressive Curriculum & Domain Randomization

Training directly on complex multi-room mazes from scratch wastes millions of steps. Instead, arenas are organized into **progressive difficulty stages**:
- **Stage 0 (Open Control):** Wide arenas with minimal obstacles to master thrust, strafing, rotation, and polar alignment.
- **Stage 1 (Obstacles & Corners):** Boxes and angled dividers requiring smooth wall-sliding and raycast obstacle avoidance.
- **Stage 2 (Occluded Mazes):** Deep cul-de-sacs and blind corridors that cannot be solved without recurrent temporal memory (`GRU`).

Within the active stage, each episode **randomly samples** an arena on `reset` (domain randomization), preventing agents from memorizing fixed coordinates. A rolling window (e.g., `30` episodes) tracks success rate:
- **Promotion:** After at least `30` episodes in the current stage, reaching high mastery (e.g., `>= 97.5%` success) advances to the next stage and clears the rolling window.
- **Demotion:** If performance collapses to `<= 20%` on a non-zero stage, the curriculum steps back to re-stabilize.
- **Stage Locking:** An optional lock flag disables automatic promotion/demotion for controlled single-stage ablations.

In **v2 CTF**, this extends naturally to a **3-stage MARL curriculum**: Stage 1 (basic flag pickup & return navigation) -> Stage 2 (team play against scripted heuristic bots) -> Stage 3 (communication-enabled self-play against a historical opponent pool).

---

## 5. Parallel Batched Environments & Auto-Reset

To feed PPO with uncorrelated on-policy experience, $N$ independent arenas step synchronously in parallel. Whenever an arena finishes (`terminated` or `truncated`), it **auto-resets inline** on the same step:
- The returned `next_obs[i]` is already the **initial observation of the new episode**, so the next policy forward pass can run immediately without stalls.
- The **terminal metadata** (success flag, episode length, reward breakdown) is preserved in `info[i]` before resetting so logging and curriculum tracking remain exact.

In a pure-JAX implementation, this entire batched step + branchless auto-reset (`where(done, reset_state, next_state)`) compiles onto the GPU with zero host-device communication during rollouts.

---

**Navigation:** [Index](README.md) | **Previous:** [02. Simulation & Perception](02_simulation_and_perception.md) | **Next:** [04. Agent Architecture & Training](04_agent_architecture_and_training.md)
