# DECO-SWARM: Environment Rules, Goals & Reward Specifications

## 1. Environment Overview & Time Horizon

The environment wrapper ([`src/envs/single_agent_env.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/single_agent_env.py)) interfaces the continuous kinematic physics engine with standard reinforcement learning APIs (PettingZoo / Gymnasium).

### Time Discretization & Action Repeat
- **Simulation Physics Tick ($\Delta t_{\text{phys}}$):** Runs at **60 Hz** ($\approx 0.0167$ s). This high frequency prevents tunneling through thin line segments and ensures stable collision restitution.
- **Agent Decision Frequency ($\Delta t_{\text{dec}}$):** Runs at **10 Hz** ($0.1$ s).
- **Action Repeat ($N=6$):** Every discrete agent action is held constant across $6$ physics substeps:
  $$\Delta t_{\text{dec}} = N \cdot \Delta t_{\text{phys}} = 6 \times \frac{1}{60} = 0.1\text{ s}$$
- **Episode Truncation Horizon:** Set to **500 decision steps** ($50$ simulation seconds). If the agent fails to reach the goal within this window, the episode truncates (`truncated = True`).

---

## 2. Goal Mechanism & Termination Conditions

Each map specifies a target area defined by coordinate $(x_g, y_g)$ and radius $r_g$:

```yaml
goal:
  position: [800.0, 800.0]
  radius: 32.0
```

### Success Condition
An episode terminates with success when the agent's bounding circle intersects the goal perimeter:
$$\| \mathbf{p}_{\text{agent}} - \mathbf{p}_{\text{goal}} \|_2 \le r_{\text{agent}} + r_{\text{goal}}$$

- **Termination Rule:** Upon intersection, `terminated = True` triggers immediately, awarding the terminal bonus.
- **Why immediate termination?** In Phase 1 navigation, immediate termination prevents the agent from lingering inside the goal area to collect redundant positive rewards, keeping gradient updates clean.

---

## 3. Reward Function Mathematical Formulation

The step reward $r_t$ balances directional incentives, temporal urgency, and safety constraints:

$$r_t = r_{\text{progress}} + r_{\text{step}} + r_{\text{collision}} + r_{\text{goal}}$$

### A. Distance Progress Potential ($r_{\text{progress}}$)
$$r_{\text{progress}} = \alpha \cdot (d_{t-1} - d_t)$$
- **Definition:** $d_t = \| \mathbf{p}_t - \mathbf{p}_{\text{goal}} \|_2$ is the Euclidean distance from the agent to the goal at decision step $t$.
- **Scaling Coefficient:** $\alpha = 0.05$.
- **Behavior:** Positive when moving closer to the goal; negative when retreating. Because this operates on distance difference, it acts as a potential-based shaping reward that preserves optimal policy invariance.

### B. Step Time Penalty ($r_{\text{step}}$)
$$r_{\text{step}} = -0.01$$
- **Behavior:** Applied at every decision step. It encourages the agent to find the shortest topological path rather than wandering in open areas.

### C. Collision Penalty ($r_{\text{collision}}$)
$$r_{\text{collision}} = -0.05 \cdot \mathbb{I}_{\text{collision}}$$
- **Behavior:** If the circle-segment collision solver registers contact during any of the $6$ physics substeps, $\mathbb{I}_{\text{collision}} = 1$. This penalizes wall scraping and reckless corner bumping.

### D. Episodic Exploration / Novelty Bonus ($r_{\text{explore}}$)
$$r_{\text{explore}} = +0.02 \cdot \mathbb{I}_{\text{new\_cell\_visited}}$$
- **Behavior:** The 2D map is partitioned into $40 \times 40$ px cells (matching the agent's bounding diameter). When the agent enters a cell it has not previously visited in the current episode, it receives a $+0.02$ novelty bonus.
- **Why this works:** Prevents map memorization and corner-idling; encourages the agent to discover unmapped corridors and alternatives without overfitting to a single route. Counts reset at the start of each episode.

### E. Terminal Goal Bonus ($r_{\text{goal}}$)
$$r_{\text{goal}} = +10.0 \cdot \mathbb{I}_{\text{goal\_reached}}$$
- **Behavior:** A prominent positive sparse return awarded when the agent satisfies the success condition.

---

## 4. Sub-Curriculum Rules & Promotion Criteria

Defined in [`src/envs/curriculum.py`](file:///home/mrctrl/Workspace/projects/deco-swarm/src/envs/curriculum.py), the sub-curriculum manages progressive difficulty across three tiers:

| Tier | Difficulty Category | Map Files | Objective |
| :--- | :--- | :--- | :--- |
| **Tier 0** | Open Navigation | `faz_1v0.1.yaml` – `faz_1v0.5.yaml` | Learn forward thrust, steering towards polar target, stopping. |
| **Tier 1** | Obstacles & Dead-Ends | `faz_1v1.1.yaml` – `faz_1v1.5.yaml` | Learn wall avoidance, escaping corners, basic raycast integration. |
| **Tier 2** | Occluded Mazes | `faz_1v2.1.yaml` – `faz_1v2.6.yaml` | Full maze traversal requiring GRU temporal memory to solve deep cul-de-sacs. |

### Progression Dynamics
- **Evaluation Window:** Rolling window of the most recent **20 episodes** on the active map.
- **Promotion Threshold:** $\ge 85\%$ success rate.
  - Advances to the next map within the tier.
  - When the final map of a tier is mastered, the agent is promoted to the next tier.
- **Demotion Safeguard:** $\le 20\%$ success rate.
  - If the policy collapses upon entering a harder tier, the curriculum demotes the agent to the previous map to re-stabilize weights.
- **Minimum Warmup:** At least $20$ episodes must be played on a new map before promotion/demotion evaluation begins.

---

## 5. Performance Metrics & Telemetry

During training and evaluation, the following metrics are tracked:

1. **Success Rate ($\%$):** Ratio of episodes terminating at the goal vs. timing out at $500$ steps.
2. **Episode Length ($\bar{T}$):** Mean steps required to reach the goal (lower is better).
3. **Cumulative Undiscounted Return ($G = \sum r_t$):** Overall quality of the trajectory.
4. **Collision Frequency:** Average number of collided frames per episode, tracking navigation safety.
5. **Curriculum Level:** Current tier and map index, indicating learning velocity.
