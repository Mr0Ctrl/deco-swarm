# 02. Headless 2D Simulation & Ego-Centric Perception

**Navigation:** [Index](README.md) | **Previous:** [01. Configuration & Arenas](01_configuration_and_arenas.md) | **Next:** [03. Environment, Rewards & Curriculum](03_environment_rewards_and_curriculum.md)

---

## 1. The Headless Principle & Two-Timescale Execution

The simulation core has **zero graphical dependencies**. It advances positions, resolves collisions, and computes sensor readings purely in memory. Keeping simulation decoupled from rendering ensures that training never pays window, font, or drawing overhead.

A critical design choice is running physics and policy decisions on **two different timescales**:
1. **Physics & Collision at 60 Hz ($\Delta t = 1/60\text{ s}$):** Fast-moving agents taking large steps can "tunnel" straight through thin wall segments in a single frame. Stepping kinematics and collisions at 60 Hz keeps per-tick displacement (~`3 px` at `180 px/s`) much smaller than the agent's `10 px` radius, making tunneling impossible.
2. **Policy Decisions & Raycasting at 10 Hz (`action_repeat = 6`):** Evaluating a neural network and casting 90 rays every 16 ms is both unnecessary for tactical steering and computationally wasteful. Holding each decision for 6 physics substeps means perception only needs to be computed once at the end of the 6-tick macro-step—cutting raycast workload by **6×** with no loss in control fidelity.

> [!TIP]
> Always compute raycast perceptions **after** the final physics substep of a decision cycle rather than on every 60 Hz tick. Only kinematics and collision resolution belong inside the inner 6-step loop.

---

## 2. Continuous 2D Kinematics

Agents operate in a continuous 2D screen-style coordinate frame (`+x` right, `+y` down), where `0°` points along `+x` and `90°` points along `+y`, with heading $\theta \in [0^\circ, 360^\circ)$. Each decision supplies three normalized continuous control signals in $[-1, 1]$: forward/backward thrust ($u_{\text{fwd}}$), lateral strafe ($u_{\text{strafe}}$), and rotation rate ($u_{\text{rot}}$).

1. **Rotation:** Heading updates linearly by $\theta \leftarrow (\theta + u_{\text{rot}} \cdot \omega_{\max} \cdot \Delta t) \bmod 360^\circ$.
2. **Quake-Style Strafe Normalization:** If an agent simultaneously presses forward and strafe, the combined vector $(u_{\text{fwd}}, u_{\text{strafe}})$ would have magnitude $\sqrt{2} \approx 1.414$, granting a 41% diagonal speed exploit. Whenever $\|(u_{\text{fwd}}, u_{\text{strafe}})\|_2 > 1$, the command vector is normalized to unit length before scaling by `linear_speed`.
3. **Body-to-World Projection:** Using the unit forward vector $\hat{f} = (\cos\theta, \sin\theta)$ and perpendicular right vector $\hat{r} = (-\sin\theta, \cos\theta)$, the world velocity is $(u_{\text{fwd}}\hat{f} + u_{\text{strafe}}\hat{r}) \cdot v_{\max}$, and position advances by $\mathbf{v} \cdot \Delta t$.

---

## 3. Smooth Sliding Collisions

Rather than halting an agent dead when it touches a wall—which traps policies in corners—the engine uses **circle-to-segment sliding collision resolution**:

1. Given an agent circle at center $\mathbf{c}$ with radius $R$, and a wall segment from $\mathbf{p}_1$ to $\mathbf{p}_2$ with direction $\mathbf{d} = \mathbf{p}_2 - \mathbf{p}_1$:
   $$t = \text{clamp}\!\left(\frac{(\mathbf{c} - \mathbf{p}_1) \cdot \mathbf{d}}{\|\mathbf{d}\|_2^2},\; 0,\; 1\right), \qquad \mathbf{q} = \mathbf{p}_1 + t\mathbf{d}$$
   Here $\mathbf{q}$ is the closest point on the finite wall segment to the agent's center.
2. If the distance $\|\mathbf{c} - \mathbf{q}\|_2 < R$, the agent is penetrating the wall. It is pushed outward along the surface normal $\hat{n} = (\mathbf{c} - \mathbf{q}) / \|\mathbf{c} - \mathbf{q}\|_2$ just enough to sit flush at distance $R$, and a collision flag is recorded. *(Edge guards: skip degenerate zero-length segments where $\|\mathbf{d}\|_2^2 = 0$, and fall back to $\hat{n} = (1, 0)$ if $\|\mathbf{c} - \mathbf{q}\|_2 \le 10^{-6}$ so division by zero never occurs.)*
3. Because pushing out along only the normal preserves tangential motion, the agent **glides smoothly along walls**. Running **2 relaxation passes** per physics tick ensures that pushing out of one wall at an acute corner does not leave the agent embedded in the adjacent wall.

Objective and zone checks use **circle-circle overlap** ($\|\mathbf{c}_{\text{agent}} - \mathbf{c}_{\text{zone}}\|_2 \le R_{\text{agent}} + R_{\text{zone}}$) rather than requiring the agent's center point to reach the zone. In multi-agent arenas, the same push-out logic extends naturally to **circle-to-circle agent collisions** (pushing overlapping agents apart symmetrically by half the penetration depth each).

---

## 4. Ego-Centric Partial Observability (POMDP)

Agents **never receive global `(x, y)` map coordinates**. Global coordinates invite policies to memorize specific map layouts and break completely when spawning on randomized arenas or opposite team sides. Instead, every observation is expressed strictly in the **agent's own body frame**:

### 4.1 Panoramic Semantic Raycasts
Evenly spaced rays sweep across $[\theta - \text{FOV}/2,\; \theta + \text{FOV}/2]$ (e.g., `90` rays over `180°`). Each ray from origin $\mathbf{o}$ in unit direction $\hat{r}$ with vector $\mathbf{r} = d_{\max}\hat{r}$ solves a **2D parametric ray-segment intersection** against every wall segment $\mathbf{p}_1 \to \mathbf{p}_2$ ($\mathbf{s} = \mathbf{p}_2 - \mathbf{p}_1$):

$$\det = r_x s_y - r_y s_x, \qquad t = \frac{(p_{1,x} - o_x)s_y - (p_{1,y} - o_y)s_x}{\det}, \qquad u = \frac{(p_{1,x} - o_x)r_y - (p_{1,y} - o_y)r_x}{\det}$$

If $|\det| \ge 10^{-8}$ and both $0 \le t \le 1$ and $0 \le u \le 1$, the ray hits the segment at distance $t \cdot d_{\max}$. In multi-agent CTF, the same ray also checks analytic **ray-circle intersections** against allies, enemies, flags, and bases, keeping only the **nearest hit** per ray:
- **Normalized Depth:** `hit_distance / max_distance` in $[0, 1]$ (`1.0` when nothing is hit).
- **Semantic Identity Flags:** One-hot / multi-flag indicators identifying what the ray struck (`wall`, `ally`, `enemy`, `objective/flag/base`).
- **Relative Ray Geometry:** $(\cos \Delta\theta, \sin \Delta\theta)$ encoding the ray's angle relative to the agent's heading so spatial convolutions know where each ray points.

### 4.2 Ego-Polar Reference Bearing
To give agents a compass toward a known reference point (such as a navigation goal, or in CTF a team's home base / flag zone), the engine computes a 2-float ego-polar vector:
- **Normalized Distance:** $\min(1,\; \|\mathbf{p}_{\text{target}} - \mathbf{p}_{\text{agent}}\|_2 / \text{diag}_{\text{arena}})$.
- **Signed Relative Angle:** The shortest angular difference between the target bearing and the agent's heading, wrapped to $[-180^\circ, 180^\circ)$ and divided by $180^\circ$ to lie in $[-1, +1]$ (negative = target is to the left, positive = right, `0` = straight ahead).

### 4.3 Proprioception & Full Observation Layout
Completing the observation are low-dimensional scalar signals: current normalized velocities, the previous motor action (helping the recurrent core infer whether a command was blocked by a wall), and agent-specific status vitals (health, cooldowns, or flag-carrying state).

In the 640-float baseline layout, all streams pack into a single normalized vector in `[-1.0, 1.0]`:

| Slice | Shape / Count | Block | Per-Element Content |
| :--- | :--- | :--- | :--- |
| `[0 : 630]` | `90 × 7` | **Panoramic Raycasts** | `[norm_dist, is_wall, is_ally, is_enemy, is_objective, rel_cos, rel_sin]` |
| `[630 : 632]` | `2` | **Ego-Polar Bearing** | `[norm_dist, norm_angle]` |
| `[632 : 635]` | `3` | **Current Velocity** | `[move_fwd, strafe, rotate_dir]` |
| `[635 : 638]` | `3` | **Previous Action** | `[prev_move_fwd, prev_strafe, prev_rotate_dir]` |
| `[638 : 640]` | `2` | **Self Status** | `[norm_health, flag_or_ammo_state]` |

---

## 5. Discrete-to-Continuous Action Mapping

Rather than predicting noisy raw continuous torques, the policy outputs **decoupled multi-discrete choices** that map cleanly to the continuous kinematic inputs:
- **Translation Branch (`5` choices):** Stop `(0, 0)`, Forward `(+1, 0)`, Backward `(-1, 0)`, Strafe Left `(0, -1)`, Strafe Right `(0, +1)`.
- **Rotation Branch (`3` choices):** Hold `0`, Turn Left `-1`, Turn Right `+1`.

Decoupling translation from rotation lets an agent combine forward movement or strafing with smooth turning simultaneously without exploding the action space ($5 + 3 = 8$ logits instead of $5 \times 3 = 15$ joint combinations). In Capture the Flag, tactical actions (such as tagging/dash or emitting a communication signal) simply attach as additional parallel heads.

---

**Navigation:** [Index](README.md) | **Previous:** [01. Configuration & Arenas](01_configuration_and_arenas.md) | **Next:** [03. Environment, Rewards & Curriculum](03_environment_rewards_and_curriculum.md)
