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

### 4.1 2-Layer Panoramic Semantic Raycasts (`90 × 12`)
Evenly spaced rays sweep across $[\theta - \text{FOV}/2,\; \theta + \text{FOV}/2]$ (e.g., `90` rays over `180°`). Each ray from origin $\mathbf{o}$ in unit direction $\hat{r}$ with vector $\mathbf{r} = d_{\max}\hat{r}$ solves a **2D parametric ray-segment intersection** against every wall segment $\mathbf{p}_1 \to \mathbf{p}_2$ ($\mathbf{s} = \mathbf{p}_2 - \mathbf{p}_1$):

$$\det = r_x s_y - r_y s_x, \qquad t = \frac{(p_{1,x} - o_x)s_y - (p_{1,y} - o_y)s_x}{\det}, \qquad u = \frac{(p_{1,x} - o_x)r_y - (p_{1,y} - o_y)r_x}{\det}$$

If $|\det| \ge 10^{-8}$ and both $0 \le t \le 1$ and $0 \le u \le 1$, the ray hits the segment at distance $t \cdot d_{\max}$.

To prevent floor zones (`bases` and `flags`) from blinding the agent to walls or enemies behind them, each ray evaluates **two non-interfering team-relative layers** plus its relative angle (`12` floats per ray):
- **Solid Layer (`4`):** `[solid_dist, is_wall, is_ally, is_enemy]` — stops only at the nearest solid wall or agent (`1.0` if nothing hit).
- **Flag Layer (`3`):** `[flag_dist, is_own_flag, is_enemy_flag]` — nearest unoccluded flag along the ray closer than `solid_dist` (`1.0` if none). Carried flags move with their carrier, so `solid_dist` and `flag_dist` align automatically on the same ray.
- **Base Layer (`3`):** `[base_dist, is_own_base, is_enemy_base]` — nearest unoccluded base pad edge closer than `solid_dist` (`0.0` when standing inside, `1.0` if none).
- **Ray Angle (`2`):** `[rel_cos, rel_sin]` — $(\cos \Delta\theta, \sin \Delta\theta)$ relative to the agent's heading.

### 4.2 Ego & HUD Scalars (`9` floats)
Matching DeepMind FTW's strict POMDP rules, agents receive **zero global GPS coordinates or base compasses** and **no redundant velocity vector** (since 2D kinematics have no inertia). Instead, the ego stream provides `9` compact scalars:
- **Previous Action (`4`):** `[prev_move_fwd, prev_strafe, prev_rotate, prev_tag]` — lets the recurrent core detect wall stalls and firing state across stochastic samples.
- **HUD & Match Status (`5`):** `[carrying_flag, own_flag_home, enemy_flag_home, in_own_base, tag_cooldown_norm]`.

### 4.3 Full v2 Observation Summary (`1,089` floats)

| Tensor | Shape | Count | Content |
| :--- | :--- | :---: | :--- |
| **Panoramic Rays** | `(90, 12)` | `1,080` | Solid (`4`: `dist, wall, ally, enemy`), Flag (`3`: `dist, own, enemy`), Base (`3`: `dist, own, enemy`), Angle (`2`: `cos, sin`) |
| **Ego & HUD** | `(9,)` | `9` | `prev_action(4)` + `carrying_flag`, `own_flag_home`, `enemy_flag_home`, `in_own_base`, `tag_cooldown_norm` |

---

## 5. Discrete-to-Continuous Action Mapping (`MultiDiscrete([5, 3, 2])`)

Following DeepMind FTW's discrete key-press formulation (where `0` means `No-Op` / keys released), the policy outputs three independent categorical branches:
- **Translation (`5`):** `0: No-Op (0, 0)`, `1: Forward (+1, 0)`, `2: Backward (-1, 0)`, `3: Strafe Left (0, -1)`, `4: Strafe Right (0, +1)`.
- **Rotation (`3`):** `0: No-Op (0)`, `1: Turn Left (-1)`, `2: Turn Right (+1)`.
- **Laser Tagger (`2`):** `0: No-Op`, `1: Fire` (1-hit instant forward beam when cooldown is ready).

Decoupling translation, rotation, and firing lets an agent strafe, aim, and fire simultaneously with only $5 + 3 + 2 = 10$ logits instead of $30$ joint combinations. In **Phase 3 / Stage 3**, a continuous communication head $m_t \in [-1, 1]^8$ runs in parallel.

---

**Navigation:** [Index](README.md) | **Previous:** [01. Configuration & Arenas](01_configuration_and_arenas.md) | **Next:** [03. Environment, Rewards & Curriculum](03_environment_rewards_and_curriculum.md)
