# SIM_CONTRACT v1

Version: `v1`. Any change to the observation or action layout bumps the version.
Checkpoints store this version.
- v0 → v1: rays split into solid / flag / base groups (bases no longer hide agents); previous action added to the observation (916 → 1010).

## Units and coordinates
- Meters, seconds, degrees. Pixels exist only in the viewer (`render.camera.pixels_per_meter`).
- Origin at bottom-left corner, +x right, +y up.
- Heading: degrees, counter-clockwise from +x, kept in [0, 360).
- The viewer flips y when drawing (screen y grows downward).

## Timing
- Physics: `simulation.physics.tick_hz` = 60 Hz.
- Decisions: every `environment.episode.action_repeat` = 4 ticks → 15 Hz (FTW).
- Episode: `environment.episode.max_steps` = 4500 decisions = 5 minutes.

## Agents
- Order: `["red_0", ..., "red_{n-1}", "blue_0", ..., "blue_{n-1}"]`, n = `team_size`.
- Arrays with an agent axis use this order. Team index: red = 0, blue = 1.

## Action space (per agent, multi-discrete)
| Index | Name    | Values                                |
| :---- | :------ | :------------------------------------ |
| 0     | move    | 0 = none, 1 = forward, 2 = back       |
| 1     | strafe  | 0 = none, 1 = left, 2 = right         |
| 2     | turn    | 0 = none, 1 = left (CCW), 2 = right   |
| 3     | fire    | 0 = no, 1 = yes                       |

- `fire` is ignored while `laser` is disabled (Phase 3), so the layout doesn't change later.
- Movement speed is `agent.body.linear_speed_mps`. Forward + strafe is normalized so the diagonal is not faster.

## Observation (per agent, egocentric, flat float32 vector)
1. Rays: `R = agent.perception.ray_count` rays spread over `fov_deg`, centered on the heading.
   Each ray reports three groups (F = 11 values per ray):
   - **Solid** (blocks the ray): `[solid_d / max_range_m, wall, ally, enemy, hit_is_carrier]`.
   - **Flag** (seen only if nothing solid is in front; does not block): `[flag_d / max_range_m, own_flag, opp_flag]`.
   - **Base** (same rule as flags): `[base_d / max_range_m, own_base, opp_base]`.
   - No hit in a group: distance = 1.0 and all class bits 0.
   - A circle (agent, flag, base) is seen only when the ray enters it from outside: an agent inside its own base does not see that base, but sees the flag in it.
   - A carried flag is not visible; `hit_is_carrier` marks its carrier.
2. Ego bearings (E = 2): `own_base`, `enemy_flag` (current position, even when carried or stray).
   Each target: `[cos(rel_angle), sin(rel_angle), distance / map_diagonal]`.
3. Self state (S = 10): `carrying_flag`, `own_flag_state` one-hot (home / carried / stray),
   `enemy_flag_state` one-hot (home / carried / stray), `respawning`, `tag_cooldown / cooldown_s`,
   `time_left / max_steps`.
4. Previous action (A = 4): `[fwd, left, turn]` in {-1, 0, 1}, `fire` in {0, 1}. Zero after reset.

Size: `R * F + 3 * E + S + A` = 90 * 11 + 6 + 10 + 4 = **1010**.

## Step output
`obs, state, reward, terminated, truncated, info = env.step(key, state, actions)`
- `reward`: per agent, from `environment.rewards`.
- `terminated`: match over (time limit reached, or `score_limit` hit if not null).
- `truncated`: only used for artificial cuts (debugging, curriculum). Normally always False.
- `info`: per-agent game events (capture, pickup, return, tag, tagged), used for logging and Phase 2 rewards.