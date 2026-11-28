# SIM_CONTRACT v0

Version: `v0`. Any change to the observation or action layout bumps the version.
Checkpoints store this version.

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
   Per ray: `[distance / max_range_m, class one-hot (C = 8), hit_is_carrier]`.
   Classes: `none, wall, ally, enemy, own_flag, enemy_flag, own_base, enemy_base`.
2. Ego bearings (E = 2): `own_base`, `enemy_flag` (current position, even when carried or stray).
   Each target: `[cos(rel_angle), sin(rel_angle), distance / map_diagonal]`.
3. Self state (S = 10): `carrying_flag`, `own_flag_state` one-hot (home / carried / stray),
   `enemy_flag_state` one-hot (home / carried / stray), `respawning`, `tag_cooldown / cooldown_s`,
   `time_left / max_steps`.

Size: `R * (1 + C + 1) + 3 * E + S` = 90 * 10 + 6 + 10 = **916**.

## Step output
`obs, state, reward, terminated, truncated, info = env.step(key, state, actions)`
- `reward`: per agent, from `environment.rewards`.
- `terminated`: match over (time limit reached, or `score_limit` hit if not null).
- `truncated`: only used for artificial cuts (debugging, curriculum). Normally always False.
- `info`: per-agent game events (capture, pickup, return, tag, tagged), used for logging and Phase 2 rewards.