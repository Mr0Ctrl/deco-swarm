# Where we left off

_Last updated: 2026-11-28_

## Status
- **Phase 0.1** (Python + JAX env) done: uv, Python 3.12, JAX sees the GPU (`CudaDevice`), matmul benchmark in `scratch/`.
- **Phase 0.2** (repo skeleton) done: `pyproject.toml` + `uv.lock`, `src/deco_swarm/` with subpackages, smoke tests (import, devices, GPU backend). `uv run pytest` gives 3 passed.
  - Committed locally as `5969b29` on `mozkaya_new`, **not pushed**. (Claude made that commit by mistake. The user keeps it for now.)
- ROS no longer interferes: it's only sourced via the `ros` alias.

## Open items
- `CLAUDE.md` and `memory.md` are not committed yet (the user decides).
- Optional: add `runs/` to `.gitignore`. Add a trailing newline at the end of `.gitignore`.

## Next step
- **Phase 0.3 done:** `scratch/03_jax_basics.py` (jit, vmap, scan, static config, PRNG) works, and Claude explained what each piece is for in the project.
- **Config decisions (0.4):** **YAML** (replaces JSON and matches RS IR-DATA-01). Add `pyyaml` as an explicit dependency.
  - The roadmap and faz0 docs still say JSON. The user will update them.
  - YAML pitfalls: scientific notation needs a dot AND an exponent sign (`1.0e-5` and `5.0e+7` are fine; `1e-5` and `5.0e7` become strings). `50_000_000` is read as an int. Quote yes/no/on/off strings. Use `yaml.safe_load`.
  - **Rules = DeepMind FTW CTF (Jaderberg et al. 2018, arXiv:1807.01281)**, decided 2026-11-28. Claude listed config changes:
    - timing: 4 ticks/decision → 15 Hz, 4500 steps = 5 min
    - capture needs own flag at base
    - dropped flag becomes stray: own team touches → returns to base; enemy touches → picks up
    - carrier can tag; no half restriction; multi-hit shield; respawn at own base
    - point symmetry (`rotate_180`)
    - rewards: win/draw/loss +1/0/−1 plus the 13 game events with Quake weights
  - Not stated in the paper (stays null): tag range, cooldown, hits per tag, respawn delay, protection, auto-return timer, speeds.
  - Roadmap "dropped flag returns to stand" decision is now outdated. The user updates the docs.
- **Phase 0.4 done (2026-11-28):** we took only the config files from main's layout, converted to meters + FTW.
  - Files: `agent`, `environment`, `simulation`, `render`, `arenas/ctf_open`, `arenas/ctf_pillars`.
  - Claude deleted the old drafts (`game_rules`, `observation`, `rewards`) at the user's request.
  - The `render.yaml`/`ctf_open.yaml` swap was fixed by the user.
  - We continue our own path in `src/deco_swarm/` and do NOT adopt main's code.
- **Phase 0.5 done:** strict loader + 10 tests, `uv run pytest` gives 13 passed. Committed `b102c65`.
  - `pyyaml` added to pyproject.
- **Phase 0.6 done:** arena compiler + 9 tests, `uv run pytest` gives 22 passed. Committed `5839d34`.
- **Phase 0.7 done (lightweight):** the JaxMARL MAPPO-RNN MPE run works on our GPU. Real VRAM about 1 GB. Return went from -88 to about -47 within 0.8M steps, so it is learning.
  - The user doesn't want to read the source. Claude brings JaxMARL patterns into the code it gives (GAE now; rollout, GRU reset and minibatching in Phase 2).
  - No notes file needed.
- **Phase 0.8 done:** GAE + 4 tests, `uv run pytest` gives 26 passed. Committed `0af342b`. Claude gave GAE learning resources.
- **Phase 0.9 done:** `documents/SIM_CONTRACT.md` v0 written (not committed yet). Decisions:
  - +y up, angle CCW from +x (differs from main's +y down)
  - 15 Hz decisions
  - actions are multi-discrete [move 3, strafe 3, turn 3, fire 2]; fire is a no-op while the laser is disabled
  - obs = 90 rays × 10 + 6 ego + 10 self = 916
  - the 4500-step time limit counts as `terminated` (time_left is in obs, the match really ends) → no truncation fix needed in GAE
- **Phase 0 gate:** passed. Exception: the second laptop (RTX 3070) setup is not verified yet.
- **Phase 1 started (2026-11-28).** Detailed plan written to `documents/new/faz1_tr.md` at the user's request; follow it in order. Plan:
  - 1.1 kinematics
  - 1.2 collision (wall slide + agent push)
  - 1.3 raycast
  - 1.4 rules (pickup, capture, stray flag, respawn; no laser)
  - 1.5 env (reset/step/obs/auto-reset, MultiAgentEnv-like)
  - 1.6 pygame viewer (needs `pygame` dependency)
  - 1.7 SPS + determinism
- Phase 0 closed in commit `0361708`. The user approved the faz1 plan decisions (raycast entry rule, laser/respawn deferred to Phase 3, symmetry test, pygame).
- **1.1 done:** kinematics, 33 tests passed, commit `85dcb17`.
- **1.2 done:** collision, 42 tests passed, commit `3b1b316`.
- **1.3 done:** raycast, 48 tests passed, commit `e548afa`. Carried flags are invisible to rays; distances are in meters.
- **1.4 done:** state + rules, 57 tests passed, commit `b5f6cf0`.
- **1.5 done:** `CTFEnv`, 63 tests passed, commit `17a98e2`. Reward: zero-sum capture, plus pickup/return, plus win/loss at the end. Symmetry test included.
- **1.6 done:** viewer + play.py, 65 tests passed, commit `3c3fb70`. The user played it and it looks good.
- **1.7 done / Phase 1 closed:** 67 tests passed, commit `aededb2`. SPS on RTX 4060 is about 0.92M env-steps/s at 4096 envs (saturated). Lesson: if obs is unused, XLA removes the raycast as dead code and the number is fake.
  - Still open: the raycast note is not yet in SIM_CONTRACT. `tests/test_determinism.py` has an uncommitted change.
- **Phase 2 plan written:** `documents/new/faz2_tr.md` (at the user's request). Steps:
  - 2.1 experiment configs (deep-merge overrides + `train.yaml`)
  - 2.2 env: `opponents: none`, potential-based progress shaping, `LogWrapper`
  - 2.3 MultiCategorical + MLP actor-critic
  - 2.4 PPO `make_train` (+ toy-task test, `valid` mask)
  - 2.5 TensorBoard + flax checkpoint + resume
  - 2.6 1v0 MLP runs
  - 2.7 1D-CNN + GRU recurrent PPO
  - 2.8 AI watch mode
  - Gate: 1v0 ctf_pillars GRU ≥90% success.
  - Dependency decision pending: `tensorboardX` + `tensorboard` (no orbax, no distrax).
- **main vs ours (analyzed 2026-11-28):** `origin/main` has no new commits since the earlier review (it stops at the viewer). Main SPS measured by Claude on the same GPU:
  - main: about 0.37M at 4096 envs × 6 ticks, about 2.2M ticks/s
  - ours: about 0.92M × 4 ticks, about 3.7M ticks/s
  - Main has laser + respawn + carrier drop, `prev_action` in obs, a 2-layer raycast (solid + objects) and a 60 FPS camera viewer.
  - Main lacks reset randomness, reward fn, auto-reset, flat obs, strict config, determinism tests and symmetry checks.
  - **Found a flaw in ours:** a base circle blocks rays, so an enemy standing inside a base is invisible from outside. Proposed fix before Phase 2: main-style 2-layer rays (`SIM_CONTRACT` v1), and maybe add `prev_action` to obs (FTW uses it). Awaiting the user's decision.
- **2.0 done (observation v1):** the user applied raycast/state/env. Claude applied the rest at the user's explicit request: viewer, test_raycast, test_env, SIM_CONTRACT v1. 70 tests passed, viewer smoke run OK. Not committed yet (user's call).
  - Rays: solid group blocks the ray; flag and base groups are seen only if nothing solid is in front. 11 features per ray. `prev_action` is in the obs. Obs size 1010.
- **Remaining gaps vs main (after v1):**
  - Main has, we don't: laser + respawn + carrier drop (our Phase 3), `collided` flag (for a collision penalty), `in_own_base` HUD bit, viewer camera follow/pan + beam drawing.
  - Deliberate design differences: we use meters, +y up, 3×3×3×2 actions, 15 Hz, time-limit = terminated, bearings + time_left in obs, carried flag invisible, entry-only circle rule, strict config.
- **Integration branch idea (2026-11-28):** the user wants a merged branch with the friend's main. Claude dry-ran the merge (base `abcec45`):
  - Real path conflicts: only configs/{agent,environment,render,simulation}.yaml and tests/test_raycast.py (both sides added them). docs: we renamed them to previous/ and main modified them; git should carry the changes over to the renamed files.
  - The big risk is silent: main's src/configs, src/sim, src/rendering, play.py and tests merge cleanly, so we would end up with two parallel codebases.
  - Plan proposed: agree on a decision table with the friend → `integration` branch from ours → `git merge --no-ff origin/main` → resolve conflicts with ours and remove main's parallel modules → port main's laser/respawn/drop, `collided` and camera/beam viewer as separate commits → PR to main.
  - Waiting for the user and the friend to agree.

## Working rules
- The user does everything (code, files, git). Claude explains, reviews and answers, and acts only on explicit requests. This file is the one exception.
- Go phase by phase. One commit per sub-phase, made by the user.
- Guidance style: learn by writing, not by reading external source code. For each step Claude gives the goal, **why it's needed and where it's used in the project**, near-final code (in chat) and a short explanation. The user types it in. No quizzes or exercises.
