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
  - YAML pitfalls: write `1.0e-5`, not `1e-5`; quote yes/no/on/off strings; use `yaml.safe_load`.
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
- **1.7 in progress:** Claude gave `scratch/sps.py` and `tests/test_determinism.py` (2 tests, bit-exact).
  - Claude measured SPS on the RTX 4060 (ctf_pillars, 128 steps):
    - 1024 envs: about 0.79M SPS
    - 4096 envs: about 0.92M SPS (saturated)
    - 8192 envs: about 0.90M SPS
  - Lesson: if obs is not used in the output, XLA drops the raycast as dead code and the number is fake (about 7.6M).
- Next: Phase 1 gate check (SIM_CONTRACT raycast note), then Phase 2.

## Working rules
- The user does everything (code, files, git). Claude explains, reviews and answers, and acts only on explicit requests. This file is the one exception.
- Go phase by phase. One commit per sub-phase, made by the user.
- Guidance style: learn by writing, not by reading external source code. For each step Claude gives the goal, **why it's needed and where it's used in the project**, near-final code (in chat) and a short explanation. The user types it in. No quizzes or exercises.
