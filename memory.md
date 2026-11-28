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
  - **Bug to fix (user):** `render.yaml` holds the ctf_open map content, and `arenas/ctf_open.yaml` holds the old `ctf_open_v0`.
  - We continue our own path in `src/deco_swarm/` and do NOT adopt main's code.
- **Phase 0.5 in progress:** strict loader. Claude gave `config/schema.py` (nested frozen dataclasses mirroring the YAML), `config/loader.py` (`load_all`, `require`, `ConfigError`) and `tests/test_config.py` (10 tests). Prototype passed 10/10 in the scratchpad.
- Next: 0.6 arena compiler (maps → padded PyTree + rotate_180 symmetry check).
- 0.7 (JaxMARL) can run in parallel.

## Working rules
- The user does everything (code, files, git). Claude explains, reviews and answers, and acts only on explicit requests. This file is the one exception.
- Go phase by phase. One commit per sub-phase, made by the user.
- Guidance style: for each step Claude gives the goal, **why it's needed and where it's used in the project**, near-final code (in chat) and a short explanation. The user types it in. No quizzes or exercises.
