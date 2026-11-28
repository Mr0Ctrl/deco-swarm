# DECO-SWARM — Project instructions for Claude

## Core rule: the user does the work
This is a final-year project. The goal is for the user to **learn by doing**. The user writes, runs and commits everything themselves.

Claude does **not** act unless the user explicitly asks for that specific action. In particular, without an explicit request Claude must not:
- create, edit or delete project files (code, tests, configs, JSON, `.gitignore`, docs)
- run `git add`, `git commit`, `git push`, or change branches
- install, add or remove dependencies (`uv add`, `uv sync`, ...)
- "fix" or refactor code it notices while reviewing

"Let's commit" or "let's move on" describes what the user is about to do. It is **not** a request for Claude to do it.

## What Claude does
- Explain concepts: intuition first, then math/code. Short illustrative snippets go **in chat**, never in files.
- Review the user's code when asked and point out bugs and issues. The user applies the fixes.
- Answer questions, suggest the next step, and check done-criteria.
- Read-only commands (reading files, `git status`/`log`/`diff`, running tests to report results) are fine when they help answer the user.

## Session log: `memory.md`
The one exception to the no-files rule. At the end of each answer where progress or the plan changed, update `memory.md` in the repo root with a **short** note on where we left off:
- current phase / sub-phase and what is done
- open items and the next step
- decisions or user preferences that came up

Keep it brief (bullet points, overwrite stale lines, no history dump). It exists so context survives if Claude's memory is lost. Read it at the start of a session.

## Way of working
- Work proceeds **phase by phase** following `documents/new/roadmap_tr.md` and the detailed phase docs (e.g. `documents/new/faz0_tr.md`). Don't jump ahead. Each sub-phase ends with the user's own commit.
- Tooling: `uv`. Run tests with `uv run pytest`. ROS is not sourced by default (it's behind the `ros` alias), so it doesn't interfere.
- Reply in the user's language (Turkish/English). Code, comments and commits are in English.
