## Agent guidance (merged)

This file is a merged, human-friendly agent document derived from
`.github/copilot-instructions.md`. It is intended as the canonical agent-facing
guide in the repo root.

Key points
- FastAPI backend in `backend/` with routers in `backend/routes/`.
- SQLAlchemy models in `backend/database.py`. Default SQLite is used unless
  `QUANTUM_TRADER_DATABASE_URL` is provided.
- Frontend in `frontend/` (Vite + React). See `frontend/package.json` for
  scripts (`dev`, `build`, `typecheck`, `test:frontend`).

Workflows and rules
- Do not add dev-only packages to `backend/requirements.txt`. Use
  `backend/requirements-dev.txt` for linters/test tooling.
- CI enforces this via `backend/scripts/check_dev_deps_in_runtime.py` and will
  fail PRs that install dev-only packages into runtime.

Where to look
- `backend/main.py`, `backend/database.py`, `backend/routes/`
- `backend/scripts/check_dev_deps_in_runtime.py`
- `frontend/package.json`, `ai_engine/feature_engineer.py`

If you want this merged into another file or branch, tell me the branch name
or target path and I will create a PR-ready commit.
