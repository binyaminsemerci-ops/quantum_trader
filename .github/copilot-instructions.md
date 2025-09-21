This repository is a full‑stack trading system combining a FastAPI backend, a React + Vite frontend (currently being migrated to TypeScript), and optional AI/model code under `ai_engine`.

Keep guidance concise and specific — follow these project rules when making edits or generating code.

- Architecture at-a-glance
  - Backend: FastAPI app in `backend/` (entry: `backend/main.py`). Routes live under `backend/routes/` and use SQLAlchemy models in `backend/database.py`. Dockerfile uses `uvicorn backend.main:app` for production.
  - Frontend: Vite + React in `frontend/`. Dev: `npm --prefix frontend run dev`. Build: `npm --prefix frontend run build`. Tests: `npm --prefix frontend run test:frontend` (uses Vitest). Type checking: `npm --prefix frontend run typecheck`.
  - AI & models: `ai_engine/` contains model artifacts (`models/`), feature engineering (`ai_engine/feature_engineer.py`), and an experimental FastAPI wrapper at `ai_engine/backend/main.py`.

- Developer workflows & useful commands
  - Run backend locally: set PYTHONPATH to repo root, install `requirements.txt`, then `uvicorn backend.main:app --reload --port 8000` (or use Docker described in `backend/Dockerfile`).
  - Frontend dev: from repo root: `npm --prefix frontend run dev` (Vite dev server). Use the `frontend` npm scripts in `frontend/package.json`.
  - Run frontend typecheck and tests during migration: `npm --prefix frontend run typecheck` and `npm --prefix frontend run test:frontend`.
  - CI: GitHub workflows live in `.github/workflows/` (see `ci.yml` and `frontend-ci.yml`) — follow their steps when adding new checks.

- Project conventions and gotchas
  - TypeScript migration: follow `MIGRATION_PLAN.md`. When replacing `.jsx` with `.tsx`, the repo uses re-export stubs created by `scripts/generate-reexports.js`. Keep `allowJs` disabled until migration completes.
  - Backend DB: `backend/database.py` creates a SQLite DB under `backend/data/trades.db`. Don't assume a remote DB in dev.
  - PYTHONPATH: many backend modules expect `PYTHONPATH=/app` or repo root on sys.path (see Dockerfile). When running tests or scripts, ensure the repo root is importable.
  - Tests: Python tests use pytest; frontend tests use Vitest. Backend unit tests may expect local SQLite files (`db.sqlite3` or `backend/data/trades.db`). Prefer ephemeral test DBs during changes.

- Patterns and examples (copy/paste friendly)
  - Add a new backend route: create a router under `backend/routes/`, register with `app.include_router(..., prefix="/yourprefix")` in `backend/main.py`.
  - Read/write DB session (pattern): import `get_db` from `backend/database.py` and use it as a FastAPI dependency.
  - Frontend import migration pattern (example from `MIGRATION_PLAN.md`): replace `MyComp.jsx` with `export { default } from './MyComp.tsx'` and add a typed `MyComp.tsx`.

- Integration and external deps
  - Binance: planned integration uses `binance` library (see `requirements.txt`) and a wrapper in `backend/routes/binance.py`.
  - Model artifacts: `models/` holds `xgb_model.pkl` and `scaler.pkl`. Load these in `ai_engine` code; prefer safe file paths and explicit errors if artifacts are missing.

- When to ask for human review
  - Any change touching trading logic, live order placement, or model thresholds must be reviewed by a developer with trading context.
  - Large TypeScript migrations (>10 files) should be split into waves as described in `MIGRATION_PLAN.md`.

Reference files:
- `backend/main.py`, `backend/routes/`, `backend/database.py`
- `frontend/package.json`, `MIGRATION_PLAN.md`, `scripts/generate-reexports.js`
- `ai_engine/`, `models/xgb_model.pkl`, `models/scaler.pkl`

If anything in this file is unclear or missing, please point to a file or example and request a short update.

---

Quick examples & exact commands

- FastAPI router + DB dependency (copy/paste):

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from backend.database import get_db

router = APIRouter()

@router.get('/')
def list_trades(db: Session = Depends(get_db)):
    # db is a SQLAlchemy session from backend.database.SessionLocal
    return db.query(...).all()

# register with app.include_router(router, prefix='/trades')
```

- CI / workflow exact commands (what runs in GitHub Actions):
  - Backend CI (`.github/workflows/ci.yml`):
    - python -m pip install --upgrade pip
    - pip install -r backend/requirements.txt
    - pip install ruff mypy black pytest coverage bandit safety sqlalchemy-utils
    - ruff check backend
    - mypy backend
    - black --check backend
    - bandit -r backend
    - safety check
    - coverage run -m pytest backend/tests && coverage report -m
  - Backend Tests (`.github/workflows/tests.yml`):
    - pip install -r backend/requirements.txt && pip install -r backend/requirements-dev.txt
    - pytest -v backend/tests
  - Frontend CI (`.github/workflows/frontend-ci.yml` / `frontend.yml`):
    - working dir: ./frontend
    - npm ci (or npm install in `frontend.yml`)
    - npm run typecheck
    - npm run build

- Repository-specific conventions
  - Branch naming: feature/<short-desc>, fix/<short-desc>, hotfix/<short-desc> (follow conventional-style PR titles).
  - Commit messages: keep a short imperative prefix (e.g., "fix(trades): handle zero qty" or "feat(frontend): migrate TradeCard to TSX").
  - PRs that change trading logic, models, or order placement require a human reviewer with trading context and a short explanation of the risk and rollback plan in the PR description.
  - TypeScript migration waves: keep PRs small (1–10 files), use `scripts/generate-reexports.js` to create safe re-export stubs for `.jsx` files and include `npm --prefix frontend run typecheck` in CI checks.

  ---

  Migration (TypeScript) — practical steps

  Run the re-export script (PowerShell - from repo root). This creates backups and replaces `.jsx` files with re-export stubs when a `.tsx` / `.ts` counterpart exists:

  ```powershell
  # ensure node is available, then from repo root:
  node .\scripts\generate-reexports.js
  # or with explicit npm script in the future: npm --prefix frontend run migrate:stubs
  ```

  Migration wave checklist (recommended for each PR):
  - Pick a small group of components (dashboard, trading, widgets) — 1–10 files.
  - Create `Component.tsx` with conservative types in `frontend/src/`.
  - Run the re-export script to replace the legacy `.jsx` with a stub that re-exports the `.tsx`.
  - Run typecheck and unit tests locally:

  ```powershell
  # from repo root (PowerShell)
  npm --prefix frontend run typecheck
  npm --prefix frontend run test:frontend
  ```

  - Push the small PR and include in the PR description:
    - list of migrated files
    - any added or widened types
    - manual QA steps (pages to smoke test)

  PR checklist template (add to PR description when migrating files):
  - Files migrated (list)
  - Type coverage notes (any `any` left intentionally)
  - Local smoke tests performed (routes/pages)
  - CI checks passed (link to run)
  - Rollback instructions: restore backup from `frontend/backups/` if any regressions


