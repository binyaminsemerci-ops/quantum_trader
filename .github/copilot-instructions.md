## Quick orientation for AI coding agents

This file captures the essential, discoverable knowledge an AI coding agent needs
to be productive in the quantum_trader repository. Keep changes small, explicit,
and reference the project's existing patterns and files.

1) Big-picture architecture
- Backend: FastAPI app at `backend/main.py` exposing routers in `backend/routes/`
  (notable routers: `trades`, `stats`, `chart`, `settings`, `binance`).
- Database: SQLAlchemy in `backend/database.py`. Default: file SQLite under
  `backend/data/trades.db` unless `QUANTUM_TRADER_DATABASE_URL` env var is set.
  Use the `get_db()` dependency to obtain a session.
- Frontend: React + Vite in `frontend/` (see `frontend/package.json`). Frontend
  talks to the backend on port 8000 (see `docker-compose.yml`).
- AI / ML code: features and model helpers live under `ai_engine/` and
  the training/backtest runner is `main_train_and_backtest.py`.

2) Typical data / request flow
- Frontend components call REST endpoints under `/trades`, `/stats`, `/chart`,
  and `/settings` served by FastAPI. Those endpoints use SQLAlchemy models
  (`TradeLog`, `Settings`) defined in `backend/database.py`.
- The AI code consumes historical OHLC data and produces features in
  `ai_engine/feature_engineer.py` (functions like `add_technical_indicators`).

3) Developer workflows & concrete commands
- Backend local setup (PowerShell):
  - `python -m pip install --upgrade pip`
  - `pip install -r backend/requirements.txt`
  - (dev tools) `pip install -r backend/requirements-dev.txt`
  - Run API (typical): `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
    (assume `backend.main` is the ASGI app module).
- Frontend local setup:
  - `cd frontend && npm install`
  - `npm run dev` to start Vite dev server, `npm run build` for CI-style build.
- Docker: `docker-compose up --build` uses `docker-compose.yml` to bring up
  `backend` and `frontend` containers (backend mounts source and database dirs).

4) CI, linting and constraints agents must respect
- CI separates runtime vs dev/test deps: `backend/requirements.txt` vs
  `backend/requirements-dev.txt`. Do not add dev-only packages to runtime
  requirements. The enforcement script is `backend/scripts/check_dev_deps_in_runtime.py`.
  CI will fail when dev-only packages are present in runtime.
- Linters and scanners are installed into an isolated venv `.venv_linters` in CI
  (see `.github/workflows/ci.yml`). CI runs ruff, mypy (with test exclusions),
  black, bandit and safety. Use the same boundaries locally to avoid surprises.

CI failure examples (what you'll see and how to fix it)
- Ruff / Black failures
  - Symptoms: CI job shows `ruff check backend` or `black --check backend` failing.
  - Fix: run the same locally and apply the formatter/linter fixes:
    - `ruff check backend` to list offences, or `ruff check backend --fix` to auto-fix where safe.
    - `black backend` to reformat files (then commit).
- Mypy failures (common):
  - Symptoms: `mypy backend --exclude 'backend/tests/.*'` or the later full mypy run fails with import/type errors.
  - Root cause: CI runs mypy twice — first excluding tests (so test-only deps aren't required), then installs test tooling and runs mypy again. Local failures commonly come from missing dev/test packages (pytest, pytest-asyncio, types).
  - Fix:
    - If you're running mypy locally to match CI first-run: `mypy backend --exclude 'backend/tests/.*'`.
    - To run the full mypy run like CI's second pass, install test tooling after ensuring dev-deps are not accidentally in runtime:

```pwsh
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
python -m pip install pytest pytest-asyncio
mypy backend
```

- Dev-only packages enforcement (most common CI blocker)
  - Symptoms: CI step `Check for dev-only packages in runtime` fails and annotations/errors point to `backend/dev_in_runtime.txt`.
  - Why it happens: you installed dev-only packages into the interpreter that CI treats as the runtime env. CI intentionally installs dev/test tooling only after this check.
  - Fix:
    - Run the enforcement script locally to reproduce:

```pwsh
python backend/scripts/check_dev_deps_in_runtime.py
```

    - If the script writes `backend/dev_in_runtime.txt`, either uninstall the offending packages from the runtime interpreter or move them into `backend/requirements-dev.txt` instead of `backend/requirements.txt`.
    - Quick repair helpers are provided:

```pwsh
.\scripts\repair-dev-deps.ps1   # PowerShell (Windows)
./scripts/repair-dev-deps.sh    # POSIX
```

  - Note: CI will annotate the PR with offending package names; don't add test tooling into runtime requirements to bypass this check.

5) Project-specific conventions & patterns
- FastAPI routers: define an `APIRouter` in `backend/routes/<name>.py` and
  register via `app.include_router(..., prefix="/<name>")` in `backend/main.py`.
- DB pattern: use `get_db()` dependency from `backend/database.py`. Use
  `db.add()`, `db.commit()`, `db.refresh()` for persisted objects (see
  `backend/routes/trades.py`).
- Type & tests: backend tests live under `backend/tests`; frontend tests use
  Vitest. CI runs mypy twice: first excluding tests, then again after test
  tooling is installed.

6) Integration points & external dependencies to watch
- Binance placeholder router: `backend/routes/binance.py` contains the external
  API wrapper stubs used by the project. If implementing live exchange code,
  respect existing route shapes and test mocks used in tests.
- Environment variables:
  - `QUANTUM_TRADER_DATABASE_URL` to override DB for tests/CI.
  - `.env` is mounted into the backend container in `docker-compose.yml`.

7) Small implementation examples to follow
- Add a router: copy pattern from `backend/routes/trades.py` (Pydantic request
  model, DB dependency, commit/refresh). Keep responses simple dicts for now.
- Feature engineering: mirror style in `ai_engine/feature_engineer.py` — pure
  functions that accept and return pandas DataFrames and call them from the
  runner `main_train_and_backtest.py`.

8) Safety & editing rules for agents
- Never add developer-only packages to `backend/requirements.txt`. If a
  package is strictly dev-use, add it to `backend/requirements-dev.txt`.
- When editing DB models, also run `Base.metadata.create_all(bind=engine)`
  or update migrations (project currently uses SQLite files; check `migrations/`).
- For any change touching CI or linting, reference `.github/workflows/ci.yml`
  — CI has intentional ordering (install runtime deps, run dev-deps check,
  then install test tooling).

DB & migrations (practical notes)
- Current state: `backend/database.py` defines models (e.g., `TradeLog`, `Settings`) and calls `Base.metadata.create_all(bind=engine)` when imported. The repository contains a `migrations/` folder but `migrations/versions/` is empty — there is no Alembic scaffolding or `alembic.ini` in the repository today.
- What this means:
  - Local development and tests rely on `Base.metadata.create_all` to create tables automatically.
  - There is no automated schema migration tool configured (Alembic is not present). Do not attempt to create Alembic revisions unless you add Alembic configuration and coordinate with maintainers.
- When you add or change ORM models:
  1. Update the SQLAlchemy model class in `backend/database.py` (or the module you choose to host models).
  2. Run the app or execute the create_all helper to update the local SQLite DB:

```pwsh
python -c "from backend.database import Base, engine; Base.metadata.create_all(bind=engine)"
# Or just run the backend (imports create_all on module import):
uvicorn backend.main:app --reload
```

  3. Run tests (`pytest backend/tests`) to ensure migrations aren't required by tests and the DB schema matches expectations.
- For production deployments or when schema migrations are needed:
  - Add Alembic (or another migration tool) to the repo, commit `alembic.ini` and `migrations/` scripts, and add CI steps that run `alembic upgrade head` against a real DB instance.
  - If you introduce Alembic, update `.github/workflows/ci.yml` so CI runs migration checks before tests, and document the migration workflow here.

CI snippet (example)
Add the following step in the CI job before running tests when you enable Alembic:

```yaml
- name: Run Alembic migrations
  run: |
    pip install alembic
    alembic -c migrations/alembic.ini upgrade head
```

Note: For SQLite this is a no-op on the CI ephemeral FS unless you mount a persistent DB; for Postgres/other DBs configure `sqlalchemy.url` in `migrations/alembic.ini` or set `DATABASE_URL`.

9) Where to look first (important files)

9) Where to look first (important files)
- `backend/main.py`, `backend/database.py`, `backend/routes/` (all routers)
- `backend/requirements.txt`, `backend/requirements-dev.txt`
- `backend/scripts/check_dev_deps_in_runtime.py` (dev-deps enforcement)
 - `migrations/` (present but not configured; check before assuming Alembic)
- `frontend/package.json`, `frontend/src/` (React + Vite)
- `ai_engine/feature_engineer.py`, `main_train_and_backtest.py`

If anything here is unclear or you'd like more detail in a particular area
(CI, DB, frontend integration, or ML wiring), tell me which section to expand
and I will iterate.
