# Quantum Trader (demo snapshot)

Quantum Trader is an experiment-friendly trading bot sandbox. The repository currently ships a **demo environment**:

- A FastAPI backend that serves deterministic sample data for prices, signals and stats.
- A React/Vite dashboard that visualises that data (price chart, signal feed, stress trends).
- Stress-test utilities under `scripts/stress/` for running repeatable end-to-end checks in CI.

The original vision (full AI + live exchange connectivity + PostgreSQL) is documented in
`ARCHITECTURE.md`, but the code base now focuses on demos and tooling. This README describes the
*current* state so contributors know what is implemented and what remains work-in-progress.

---

## Quick start

### Backend (FastAPI demo APIs)
```bash
python -m venv .venv
. .venv/bin/activate           # PowerShell: .venv\Scripts\Activate.ps1
pip install -r backend/requirements-dev.txt  # includes runtime + test tooling
# For production/minimal installs use: pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

To initialise or upgrade the database schema (SQLite by default; set `QUANTUM_TRADER_DATABASE_URL` for Postgres), run:
```bash
alembic upgrade head
python backend/seed_trades.py
```
If you prefer PostgreSQL, install the optional dependencies, create a database (e.g. `quantum_trader`), and set `QUANTUM_TRADER_DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/quantum_trader` # pragma: allowlist secret before running Alembic.
The backend uses SQLite (created under `backend/data/`) and exposes demo endpoints such as:

- `GET /prices/recent` - deterministic candle series (ccxt when live data is enabled)
- `GET /signals/recent` - rolling signals with direction/confidence metadata
- `POST /ai/train` - schedule an async training job; poll `/ai/tasks/{id}` for status
- `GET /stress/summary` - aggregates produced by `scripts/stress/harness.py`

### Frontend (React/Vite)
```bash
cd frontend
npm install
npm run dev
```
Vite serves the dashboard on <http://localhost:5173>.

Run the Vitest suite (headless):
```bash
npm run test
```. The dashboard talks to the FastAPI instance
at `http://localhost:8000` (configurable with `VITE_API_BASE_URL`).

### Stress harness
```bash
python scripts/stress/harness.py --count 1 --zip-after
```
### Model training & backtesting
```bash
python main_train_and_backtest.py train
```
Runs the end-to-end training pipeline: fetches demo data (or live ccxt data when
`ENABLE_LIVE_MARKET_DATA=1`), trains the regressor across the pairs in `DEFAULT_SYMBOLS`,
executes a quick backtest, and writes artifacts plus `training_report.json` under `ai_engine/models/`.

Run a fresh evaluation with the latest saved model:
```bash
python main_train_and_backtest.py backtest --symbols BTCUSDC ETHUSDC
```
Add `--entry-threshold 0.001` to require a minimum predicted return before taking
trades, or `--skip-backtest` during `train` if you just want artifacts.

### Optional extras
Install additional adapters when needed:
```bash
pip install -r backend/requirements-optional.txt
```
This pulls in ccxt, PostgreSQL drivers and other heavyweight tooling only when you
explicitly need them.

### Docker Compose (full stack demo)
```bash
docker compose up --build
```
Run migrations inside the backend container when you introduce schema changes:
```bash
docker compose exec backend alembic upgrade head
```
Stop the stack with `docker compose down` when finished.

Artifacts are written to `artifacts/stress/`. See `DEVELOPMENT.md` for advanced scenarios (Docker
runs, artifact retention, experiments).

---

## Project layout

```
backend/               FastAPI app, mock routes, SQLite models
frontend/              React/Vite dashboard (TypeScript)
scripts/stress/        Harness, experiments, helpers (+ pytest tests)
ai_engine/             Demo model artefacts and helpers (not wired into CI)
config/stress/         Matrix definitions for stress experiments
artifacts/             Generated results (aggregated.json, reports, experiments)
.github/workflows/     CI pipelines (mypy, frontend/dev tests, stress runs, pre-commit)
```

---

## Current capabilities
- Demo data by default: the training helpers supply deterministic datasets; enable live ccxt data by toggling config when credentials are present.
- SQLite/Postgres via SQLAlchemy + Alembic with tables for trades, candles, equity snapshots, and background `training_tasks`.
- `/ai/predict`, `/ai/train`, and `/ai/tasks` expose the refreshed model pipeline and async training queue.
- Frontend renders charts, signal feed, and backtest views backed by the richer APIs; it now surfaces data provenance (live vs demo) and poll intervals.
- `/prices` and `/signals` call ccxt-backed adapters when `ENABLE_LIVE_MARKET_DATA=1`, falling back to deterministic demo payloads otherwise.
- Stress harness creates aggregated statistics consumable by `/stress/summary` and dashboard widgets.


---

## Backlog & roadmap
The legacy README mixed long-term ambitions with the current MVP. The live backlog now lives in
`TODO.md`, grouped by priority (security, CI policy, backend adapters, frontend UX, observability,
Docs & onboarding). Open that file to see the next actionable tasks.

High level themes:
1. **Security & secrets** - centralise env handling and scrub keys from storage/logs.
2. **CI / dependency hygiene** - optional heavy deps, reproducible installs, clearer workflows.
3. **Real data adapters** - wire backend routes to ccxt / real indicators, add tests.
4. **AI + feature engineering** - document and automate model training/backtesting (pipeline + CLI now in place).
5. **Frontend polish** - flesh out settings, real-time updates, remove legacy TSX duplicates.
6. **Operations** - logging, metrics, health checks, deploy scripts.


---

## Contributing checklist
1. Install pre-commit and run `pre-commit run --all-files` before pushing.
2. Activate a virtual environment; install backend dependencies from
   `backend/requirements.txt` (runtime) and `backend/requirements-dev.txt` (tests/dev tools).
3. For frontend changes run `npm run lint` / `npm run test` if you touch TypeScript.
4. For stress tooling run `pytest scripts/stress/tests`.
5. Update `TODO.md` when scope changes so the backlog stays accurate.

See `CONTRIBUTING.md` and `DEVELOPMENT.md` for detailed setup instructions.
