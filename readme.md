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
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
The backend uses SQLite (created under `backend/data/`) and exposes demo endpoints such as:

- `GET /prices/recent` – deterministic candle series
- `GET /signals/?page=1&page_size=10` – mock trade signals
- `GET /stress/summary` – aggregates produced by `scripts/stress/harness.py`

### Frontend (React/Vite)
```bash
cd frontend
npm install
npm run dev
```
Vite serves the dashboard on <http://localhost:5173>. The dashboard talks to the FastAPI instance
at `http://localhost:8000` (configurable with `VITE_API_BASE_URL`).

### Stress harness
```bash
python scripts/stress/harness.py --count 1 --zip-after
```
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
- Demo data only: no live exchange connectivity and no real model training pipeline.
- SQLite-backed API with minimal tables (`TradeLog`, `Settings`). Postgres/Alembic integration is a
  future goal (see TODO).
- Frontend renders charts, signal feed and stress trends using the demo APIs.
- Stress harness creates aggregated statistics consumable by `/stress/summary` and the dashboard.

---

## Backlog & roadmap
The legacy README mixed long-term ambitions with the current MVP. The live backlog now lives in
`TODO.md`, grouped by priority (security, CI policy, backend adapters, frontend UX, observability,
Docs & onboarding). Open that file to see the next actionable tasks.

High level themes:
1. **Security & secrets** – centralise env handling and scrub keys from storage/logs.
2. **CI / dependency hygiene** – optional heavy deps, reproducible installs, clearer workflows.
3. **Real data adapters** – wire backend routes to ccxt / real indicators, add tests.
4. **AI + feature engineering** – document and automate model training/backtesting.
5. **Frontend polish** – flesh out settings, real-time updates, remove legacy TSX duplicates.
6. **Operations** – logging, metrics, health checks, deploy scripts.

---

## Contributing checklist
1. Install pre-commit and run `pre-commit run --all-files` before pushing.
2. Activate a virtual environment; install backend dependencies from
   `backend/requirements.txt` (runtime) and `backend/requirements-dev.txt` (tests/dev tools).
3. For frontend changes run `npm run lint` / `npm run test` if you touch TypeScript.
4. For stress tooling run `pytest scripts/stress/tests`.
5. Update `TODO.md` when scope changes so the backlog stays accurate.

See `CONTRIBUTING.md` and `DEVELOPMENT.md` for detailed setup instructions.
