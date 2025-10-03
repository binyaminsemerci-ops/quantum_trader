# Development Guide — Quantum Trader

This guide covers local stress runs, retention, and artifact handling.

## Prerequisites
- Python 3.11+ installed and on PATH (`python`/`python3`)
- Docker installed (for frontend tests inside a container)
- Optional: Node.js if you want to run frontend tests without Docker

## Dependencies
- Runtime: `pip install -r backend/requirements.txt`
- Full dev/test: `pip install -r backend/requirements-dev.txt`
- Optional adapters (ccxt, Postgres, plotting): `pip install -r backend/requirements-optional.txt`

- Run migrations with `alembic upgrade head` after changing `QUANTUM_TRADER_DATABASE_URL`. Use `python backend/seed_trades.py` to populate demo trades/equity.

- Enable live market data by setting `ENABLE_LIVE_MARKET_DATA=1` (and installing optional adapters). The `/prices` and `/signals` endpoints will then pull via ccxt.

## Database
- Default SQLite file lives under `backend/data/trades.db`.
- Override `QUANTUM_TRADER_DATABASE_URL` (e.g. `postgresql+psycopg://user:pass@localhost:5432/quantum_trader`) # pragma: allowlist secret before running Alembic to use Postgres.
- Apply migrations with `alembic upgrade head`.
- Seed demo trades/equity with `python backend/seed_trades.py` (idempotent).

## Environment
- Copy `.env.example` to `.env` (optional) and adjust values as needed.
  - `DEFAULT_SYMBOLS` controls which trading pairs training/scans pull by default (comma-separated, defaults to ~100 high-volume L1/L2 pairs).
- Useful env vars for stress runs:
  - `STRESS_PREFER_DOCKER=1` — prefer running frontend tests in Docker
  - `DOCKER_FORCE_BUILD=1` — force rebuild of frontend test image
  - `STRESS_KEEP_ZIPS=5` — keep latest N zip archives after a run with `--zip-after`
  - `STRESS_KEEP_ITERS=200` — keep latest N per-iteration JSON files after `--zip-after`
  - `STRESS_PRUNE_ALERT_THRESHOLD=100` — emit warning if retention removes more than this many iteration files in one pass
  - `STRESS_PRUNE_ALERT_WEBHOOK=https://hooks.slack.com/...` — optional webhook invoked when threshold is exceeded
  - `STRESS_REPORT_OUTDIR=relative/or/absolute/path` — override where `generate_report.py` reads/writes aggregated/report (useful for testing)
  - `STRESS_OUTDIR=/tmp/stress` - overstyr hovedmappen for artefakter (relativ sti tolkes relativt til repo-root).
  - `STRESS_FRONTEND_BASE_IMAGE=node:18-bullseye-slim` - overstyr Docker-bilde brukt for frontend-testene.
  - `STRESS_FRONTEND_EXTRA_NPM_DEPS="playwright@1.39.0"` - installer ekstra npm-avhengigheter i Docker-testbildet.
  - `STRESS_FRONTEND_IMAGE=quantum_trader_frontend_test:node18` - angi egen tag for frontend-testimage (nyttig for matriser).

## Running stress harness
Run a single iteration and zip artifacts after:

```bash
python scripts/stress/harness.py --count 1 --zip-after
```

Run 10 iterations, prefer Docker, and keep last 200 iteration JSON files:

```bash
export STRESS_PREFER_DOCKER=1
export STRESS_KEEP_ITERS=200
python scripts/stress/harness.py --count 10 --zip-after
```

Artifacts:
- Iteration results: `artifacts/stress/iter_*.json`
- Aggregated: `artifacts/stress/aggregated.json` (includes `stats` section)
- HTML report: `artifacts/stress/report.html`
- Zips: `artifacts/stress_artifacts_<timestamp>.zip`

## Model training & backtesting
- Quick refresh (offline stubs): `python -m ai_engine.train_and_save --limit 600` regenerates `xgb_model.pkl`, `scaler.pkl`, `metadata.json`, and `training_report.json` under `ai_engine/models/`.
  The helper automatically falls back to deterministic synthetic data when live adapters are not configured.
- Full CLI pipeline: `python main_train_and_backtest.py train --limit 480 --symbols BTCUSDC ETHUSDC` fetches data via the FastAPI helpers (ccxt when `ENABLE_LIVE_MARKET_DATA=1`), trains the regressor,
  runs a capped equity backtest, and writes the same artefacts.
  Use `--skip-backtest` when you only need refreshed model/scaler files.
- Evaluate an existing model: `python main_train_and_backtest.py backtest --symbols BTCUSDC ETHUSDC --entry-threshold 0.001` reruns metrics using previously saved artefacts.
- Background workflow: `POST /ai/train` schedules the same pipeline via FastAPI; follow up with `GET /ai/tasks/{id}` to monitor progress.
  Ensure the database schema is up to date (`alembic upgrade head`) so the `training_tasks` table exists before scheduling jobs.
## Deployment
- Build locally with Docker Compose: `docker compose up --build`. The stack includes backend (uvicorn), frontend (nginx) and Postgres.
- Override database/env secrets by setting environment variables before running compose or by supplying a `.env` file (Compose reads from the shell by default).
- Production images are built/published by `.github/workflows/deployment-build.yml`; set `GHCR_PAT` in repository secrets to enable pushes.

## Retention utilities
- Zip rotation is automatic with `STRESS_KEEP_ZIPS` or via uploader `--retain N`.
- Prune old iteration JSONs manually:

```bash
python scripts/stress/retention.py --keep 200 --warn-over 100
```

## Cloud upload (optional)
Use `scripts/stress/upload_artifacts.py` to zip and upload stress artifacts.
Pick one provider and set credentials in your env (`.env` can help):

```bash
python scripts/stress/upload_artifacts.py --provider s3 --dest s3://my-bucket/path/stress.zip --retries 3 --retry-delay 3 --retain 5
```

Pinned provider SDKs for CI are defined in `requirements-ci-upload.txt`.

## Evolusjons-eksperimenter
- Konfigurer matriser i `config/stress/experiments.json` (node-bilder, ekstra npm-avhengigheter, antall iterasjoner).
- Forhåndsvis hvilke runs som kjøres med `python scripts/stress/experiments.py --dry-run`.
- Kjør eksperimentene: `python scripts/stress/experiments.py` (eventuelt `--count 3` for å overstyre antall iterasjoner).
- Resultater legges i `artifacts/stress/experiments/<navn>` og en samlet oversikt skrives til `artifacts/stress/experiments/index.json`.
- Hvert eksperiment setter `STRESS_OUTDIR`, `STRESS_FRONTEND_BASE_IMAGE`, `STRESS_FRONTEND_IMAGE` og `STRESS_FRONTEND_EXTRA_NPM_DEPS`; du kan bruke disse miljøvariablene direkte om du vil orkestrere egne løp.

## CI notes
- Workflow `Stress tests` builds a Docker image for frontend tests and runs one iteration with `--zip-after`.
- It uploads artifacts and (optionally) zips to cloud when secrets are set.
- The HTML report is generated and uploaded as `stress-report` artifact.
- Frontend audit triage runs in main CI and daily audit; optional issue is created when `AUDIT_CREATE_ISSUES` secret is `true`.

## Frontend test image (digest pin)
- `frontend/Dockerfile.test` supports a build arg `BASE_IMAGE` (defaults to `node:20-bullseye-slim`).
- For full reproducibility in CI, set a digest ref:

```yaml
env:
  NODE_IMAGE_REF: node@sha256:<digest>
```

The stress workflow defaults to `node:20-bullseye-slim@sha256:1c2b56658c1ea4737e92c76057061a2a5f904bdb2db6ccd45bb97fda41496b80`; override via the `NODE_IMAGE_REF` secret if you need a different digest.
