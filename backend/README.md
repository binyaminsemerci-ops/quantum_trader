# backend — developer quickstart

This file documents the minimal steps for setting up the Python backend for
local development and testing.

Install runtime requirements (for running the app):

```pwsh
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

> **Note**
> `xgboost` is pinned to `1.7.6`, which is the newest release compatible with
> our lightweight test stubs living under `xgboost/`. If you upgrade the
> dependency, ensure the stub package still deserialises the pickled models and
> update the pin accordingly.

Install developer/test requirements (for running tests, linters, and local
tools):

```pwsh
pip install -r backend/requirements-dev.txt
```

## Why dev-only for some packages
We intentionally keep some packages (for example `SQLAlchemy-Utils`) in
`backend/requirements-dev.txt` rather than runtime requirements. This avoids
installing developer-only tooling in CI/runtime, reduces the attack surface,
and ensures security advisories are tracked and addressed explicitly.

If you need the dev tools locally, run the command above. CI intentionally
installs only specific test/lint/security tools so runtime environments stay
minimal.

## Check for accidental dev-only installs
A small script is provided to help detect if any dev-only packages are
present in your runtime environment (useful for pre-commit checks or local
validation):

```pwsh
python backend/scripts/check_dev_deps_in_runtime.py
```

If it prints a list of packages, you may have installed dev requirements into
your runtime environment. CI runs this script and emits a non-blocking warning
if any dev-only packages are detected.

## Enable local git pre-commit hook (optional)
To enable the included local git hook that prevents commits when dev-only
packages are present in your runtime environment:

```pwsh
# From repo root (one-time):
git config core.hooksPath .githooks
```

After that, the `.githooks/pre-commit` script will run on each commit and abort
the commit if dev-only packages are detected.

## Makefile target
You can also run the check locally via the Makefile target from the repo root:

```pwsh
make -C backend check-dev-deps
```

## Windows / PowerShell notes
Windows developers can use PowerShell to set up and run the same tools:

```powershell
# Create and activate the venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dev deps
pip install -r backend/requirements-dev.txt

# Run the check
python backend/scripts/check_dev_deps_in_runtime.py
```

## Repair helper
If the check finds dev-only packages installed at runtime, you can run the
repair helper to uninstall them (it will prompt for confirmation):

POSIX:
```bash
./scripts/repair-dev-deps.sh
```

PowerShell:
```powershell
.\scripts\repair-dev-deps.ps1
```

Both scripts support a dry-run mode to preview what would be uninstalled:

POSIX:
```bash
./scripts/repair-dev-deps.sh --dry-run
```

PowerShell:
```powershell
.\scripts\repair-dev-deps.ps1 -DryRun
```

## Using an isolated linters virtualenv locally

The CI uses an isolated `.venv_linters` virtualenv to install linters and
security scanners so their transitive dependencies do not appear in the
application runtime environment (and therefore do not trigger the dev-deps
enforcement check).

If you'd like to replicate CI locally, create the linters venv with:

```powershell
python -m venv .venv_linters --system-site-packages
.\.venv_linters\Scripts\Activate.ps1
.venv_linters\Scripts\pip install --upgrade pip
.venv_linters\Scripts\pip install ruff mypy black bandit safety
```


Notes:

- `--system-site-packages` lets tools in the linters venv import your
  runtime packages without reinstalling them, which prevents false
  "import not found" errors in mypy while keeping the linters' own deps
  isolated from your runtime Python.

- Do not install test-only packages (pytest, pytest-asyncio, etc.) into the
  runtime interpreter used by the app; CI installs those into the runner
  Python only after the enforcement check.

## Deployment runbook

### Pre-flight (staging)

- Confirm feature flags and scheduler cadences in `backend/.env`; keep
  `QUANTUM_TRADER_DISABLE_SCHEDULER=1` during deploy, re-enable after smoke tests.
- Run `python -m pytest backend/tests` and `alembic check` locally to ensure the
  migration heads match the target environment.
- Regenerate the build artifact if model weights changed: execute
  `python ai_engine/train_and_save.py --output-dir artifacts` and verify the
  latest run is recorded via `GET /ai/model-info`.
- Ship the container/image or code bundle and apply the latest Alembic upgrade
  in staging: `alembic upgrade head` (set `DATABASE_URL` beforehand).
- Before applying the migration, run the dry-run helper to confirm schema
  changes on the latest snapshot (see `docs/database_promotion.md`):

  ```pwsh
  python backend/scripts/alembic_dry_run.py --snapshot backups/staging/trades.db `
      --sql-output artifacts/alembic-upgrade-staging.sql
  ```
- Start the backend with staging secrets mounted; tail logs for successful
  scheduler bootstrap and absence of stack traces.

### Staging verification

- Hit `/health` and `/health/scheduler` (expect `status: ok`, no pending errors).
- Inspect Prometheus metrics to ensure cache hit/miss counters increment during
  smoke traffic and the `model_inference_duration_seconds` histogram reports
  new samples.
- Call `/risk` with the staging admin token, confirm kill-switch state is
  `false`, limits reflect the requested release, and price sanity bounds match
  `config/risk.py`.
- Invoke `backend/scripts/data_pipeline.py --dry-run` to ensure the feature
  pipeline can hydrate the staging database without destructive writes.
- Record staging sign-off in the release tracker together with pytest + smoke
  results.

### Production promotion

- Freeze commits by tagging the candidate (`git tag deploy-YYYYMMDD-N`), then
  deploy the same artifact that passed staging.
- Export production secrets and `DATABASE_URL`, run `alembic upgrade head` with
  `--sql` first to generate the script for change-management review, then apply
  the upgrade once approved.
- Start the backend with `QUANTUM_TRADER_DISABLE_SCHEDULER=1`, execute smoke
  checks against `/health`, `/risk`, `/ai/model-info`, and the Prometheus
  surface.
- Re-enable the scheduler (`QUANTUM_TRADER_DISABLE_SCHEDULER=0`) and monitor the
  first execution loop for exposure drift before marking the release live.

## Incident response runbook

- **Critical trade failure**: Set the kill switch via
  `POST /risk/kill-switch` (`{"enabled": true}`) with the admin token. Confirm
  the value in `/risk`, then investigate `backend/logs/trades.log` and
  Prometheus counters for root cause.
- **Scheduler stall**: Query `/health/scheduler` and check `errors`. Restart the
  process if `running` is `false`; if errors reference third-party APIs, set
  `QUANTUM_TRADER_DISABLE_SCHEDULER=1` temporarily.
- **Model governance regression**: Inspect the latest training run in the
  `ModelTrainingRun` table (use `sqlite3 backend/quantum_trader.db`) and compare
  the `model_path` with the artifact on disk. Roll back to the previous tag if
  metrics breached guard-rails.
- **Database corruption**: Immediately swap the service to the read-only
  fallback by pointing `DATABASE_URL` to the last good snapshot. Run
  `database/init_db.py --check` before re-opening writes.
- **Incident closure**: Document timeline and recovery steps in
  `docs/incidents/YYYY-MM-DD-<slug>.md`, link to Grafana/Prometheus snapshots,
  and schedule a follow-up to backfill missing market data if the scheduler was
  disabled.

## Compliance checklist

- Secrets (API keys, admin token, DB creds) live in the secret store or
  environment-specific vault; no plaintext copies in git.
- Access controls enforced via `X-Admin-Token` for mutable endpoints and
  network-layer ACLs on production hosts.
- Database migrations reviewed with `alembic upgrade --sql` artifacts before
  applying to regulated environments.
- Model metadata (version, metrics, training config) logged and persisted in the
  `ModelTrainingRun` table; `/ai/model-info` exposes the active artifact for
  audit.
- Scheduler kill-switch and risk guard overrides captured in
  `backend/data/admin_audit.log` for compliance review.
- Incident reports written within 24 hours using the incident template under
  `docs/incidents/` and stored with supporting telemetry captures.

## Environment variables

For local development copy `backend/.env.example` to `backend/.env` and set
your local values (do not commit `backend/.env`). The repo includes a
`backend/.env.example` file with common values to get started.

Note: the example now defaults to a local SQLite database so a username
or external DB credentials are not required for standalone local runs or
for running the test-suite. If you plan to run the application against an
external Postgres instance, update `DB_USER`, `DB_PASS` and `DATABASE_URL`
in your local `backend/.env` accordingly.

### Background market data refresh

The API ships with an APScheduler-based job that keeps market and sentiment
caches warm in the background. The following environment variables can be
used to tune or disable the scheduler:

- `QUANTUM_TRADER_REFRESH_SECONDS` — refresh cadence in seconds (defaults to 180).
- `QUANTUM_TRADER_SYMBOLS` — comma-separated list of trading pairs to refresh
  (defaults to `BTCUSDT,ETHUSDT`).
- `QUANTUM_TRADER_DISABLE_SCHEDULER` — set to `1` to skip starting the scheduler
  (useful for tests or constrained environments).
- `QUANTUM_TRADER_LIQUIDITY_REFRESH_SECONDS` — cadence for the automated liquidity
  universe refresh job (defaults to 900 seconds). Set to `0` to skip scheduling
  the job without disabling the scheduler entirely.
- `QUANTUM_TRADER_DISABLE_LIQUIDITY` — set to `1` to disable the liquidity job
  while keeping the rest of the scheduler active.
- `QT_LIQUIDITY_LIQ_WEIGHT` / `QT_LIQUIDITY_MODEL_WEIGHT` — blend ratios between
  raw liquidity ranking and the ML agent output (values are normalised internally).
- `QT_LIQUIDITY_MODEL_SELL_THRESHOLD` — minimum agent confidence before a SELL
  recommendation heavily down-weights a symbol (defaults to `0.55`).
- `QT_LIQUIDITY_MAX_PER_BASE` — cap the number of picks per base asset when
  blending the final selection (defaults to `1`).
- Liquidity refreshes now emit a compact `analytics` payload summarising
  selection coverage and the top weighted symbols. The snapshot is available
  via `/scheduler/status`, `/health/scheduler`, and the admin liquidity trigger
  response for downstream dashboards.

### Automated execution loop

- `QUANTUM_TRADER_EXECUTION_SECONDS` — cadence for the automated execution job
  that rebalances the paper portfolio (defaults to 1800 seconds). Set to `0`
  to skip scheduling without disabling the scheduler entirely.
- `QUANTUM_TRADER_DISABLE_EXECUTION` — set to `1` to disable the execution job
  while keeping the rest of the scheduler active.
- `QT_EXECUTION_MIN_NOTIONAL` — per-order notional floor before an order is
  considered actionable (defaults to 50.0).
- `QT_EXECUTION_MAX_ORDERS` — cap the number of orders per execution cycle
  (defaults to 10).
- `QT_EXECUTION_CASH_BUFFER` — reserve cash from total equity when computing
  targets (defaults to 0.0).
- `QT_EXECUTION_ALLOW_PARTIAL` — toggle whether sells can partially close
  positions (defaults to `true`).
- `QT_EXECUTION_EXCHANGE` — choose the execution adapter; set to `binance` to
  use live trading keys or leave as `paper` (default) for the in-memory
  simulator.
- `QT_EXECUTION_QUOTE_ASSET` — quote asset used when mapping account balances
  to trading symbols (defaults to `USDT`).
- `QT_EXECUTION_BINANCE_TESTNET` — set to `1` to route Binance orders and
  account snapshots to the Binance Spot Testnet; defaults to `0` (production
  endpoints).

#### Testnet smoke checklist (staging)

1. Export the execution env vars (`QT_EXECUTION_EXCHANGE=binance`,
   `QT_EXECUTION_BINANCE_TESTNET=1`, `QUANTUM_TRADER_EXECUTION_SECONDS=60`) and
   supply staging API keys via the config loader or secret store.
2. Launch the backend without disabling the scheduler and tail the logs for an
   `execution-rebalance` run. The run should log planned/submitted orders and
   update `execution.gross_exposure` in `/health/scheduler`.
3. Confirm the orders appear on the Binance Spot Testnet dashboard and respect
   your configured risk limits and quote asset.
4. Revert `QUANTUM_TRADER_EXECUTION_SECONDS` (and any temporary overrides) once
   validation is complete.

### Risk guard limits

- `QT_MAX_NOTIONAL_PER_TRADE` — maximum notional the risk guard allows per
  individual order (defaults to 1000.0).
- `QT_MAX_DAILY_LOSS` — total realised loss threshold before the guard blocks
  additional trades (defaults to 500.0).
- `QT_MAX_POSITION_PER_SYMBOL` — optional cap on the projected notional for a
  single symbol after an execution cycle; omit or set to a non-positive value
  to disable the limit.
- `QT_MAX_GROSS_EXPOSURE` — optional cap on the aggregated notional across all
  symbols; omit or set to a non-positive value to disable the limit.

These limits pair with the portfolio position snapshot recorded by the
execution job so the API and scheduler telemetry surface the same exposure
figures.

Runtime health is exposed via two HTTP endpoints:

- `GET /health` — general service health including a scheduler snapshot.
- `GET /health/scheduler` — scheduler-only view that reports the most recent
  run status, errors and configured symbols.
- Liquidity snapshot responses include `analytics` describing the latest
  selection blend (top allocations, base coverage and provider metadata).
- `GET /risk` — risk guard snapshot covering kill switch, limits and rolling
  trade metrics alongside override state.
- `POST /risk/kill-switch` — override the kill switch (true/false) or reset to
  config defaults (`null`).
- `POST /risk/reset` — clear recorded trades and kill switch overrides.
- `GET /metrics` — Prometheus metrics endpoint (HTTP, scheduler, risk guard).

All `/risk` endpoints require the `X-Admin-Token` header matching the value of
`QT_ADMIN_TOKEN` to prevent unauthorised toggles in staging/production.

### Admin API token usage

- Set `QT_ADMIN_TOKEN` in `backend/.env` to a high-entropy value for any
  environment where mutable admin routes should be gated. Leaving it blank
  disables the guard and is only acceptable in throwaway local scenarios.
- Send the token via `X-Admin-Token` when calling admin routes. Today the
  guarded surface is:
  - `GET /risk`
  - `POST /risk/kill-switch`
  - `POST /risk/reset`
  - `POST /liquidity/refresh`
- The backend test-suite relies on `backend/tests/conftest.py` setting
  `QT_ADMIN_TOKEN=test-admin-token` so fixtures can exercise the admin surface
  without additional setup. If you run tests outside pytest, export the same
  value or provide your own token.
- Quick verification:

  ```pwsh
  # Expect HTTP 401 without the header
  curl http://localhost:8000/risk

  # Repeat with token to receive the snapshot
  curl http://localhost:8000/risk -H "X-Admin-Token: $env:QT_ADMIN_TOKEN"
  ```
- Audit entries for these routes, along with `/settings` updates, are written to
  the path configured via `QT_ADMIN_AUDIT_PATH` (default
  `backend/data/admin_audit.log`). Use this log when reconciling operator
  actions.

Additional endpoints (e.g. scheduler controls) will adopt the same `X-Admin-Token`
header once they are promoted from the TODO backlog. When that happens, update
the table above and keep token handling consistent across the API surface. See
`docs/admin_token_usage.md` for the living reference.

### Risk guard state persistence

- `QT_RISK_STATE_DB` — path to the SQLite file used to persist risk guard state
  (defaults to `backend/data/risk_state.db`). Relative paths resolve from the
  backend package directory.
- `QT_ADMIN_TOKEN` — shared secret for the risk admin API. Omit only in local
  throwaway environments; use a strong random string elsewhere.

When verifying a deployment (staging or production), run the application and
inspect the scheduler endpoint:

```pwsh
curl http://localhost:8000/health/scheduler | ConvertFrom-Json
```

Confirm that `status` is `ok`, `running` is `true`, and the `next_run_time`
 aligns with the configured interval. If `status` reports `degraded`, review
the `errors` payload for the affected symbols before promoting the release.

Additional staging/production readiness guidance lives in `docs/ai_production_checklist.md`
and `docs/staging_deployment_guide.md`. Review those documents before enabling
automated trading workflows outside of local development.
- `--system-site-packages` lets tools in the linters venv import your
  runtime packages without reinstalling them, which prevents false
  "import not found" errors in mypy while keeping the linters' own deps
  isolated from your runtime Python.

- Do not install test-only packages (pytest, pytest-asyncio, etc.) into the
  runtime interpreter used by the app; CI installs those into the runner
  Python only after the enforcement check.

## Safety CLI (CI)

The CI pipeline runs `safety scan --full-report` as part of the security
checks. The Safety CLI requires authentication (an API key) for non-interactive
use in CI. If you want Safety scans to run automatically on CI, follow these
steps:

1. Sign up for a (free) Safety CLI account and create an API key:

    - Visit <https://safetycli.com> and follow the sign-up flow.
    - Create an API key from the Safety dashboard and copy it.

1. Store the key as a GitHub Actions repository secret named `SAFETY_API_KEY`:

    - Through the GitHub web UI: Repository → Settings → Secrets and variables
      → Actions → New repository secret. Use `SAFETY_API_KEY` as the name and
      paste the key as the value.

    - Or using the GitHub CLI (recommended for automation):

     ```bash
     # Replace YOUR_KEY_HERE with the key you obtained from Safety CLI
     gh secret set SAFETY_API_KEY --body "YOUR_KEY_HERE"
     ```

1. Once the secret is configured, CI will run the Safety scan non-interactively
   (the workflow authenticates using the secret and executes `safety scan`).

If `SAFETY_API_KEY` is not set, CI will skip the Safety scan and emit a
warning instead of failing with an interactive prompt. This lets CI continue
while you provision the key.

Security notes

- Treat `SAFETY_API_KEY` like any other secret: limit access and rotate keys
  periodically. Prefer organization secrets for cross-repo policies when
  possible.
- Do not store the key directly in the repo or in plaintext files.

If you want, I can set the repository secret for you if you provide the API
key (not recommended to paste secrets in chat). Alternatively, I can give you
the exact `gh` command to run locally or on your machine.
