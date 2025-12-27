# Operational Scripts

Operator-facing helpers for database validation, data pipelines, and quick smoke checks live in this directory.

## `data_pipeline.py`

End-to-end ingestion + feature engineering flow used for model governance drills.

- Typical run: `python backend/scripts/data_pipeline.py --symbols BTCUSDT ETHUSDT`
- Flags: `--limit` to cap candles per symbol, `--output-dir` for raw dumps, `--features-path` to override the Parquet target.
- Emits Prometheus counters/histograms (`qt_pipeline_raw_rows_total`, `qt_pipeline_raw_ingest_duration_seconds`, `qt_pipeline_feature_rows_total`) so scheduler runs can be monitored.
- Stores raw payloads alongside the feature matrix under `artifacts/datasets/` by default.

## `alembic_dry_run.py`

Dry-run wrapper for Alembic upgrades against SQLite snapshots.

- Typical run: `python backend/scripts/alembic_dry_run.py --snapshot backups/staging/trades.db --sql-output artifacts/alembic-upgrade-staging.sql`
- Copies the snapshot to a temp directory, upgrades the copy, and optionally writes the offline SQL for change control.
- Respects `QUANTUM_TRADER_DATABASE_URL`; the original value is restored once the run completes.
- Exit code `0` indicates both the offline generation and online upgrade succeeded on the disposable copy.

## `adapter_smoke.py`

Lightweight CI/local helper to exercise exchange adapters without live credentials.

- Typical run: `python backend/scripts/adapter_smoke.py`
- Instantiates each configured adapter (`binance`, `coinbase`, `kucoin`) and invokes `spot_balance()` to ensure imports wire up.
- Handles missing API keys gracefully—exceptions from the exchange client are printed but do not fail the run.
- Use it after dependency upgrades touching `backend.utils.exchanges` or adapter initialization.

## `seed_demo_data.py`

Seeds the database with demo trades and API settings for local exploration.

- Typical run: `python backend/scripts/seed_demo_data.py`
- Requires an upgraded schema (`alembic upgrade head`) and a reachable `DATABASE_URL`.
- Idempotent: skips inserting demo settings when they already exist, making repeated runs safe.
