# Staging Deployment Guide (AI Services)

This guide explains how to spin up a staging environment that mirrors production
for the AI trading stack while routing executions to a safe simulator.

## 1. Objectives

- Validate ingestion, caching, and scheduler flows against live market data.
- Observe model predictions and backtest logic under real-time conditions.
- Exercise risk controls and operational runbooks before production release.

## 2. Environment Layout

| Component            | Staging Target                                  | Notes                                                     |
|----------------------|--------------------------------------------------|-----------------------------------------------------------|
| Backend FastAPI app  | `staging-api.<domain>` (containerized deployment) | Deploy the same image/tag as production with staging vars.|
| Database             | Isolated Postgres / SQLite instance              | Seed with anonymized historical trades for context.       |
| Scheduler/Workers    | Same APScheduler process                         | Share scheduler settings; allow shorter intervals for QA. |
| Market data feeds    | Binance/CoinGecko (staging API keys)             | Keys scoped to paper trading or test accounts.            |
| Trade execution      | Paper trading broker / simulator service         | Capture orders for analysis; do not touch live capital.   |
| Observability stack  | Grafana + Loki (or Application Insights)         | Mirror production dashboards + alerts.                    |

## 3. Configuration Checklist

1. **Secrets & API Keys**
   - Generate dedicated API keys for Binance/CoinGecko (paper trading tier).
   - Store keys in secret manager (Azure Key Vault, AWS SM, etc.).
   - Configure deployment to inject secrets as environment variables.

2. **Environment Variables**
   - Duplicate `backend/.env.example` to a staging variant.
   - Set `QUANTUM_TRADER_REFRESH_SECONDS` to `120` for tighter feedback loops.
   - Add `STAGING_MODE=1` (to be consumed by future guardrails).
   - Point `DATABASE_URL` to staging DB.
   - Set `QT_EXECUTION_EXCHANGE=binance` and `QT_EXECUTION_BINANCE_TESTNET=1` to route
     executions to the Binance spot testnet using staging keys.
   - Optionally override `QT_EXECUTION_QUOTE_ASSET` when staging accounts use
     a different quote currency.

3. **Data Storage**
   - Create isolated buckets or blob containers for staging artifacts.
   - Enable automated cleanup of staging artifacts older than 30 days.

4. **Network & Access Controls**
   - Restrict staging endpoints to VPN/IP allowlist.
   - Ensure CI/CD service principal has least privilege on staging resources.

## 4. Deployment Steps

1. Build container image (reuse production Dockerfile) and tag as `staging-<sha>`.
2. Push image to container registry.
3. Apply IaC template (Terraform/Bicep) or run `docker-compose -f docker-compose.staging.yml up -d`.
4. Seed database with `backend/scripts/seed_trades.py --env staging` (future flag) or snapshot.
5. Run database migrations (`alembic upgrade head`).
6. Verify application startup logs for scheduler activation.
7. Hit `/health` and `/health/scheduler` to ensure background jobs are running.
8. Execute `python scripts/api_smoke_check.py` (see section 5) to confirm the
   risk snapshot and scheduler telemetry expose gross exposure and position
   data end-to-end.

## 5. Validation Script

Run the following smoke script from your workstation:

```pwsh
$base = "https://staging-api.<domain>"
Invoke-RestMethod "$base/health"
Invoke-RestMethod "$base/health/scheduler"
Invoke-RestMethod "$base/api/prices/latest?symbol=BTCUSDT"
Invoke-RestMethod "$base/api/ai/signals/latest?limit=5"
python scripts/api_smoke_check.py
```

Confirm responses return HTTP 200 with fresh timestamps. Investigate if the
scheduler snapshot reports `status = degraded` or missing symbols.

## 6. Staging Regression Checklist

Run this short regression plan after every staging deploy:

1. **API exposure** — Execute `python scripts/api_smoke_check.py` from a machine with network access to staging. Confirm it prints "API smoke check completed successfully." and review `/health` manually (e.g. `Invoke-RestMethod "$base/health"`) to verify `scheduler.execution.gross_exposure` and `positions_synced` are present.
2. **Binance testnet execution** — Start the backend with `QT_EXECUTION_EXCHANGE=binance`, `QT_EXECUTION_BINANCE_TESTNET=1`, and a shortened `QUANTUM_TRADER_EXECUTION_SECONDS=60`. Provide staging testnet keys via `config/config.toml` or secret store, ensure the scheduler is enabled, and watch the logs for a completed `execution-rebalance` run. Validate orders on <https://testnet.binance.vision/> and confirm they respect the configured limits.
3. **Docs & collections** — Update any shared Postman collections or runbooks with the latest response fields and upload a short summary to the team channel.

Record the outcome in the shared staging regression log (e.g. the team Runbook tracker referenced in Section 8).

## 7. Observability Expectations

- Dashboards should display API latency, scheduler duration, cache hit rate, and error codes.
- Alerts must include:
  - Scheduler skipped runs (no completion in 2x interval).
  - API error rate > 5% for 5 minutes.
  - Model confidence dropping below configured thresholds.
- Logs should include `request_id`, `symbol`, and `profile` fields for traceability.

## 8. Exit Criteria for Production Promotion

- Staging operates continuously for at least 10 trading days.
- All critical alerts tested (manual trigger or chaos drill).
- Model evaluation report shows metrics meeting acceptance gates.
- Runbooks for incident response and manual shutdown signed off.
- Change approval recorded with deployment summary and risk assessment.
- Staging regression log updated with the latest smoke outcomes and links to supporting evidence.

---

_This guide will evolve as we add telemetry exporters, risk controls, and automation.
Track updates in `docs/ai_production_checklist.md`._
