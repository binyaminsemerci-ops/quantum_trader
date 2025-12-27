# Telemetry & Alerting Plan

This plan enumerates the metrics, logs, and alerts required to operate the AI
trading services safely in staging and production. Instrumentation work should
follow the steps below, prioritized by criticality.

## 1. Metrics Catalogue

| Metric Name | Type | Labels | Description | Target Source |
|-------------|------|--------|-------------|---------------|
| `qt_api_request_duration_seconds` | Histogram | `route`, `method`, `status_code` | End-to-end latency for FastAPI requests. | FastAPI middleware + Prometheus client |
| `qt_api_request_total` | Counter | `route`, `method`, `status_code` | Request throughput and error counting. | Same middleware |
| `qt_scheduler_run_duration_seconds` | Histogram | `job_id`, `status` | Duration per scheduler run; `status=success|degraded`. | `utils.scheduler` instrumentation |
| `qt_scheduler_runs_total` | Counter | `job_id`, `status` | Count of scheduler executions and failure modes. | `utils.scheduler` |
| `qt_cache_hits_total` | Counter | `cache_name`, `result` | Cache hit/miss counts for price & signal caches. | `utils.cache` wrappers |
| `qt_model_inference_duration_seconds` | Histogram | `profile`, `model_version` | Time spent generating AI signals. | AI model service interface |
| `qt_model_confidence` | Gauge | `profile`, `model_version` | Latest average confidence per profile. | Signal pipeline |
| `qt_trade_decisions_total` | Counter | `symbol`, `action` | Number of trades recommended/placed per symbol. | Trading executor (future) |
| `qt_retrain_runs_total` | Counter | `status` | Retraining outcomes (success/failure). | Retrain orchestration |
| `qt_retrain_duration_seconds` | Histogram | `status` | Duration of retraining jobs. | Retrain orchestration |
| `qt_risk_denials_total` | Counter | `reason` | Count of risk guard denials by reason. | Risk guard service (`record_risk_denial`) |
| `qt_risk_daily_loss` | Gauge | `-` | Rolling 24h realised loss tracked by risk guard. | Risk guard service (`set_risk_daily_loss`) |

### Metric Implementation Notes

- Adopt `prometheus_client` for now (simple to wire into FastAPI). We can add
  OpenTelemetry exporters later if the stack standardizes on OTLP.
- Store metric helpers in `backend/utils/telemetry.py` (to be created) to avoid
  scattering registry logic.
- Respect `STAGING_MODE` to enable extra debug labels without polluting prod metrics.
- Risk guard metrics derive from the persisted `RiskGuardService`; ensure the
  Prometheus registry is initialised before the service updates kill switch
  overrides or trade history so gauges reflect the SQLite-backed state.

## 2. Logging Strategy

- Standardize on structured JSON logs with fields: `timestamp`, `level`,
  `message`, `request_id`, `symbol`, `profile`, `job_id`, `model_version`.
- Add request/response logging middleware that injects a UUID request ID.
- Emit explicit log events for:
  - Scheduler run start/finish with summary (already partially in place).
  - Cache fallback activation (source fails, using cached data).
  - Retraining job lifecycle events (start, success, failure).
- Ship logs to Loki/ELK via sidecar or fluentbit; in local dev continue printing JSON to stdout.

## 3. Alert Definitions

| Alert | Condition | Severity | Action |
|-------|-----------|----------|--------|
| Scheduler stalled | `qt_scheduler_runs_total` absent for 2× interval OR `status=degraded` rate > 0.2 in 15m | High | Page on-call, check `/health/scheduler`, restart scheduler if needed |
| API error spike | `qt_api_request_total{status_code=~"5.."}` rate > 5% for 5m | High | Investigate backend logs, roll back recent deploy |
| Cache misses | `qt_cache_hits_total{result="miss"}` exceeds 50% for 10m | Medium | Review data-source availability, consider failover |
| Model confidence drop | `qt_model_confidence` < threshold (e.g. 0.55) for 30m | Medium | Trigger manual review of incoming data & model |
| Retraining failure | `qt_retrain_runs_total{status="failure"}` increments | High | Inspect retrain logs, halt promotions |

Alerts should be codified in the observability platform (Grafana Alerting or Azure Monitor).

## 4. Dashboard Layout

1. **Service Overview**
   - API latency heatmap
   - Error rate graph
   - Cache hit ratio gauge
   - Scheduler next-run countdown

2. **Model Performance**
   - Confidence trend per profile
   - Trade decision counts (buy/sell/hold)
   - Retrain job timeline with statuses

3. **Data Source Health**
   - External API latency (Binance/CoinGecko) via synthetic checks
   - Failover events log (see Section 5 roadmap)

## 5. Implementation Roadmap

1. Create telemetry module with Prometheus registry and expose `/metrics` endpoint. (Done — FastAPI app now exposes `/metrics` via `backend/utils/telemetry.py`).
2. Instrument FastAPI middleware for request counters/histograms. (Done — `PrometheusMiddleware` wraps the ASGI app).
3. Extend `utils.scheduler` to emit counters/histograms defined above. (Done — `warm_market_caches` and provider helpers emit run and provider metrics).
4. Wrap cache reads/writes to increment hit/miss counters.
5. Add model inference instrumentation inside signal generation pipeline.
6. Prototype Grafana dashboards using staging data.
7. Configure alert rules aligned with table above.
8. Document runbooks for responding to each alert.

## 6. Dependencies & Tooling

- Python packages: `prometheus-client`, optional `structlog` for JSON logging.
- Dashboards: Grafana or Azure Dashboard.
- Alert routing: PagerDuty/Teams/Slack integration via observability platform.

## 7. Open Questions

- Do we standardize on OpenTelemetry across backend + frontend? If yes, we should
  implement OTLP exporters instead of raw Prometheus.
- Where should synthetic checks live (separate worker vs existing scheduler)?
- How granular should symbol-level metrics be (risk of cardinality explosion)?

---

_Last updated: 2025-11-03. Owner: AI Platform team._
