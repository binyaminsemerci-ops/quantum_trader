# AI Production Readiness Checklist

This checklist breaks down the work required to operate the AI trading stack in a
production-like environment with staged rollout, monitoring, and guardrails.

## 1. Staging Environment & Live Data Validation

- [ ] Provision a staging environment that mirrors production settings.
- [ ] Connect staging to live market data using dedicated API keys (paper trading/beta).
- [ ] Route trade execution to a simulator or paper trading broker.
- [ ] Exercise end-to-end workflow (ingest → predict → mock order) for at least two weeks.
- [ ] Capture telemetry (latency, error codes, cache hits) during the staging run.

## 2. Monitoring, Telemetry, and Alerting

- [ ] Instrument scheduler metrics (success rate, duration, next run) via Prometheus/Otel.
- [ ] Emit per-signal/model metrics (confidence, drift, PnL) to dashboards.
- [ ] Configure alerts for data source outages, high error rates, and stale caches.
- [ ] Ship structured logs to centralized storage with searchable request IDs.
- [ ] Create on-call dashboards summarizing health endpoints and key KPIs.

## 3. Risk Controls and Guardrails

- [x] Implement kill-switch and max daily loss guard.
- [x] Enforce position sizing / max trade notional per asset.
- [x] Persist risk guard state to SQLite and expose admin override endpoints.
- [ ] Add sanity-checks for anomalous prices before trade submission.
- [ ] Simulate failure scenarios (API downtime, latency spikes) and verify failover.
- [ ] Document playbooks for emergency shutdown and recovery.

## 4. Model Lifecycle Governance

- [ ] Automate data ingestion, validation, and feature generation pipelines.
- [ ] Track every training run (hyperparameters, metrics, dataset hashes).
- [ ] Require hold-out evaluation and drift analysis before promotion.
- [ ] Archive model artifacts with version tags and rollback scripts.
- [ ] Expose model status/version via API endpoint for frontend visibility.

## 5. Operational Runbooks & Compliance

- [ ] Expand backend README with staging/production runbooks and troubleshooting.
- [ ] Maintain checklists for deployments, incident response, and routine operations.
- [ ] Review regulatory/compliance considerations for automated trading in target regions.
- [ ] Ensure secrets management, access controls, and audit logging meet company policy.
- [ ] Conduct a "game day" rehearsal prior to first production cutover.

---

### Next actionable steps

1. Draft the staging deployment guide (environments, credentials, paper trading configuration). **Done** — see `docs/staging_deployment_guide.md`.
2. Introduce scheduler state exposure (completed via `/health/scheduler`). **Done**.
3. Outline telemetry integration plan (metric names, exporters, dashboard layout). **Done** — see `docs/telemetry_plan.md`.
4. Design risk guard interfaces (service contracts, configuration format). **Done** — see `docs/risk_guard_spec.md` (implementation pending).
5. Plan data-source failover workflow and instrumentation. **Done** — see `docs/data_source_failover_plan.md`.
6. Implement risk guard service and failover logic. **Done** — risk guard wired into trade endpoints with health exposure; scheduler now tracks provider failover.
7. Persist risk guard state and add admin controls. **Done** — SQLite-backed store with `/risk` admin APIs.
8. Produce repeatable staging smoke validation (health + exposure). **Done** — see `scripts/api_smoke_check.py`.
9. Promote latest Alembic migrations and schema snapshot through staging/production. **Pending** — schedule downtime window and run `alembic upgrade head` in each environment.
