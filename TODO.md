# Quantum Trader – Autonomous Trading Backlog & Roadmap

This backlog reflects the shift from a demo snapshot to a fully autonomous, self‑optimising AI trading platform.
Tasks are grouped by execution horizon. Keep each item granular enough to turn into a PR / issue.
Mark state with: [ ] planned · [~] in progress · [x] done · [↺] revisit/refactor.

---

## 0. Guiding Principles

- Safety first: no live capital risk until risk & audit gates pass.
- Deterministic reproducibility: every model + backtest must be reproducible from metadata.
- Observability as a feature: every autonomous action emits structured events.
- Progressive hardening: expand autonomy only after previous layer passes objective metrics.

---

## 1. Immediate (Days – Week 1)

- [ ] Config unification: single `Config` (pydantic) exported to frontend (public vs sensitive split).
- [ ] AI model registry: version + hash + training params persisted (SQLite table `model_registry`).
- [ ] Feature parity tests: unit tests asserting no NaN / drift in `feature_engineer` outputs.
- [ ] Backtest metrics expansion: Sharpe, Sortino, max drawdown, hit ratio, avg hold time.
- [ ] Risk baseline enforcement: max open positions + per-position notional limit + cooldown after loss.
- [ ] WebSocket consolidation: unify watchlist & alerts into multiplexed channel with type field.
- [ ] README links: cross-link to `ARCHITECTURE.md` & AI docs (ensure no dead anchors).
- [ ] TODO auto-check: lightweight script verifying unchecked tasks sorted by horizon.
- [ ] Cleanup orphan legacy files (list + delete PR): leftover `.bak`, `served-*` duplicates.

## 2. Short-Term (Weeks 2–4)

### AI & Strategy

- [ ] Multi-horizon feature set (5m / 15m / 1h aggregation pipeline).
- [ ] Label refinement: future return horizon configurable; volatility-adjusted targets.
- [ ] Strategy abstraction: interface for pluggable signal generators (ML / rules / ensemble).
- [ ] Ensemble scaffolding: weighted blending of base learners (XGBoost + simple momentum rule).

### Trading Engine

- [ ] Position state table (entry_time, entry_price, size, stop, take_profit, status, pnl_realised).
- [ ] Execution simulator v1 (slippage model + fee model) for offline evaluation.
- [ ] Trade audit log (append-only) with hash-chain for tamper detection.

### Data & Ingestion

- [ ] Sentiment ingestion stub (CryptoPanic or RSS) -> normalised sentiment score cache.
- [ ] Candle gap detector + auto backfill.

### Observability / Ops

- [ ] Structured event schema (JSON) for: order_decision, trade_fill, retrain_start, model_promote.
- [ ] Prometheus exporter (latency, cache hit ratio, training duration, signal freshness).

### Frontend UI

- [ ] AI Model panel: current model id, trained_at, metrics, promote/rollback buttons (mock actions initially).
- [ ] Live P&L chart (equity curve) using simulated fills.

### Tooling

- [ ] Deterministic seed propagation (Python, NumPy, XGBoost) via central util.
- [ ] CLI: `qtctl retrain --limit 2000 --tag exp1` + `qtctl promote <model_id>`.

## 3. Mid-Term (Months 2–3)

### Advanced AI

- [ ] Genetic strategy tuner: population = parameterised strategies, fitness = risk-adjusted return.
- [ ] Reinforcement Learning prototype (offline replay using simulated fills).
- [ ] Drift detection: feature distribution & performance drift alert.
- [ ] Regime classification (volatility clustering / HMM) to condition model choice.

### Real Trading Enablement

- [ ] Exchange abstraction: real vs paper vs simulator (strategy uses common interface).
- [ ] Testnet order placement (Binance) with idempotent retries + clock skew handling.
- [ ] Risk Layer v2: dynamic position sizing (volatility + Kelly ceiling) + global exposure cap.
- [ ] Kill-switch: disable trading if drawdown > threshold or heartbeat missing.

### Data

- [ ] On-chain metrics integration (placeholder adaptor + caching).
- [ ] Market microstructure features (spread, order book imbalance) – sampling prototype.

### Observability

- [ ] Grafana dashboards (trading loop latency, model versions, drift metrics).
- [ ] Alerting rules (Slack/webhook) for: missed heartbeat, model drift, anomalous loss cluster.

### Frontend

- [ ] Strategy evolution UI: population snapshot, fitness charts.
- [ ] Trade blotter with filtering & export (CSV/JSON).

## 4. Strategic / R&D (Months 3+)

- [ ] Online learning / partial fit loop (evaluate stability constraints first).
- [ ] Cross-exchange arbitrage module feasibility study.
- [ ] Portfolio optimiser (mean-variance / risk parity baseline) guiding allocation.
- [ ] Latency-aware execution tactics (TWAP/VWAP prototype) for larger notional sizing.
- [ ] Anomaly detection on returns & slippage distribution.
- [ ] Auto-hyperparameter tuning service (Optuna) feeding model registry.

---

## Cross-Cutting Concerns

- [ ] Security: secrets isolation (env separation + scanning), add secret detection pre-commit hook.
- [ ] Compliance / logging: time-synchronised (UTC) structured logs + retention policy.
- [ ] Performance baseline: profile training & signal path; set SLO targets (p95 signal latency < 150ms).
- [ ] Reliability: chaos test – simulate exchange downtime & degraded latency.
- [ ] Documentation: update `ARCHITECTURE.md` with autonomous lifecycle & state machines.

---

## Quality Gates (Definition of “Production-Ready” Trading)

- [ ] 4 consecutive weeks stable simulated performance (configurable metrics thresholds).
- [ ] All critical risk controls active (position caps, kill-switch, audit log hash verified).
- [ ] Model registry complete (version lineage + reproducible artifact set).
- [ ] Alert coverage: drift, heartbeat, drawdown, exception spikes.
- [ ] Recovery drill documented (cold start from empty DB + latest model artifact).

---

## Metrics To Track (Implement incrementally)

- Signal freshness (seconds since feature snapshot)
- Training duration & queue wait time
- Equity curve stats (rolling Sharpe, drawdown)
- Fill quality (slippage vs mid / expected)
- Drift scores (PSI / KL on selected features)
- Cache hit ratio (signals / features)

---

## Archive (Completed / Legacy)

- [x] Secrets & configuration centralisation
- [x] Dependency hygiene (runtime vs dev split, optional extras)
- [x] CI policy stratification (fast vs heavy jobs)
- [x] Real adapters (prices/signals via ccxt)
- [x] Frontend cleanup of duplicate files
- [x] Training pipeline initial wiring + docs
- [x] Observability baseline (structured logging + heartbeat)

---

## Working Style Notes

- Prefer opening an issue referencing the checklist item before a large PR.
- Keep PRs < ~500 LOC diff when feasible; large refactors split by subsystem.
- Every new model-affecting change: add or extend a test (feature shape, training determinism, metrics expectation).

---

## Next Up (Rolling Focus Queue)

Populate dynamically (move top immediate items here when starting sprint):

- [ ] (slot 1)
- [ ] (slot 2)
- [ ] (slot 3)

---

_Synchronise README ↔ this file on each planning iteration._
