# Data Source Failover Plan

This plan outlines how the backend refresh scheduler and live endpoints should
handle upstream market data outages while maintaining service continuity.

## 1. Objectives

- Detect failures from primary data providers (Binance) quickly.
- Switch to secondary sources (CoinGecko, cached data) with minimal disruption.
- Surface failover state via telemetry and health endpoints.

## 2. Provider Priority

1. **Binance REST/Klines** — primary for OHLCV & live prices.
2. **CoinGecko Markets API** — backup for price snapshots.
3. **Cached historical data** — final fallback to avoid serving errors.

For sentiment, identify secondary provider (placeholder) or degrade gracefully.

## 3. Failure Detection

- Wrap provider calls with:
  - Timeout (3s default) and retry (2 attempts, exponential backoff).
  - Circuit breaker (open after N failures within window).
- Track failure counts per provider in memory and via metrics
  (`qt_provider_failures_total`).

## 4. Failover Flow

Pseudo-logic for scheduler refresh:

```python
for provider in PROVIDER_PRIORITY:
  try:
    data = provider.fetch(symbol)
    record_success(provider)
    return data
  except ProviderError as exc:
    record_failure(provider, exc)
    continue

use_cached_data(symbol)
raise ProviderUnavailable if no cache
```

All failure events should be logged and surfaced in `/health/scheduler` under
`last_run.errors` with provider context.

## 5. Circuit Breaker Details

- Maintain per-provider state: `failure_count`, `opened_at`, `cooldown_seconds`.
- When circuit is open, skip provider and log `skipped_due_to_circuit`.
- Auto-close after cooldown if a lightweight HEAD check succeeds.

## 6. Telemetry Extensions

- Counter `qt_provider_failures_total{provider}`.
- Gauge `qt_provider_circuit_open{provider}` (0/1).
- Histogram `qt_provider_latency_seconds{provider}`.

> **Implementation note:** Scheduler snapshot now surfaces provider success/failure
> counts and circuit status. Metrics emission remains a follow-up task aligned
> with the telemetry plan.

Alerts:

- Provider failure rate > 80% for 10 minutes → notify on-call.
- Circuit open longer than 15 minutes → escalate.

## 7. Health Endpoint Additions

Extend `/health/scheduler` payload to include:

```json
{
  "providers": {
    "binance": {"status": "ok", "failure_count": 0, "circuit_open": false},
    "coingecko": {"status": "degraded", "failure_count": 3, "circuit_open": true}
  }
}
```

## 8. Testing Checklist

- Unit tests for provider fallback order.
- Tests for circuit breaker state transitions.
- Integration test simulating outage to ensure cached data returned.
- Health endpoint test verifying provider status render.

## 9. Implementation Steps

1. Create provider abstraction (`DataProvider` protocol) with fetch contracts.
2. Implement Binance + CoinGecko adapters conforming to protocol.
3. Introduce circuit breaker utility (time-based).
4. Update scheduler to iterate providers using plan above.
5. Instrument telemetry counters/gauges.
6. Update `/health/scheduler` to include provider snapshot.
7. Write unit/integration tests covering failover scenarios.

---

_Last updated: 2025-11-03. Owner: AI Platform team._
