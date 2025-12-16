# Risk Guard Failover Runbook

This runbook describes the configuration knobs and drill steps for pausing
order flow when market data becomes suspect or trading venues experience
instability.

## Price sanity guard configuration

Configure the price-validation layer via environment variables before
promotion:

- `QT_MIN_UNIT_PRICE` — optional per-order price floor. Orders priced below this
  threshold are rejected.
- `QT_MAX_UNIT_PRICE` — optional per-order ceiling. Orders priced above this
  threshold are rejected.
- `QT_MAX_PRICE_STALENESS_SECONDS` — optional freshness guard. Any price snapshot
  older than the configured number of seconds is treated as stale and blocked.

All limits apply to both automated rebalances and manual `POST /trades`
requests.

## Emergency shutdown drill

Perform the following steps at least once per quarter to ensure the team can
halt execution safely:

1. Announce the drill in the incident channel and enable the kill switch via
   `QT_KILL_SWITCH=1` or `POST /risk/kill-switch {"enabled": true}`. Confirm the
   override in `GET /risk` and check `/metrics` for a `qt_risk_denials_total`
   increment with `reason="kill_switch"`.
2. Attempt a manual order (`POST /trades`) to verify a `403 RiskCheckFailed`
   response, demonstrating that order flow is blocked platform-wide.
3. Restore normal trading by clearing the override
   (`POST /risk/kill-switch {"enabled": null}`) and monitor the next scheduler
   execution for fresh price snapshots.
4. Record start/end timestamps, observed metrics, and any manual remediation
   steps in the operations log so future drills can benchmark response time.

## Post-drill checklist

- Kill switch returned to its configured state.
- Execution and scheduler logs show healthy runs post-drill.
- Prometheus counters (`qt_risk_denials_total`, `qt_model_inference_duration_seconds`)
  align with expectations gathered during the drill.
- Runbook updated with lessons learned.
