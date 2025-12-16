# Risk Guard Interface Specification

This document describes the configuration surface, service interfaces, and
fallback behaviour for runtime risk controls that gate automated trading. The
initial scope covers safeguards suitable for staging and early production
rollouts.

## 1. Goals

- Provide explicit controls to stop or throttle trading in adverse conditions.
- Support environment-specific behaviour (staging vs production).
- Ensure guards integrate with telemetry (metrics + logs) and are testable.

## 2. Configuration Surface

Environment variables (or equivalent config keys) to be consumed by the backend:

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `STAGING_MODE` | bool | `0` | Enables relaxed thresholds and dummy executors. |
| `QT_KILL_SWITCH` | bool | `0` | When `1`, prevent all new trade actions. Logged + exposed via health. |
| `QT_MAX_NOTIONAL_PER_TRADE` | decimal | `1000` | Max notional size for a single trade (USD equivalent). |
| `QT_MAX_DAILY_LOSS` | decimal | `500` | Aggregate loss (USD) before trading pauses. |
| `QT_ALLOWED_SYMBOLS` | list | `BTCUSDT,ETHUSDT` | Restrict execution to approved symbols. |
| `QT_FAILSAFE_RESET_MINUTES` | int | `60` | Cooldown before guards auto-reset (optional). |
| `QT_RISK_STATE_DB` | str | `backend/data/risk_state.db` | Path to SQLite file for persisted risk state (absolute or relative to backend). |
| `QT_ADMIN_TOKEN` | str | _unset_ | Shared secret required in `X-Admin-Token` header to access risk admin endpoints. |

Configuration will be centralized in a new module `backend/config/risk.py`. The
module exposes a `RiskConfig` dataclass loaded on startup and cached.

## 3. Runtime Interfaces

### 3.1 RiskGuardService

Create `backend/services/risk_guard.py` exposing:

```python
class RiskGuardService:
    def __init__(self, config: RiskConfig, store: RiskStateStore) -> None: ...

    async def can_execute(self, *, symbol: str, notional: float) -> Tuple[bool, str]:
        """Return (allowed, reason)."""

    async def record_execution(self, *, symbol: str, notional: float, pnl: float) -> None:
        """Update rolling totals after simulated/executed trade."""

    async def reset(self) -> None: ...

    def snapshot(self) -> Dict[str, Any]: ...
```

`RiskGuardService` enforces:

- Kill switch check (`QT_KILL_SWITCH`).
- Allowed symbols filter.
- Per-trade notional cap.
- Daily loss accumulator (rolling 24h window, stored server-side).

It publishes metrics (`qt_risk_denials_total` counter) and logs decisions.

`RiskGuardService` ships with an `SqliteRiskStateStore` that persists the
rolling trade history and kill switch overrides between restarts. For
stateless deployments, swap in another `RiskStateStore` implementation (e.g.
Redis) that satisfies the protocol defined below.

### 3.2 RiskStateStore

Abstract persistence for rolling aggregates; default implementation uses
in-memory store with background persistence to SQLite/Redis later. Define an
interface:

```python
class RiskStateStore(Protocol):
    async def get_records(self) -> Iterable[_TradeRecord]: ...
    async def add_record(self, record: _TradeRecord) -> None: ...
    async def clear(self) -> None: ...
    async def get_kill_switch_override(self) -> Optional[bool]: ...
    async def set_kill_switch_override(self, enabled: Optional[bool]) -> None: ...
```

A basic SQLite-backed store (`backend/services/risk_state_sqlite.py`) will
suffice initially.

## 4. API Integration Points

- Introduce `/risk` admin endpoint returning `RiskGuardService.snapshot()` along
    with config (excluding sensitive values) and allowing overrides/reset.
- Protect `/risk`, `/risk/kill-switch`, and `/risk/reset` with the `X-Admin-Token`
    header sourced from `QT_ADMIN_TOKEN` to prevent unauthorised toggles.
- Update trade execution endpoints to call `can_execute` before placing orders.
- Log denials with request ID, symbol, notional, reason.
- Expose admin endpoint (protected) to toggle kill switch and reset counters.

## 5. Telemetry Hooks

- Counter `qt_risk_denials_total{reason="kill_switch|notional|daily_loss|symbol"}`.
- Gauge `qt_risk_daily_loss` tracking cumulative loss.
- Log events with `event="risk_denial"` and reason.

## 6. Testing Strategy

- Unit tests for `RiskGuardService` covering each guard path.
- Integration test simulating multiple trades exceeding limits.
- API test verifying `/health/risk` exposes expected snapshot.

## 7. Open Items

- Decide whether daily loss window resets at UTC midnight or rolling 24h.
- Define how manual overrides are authenticated (admin token, UI).

---

_Last updated: 2025-11-03. Owner: AI Platform team._
