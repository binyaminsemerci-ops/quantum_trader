# Exit Brain v1 — Phase 1 (Shadow-Only)

> **Version:** 1.0.0-shadow  
> **Status:** Shadow mode — NO execution writes, NO order routing  
> **Owner:** Quantum Trader OS

---

## Purpose

Exit Brain v1 Phase 1 builds the **state + feature foundation** for intelligent exit decisions.
It produces enriched position state, geometry features, and regime analysis — all published
to shadow streams for observation and validation.

**This module does NOT:**
- Send orders
- Write to `trade.intent`, `apply.plan`, `apply.result`, or `exit.intent`
- Modify any live execution state

---

## Architecture

```
Redis (P3.3 snapshots, MarketState, meta.regime, ATR)
        │
        ▼
┌──────────────────────┐
│ PositionStateBuilder  │  ← reads Redis, fail-closed
│ (services/)           │
└──────────┬───────────┘
           │  PositionExitState
           ▼
┌─────────────────┐  ┌─────────────────────┐
│ GeometryEngine  │  │  RegimeDriftEngine   │
│ (engines/)      │  │  (engines/)          │
│ pure math       │  │  pure math           │
└────────┬────────┘  └──────────┬───────────┘
         │                      │
         ▼                      ▼
┌──────────────────────────────────────────┐
│          ShadowPublisher                 │
│ → quantum:stream:exit.state.shadow       │
│ → quantum:stream:exit.geometry.shadow    │
│ → quantum:stream:exit.regime.shadow      │
└──────────────────────────────────────────┘
```

---

## Directory Structure

```
microservices/exit_brain_v1/
├── __init__.py
├── README.md                          ← this file
├── models/
│   ├── __init__.py
│   └── position_exit_state.py         ← PositionExitState dataclass
├── engines/
│   ├── __init__.py
│   ├── geometry_engine.py             ← MFE/MAE/drawdown/momentum/RtR
│   └── regime_drift_engine.py         ← drift detection, trend alignment, reversal/chop risk
├── services/
│   ├── __init__.py
│   └── position_state_builder.py      ← Redis reader, fail-closed assembly
├── publishers/
│   ├── __init__.py
│   └── shadow_publisher.py            ← shadow-only stream writer
└── tests/
    ├── __init__.py
    ├── test_position_exit_state.py
    ├── test_geometry_engine.py
    ├── test_regime_drift_engine.py
    ├── test_position_state_builder.py
    └── test_shadow_publisher.py
```

---

## Key Redis Keys (Input)

| Key Pattern | Source | Content |
|---|---|---|
| `quantum:position:snapshot:<SYMBOL>` | P3.3 | position_amt, side, entry_price, mark_price, unrealized_pnl, leverage, ts_epoch |
| `quantum:position:ledger:<SYMBOL>` | P3.3 | Ledger with updated_at |
| `quantum:marketstate:<SYMBOL>` | MarketStatePublisher | sigma, mu, ts, regime_probs (TREND/MR/CHOP) |
| `quantum:stream:meta.regime` | MetaRegimeService | label (BULL/BEAR/RANGE/VOLATILE/UNCERTAIN) |
| `quantum:atr:<SYMBOL>` | ATR publisher | ATR value |

---

## Shadow Streams (Output)

| Stream | Content | Maxlen |
|---|---|---|
| `quantum:stream:exit.state.shadow` | Full PositionExitState | 5000 |
| `quantum:stream:exit.geometry.shadow` | MFE, MAE, drawdown, PPR, momentum, RtR | 5000 |
| `quantum:stream:exit.regime.shadow` | Label, confidence, alignment, reversal/chop risk | 5000 |

---

## Running Tests

```bash
cd /path/to/quantum_trader
python -m pytest microservices/exit_brain_v1/tests/ -v
```

---

## Assumptions to Verify at Runtime

1. **MarketState key format:** assumed `quantum:marketstate:<SYMBOL>` — verify with `KEYS quantum:marketstate:*`
2. **ATR key format:** builder tries 3 patterns (`quantum:atr:`, `quantum:indicator:atr:`, `atr:`)
3. **open_timestamp:** uses ledger `updated_at` as proxy since P3.3 doesn't store actual entry time
4. **Regime stream format:** expects `label` field in `quantum:stream:meta.regime` entries

---

## Safety Guards

- `shadow_only=True` is **hard-coded**, not configurable
- `ShadowPublisher` has a **forbidden stream blocklist** (trade.intent, apply.plan, etc.)
- All stream names must end in `.shadow` — double-checked at write time
- Engines are **pure functions** — zero IO, zero side effects
- Builder is **fail-closed** — returns `None` on any data issue
