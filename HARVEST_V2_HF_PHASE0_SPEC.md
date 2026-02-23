# HARVEST V2 HF — PHASE 0: FULL TECHNICAL SPECIFICATION
## SOURCE OF TRUTH — DO NOT DEVIATE WITHOUT EXPLICIT REVISION

**Version:** 1.0  
**Date:** 2026-02-22  
**Scope:** Exit Engine V2 (shadow-mode only until Phase 7 live switch)  
**Branch:** main  
**Author:** Rollout sequence — binyaminsemerci-ops  

---

## 0. OVERVIEW

Harvest V2 HF is a complete rewrite of the exit engine.  
It runs in **shadow mode only** until Phase 7.  
It writes **zero** to `apply.plan`, `trade.intent`, or any execution stream.  
It reads live production data (positions, ATR, heat) without side effects.

---

## 1. INPUTS

### 1.1 Position Snapshot
Source: Redis  
Key pattern: `quantum:position:{SYMBOL}` (HGETALL)

| Field | Type | Notes |
|-------|------|-------|
| `symbol` | str | e.g. `AIUSDT` |
| `side` | str | `LONG` or `SHORT` |
| `quantity` | float | absolute position size |
| `entry_price` | float | average fill price |
| `unrealized_pnl` | float | mark-price PnL in USDT |
| `leverage` | int | effective leverage |
| `atr_value` | float | ATR at scan time (updated by ATR provider) |
| `volatility_factor` | float | vol scalar at entry (default 1.0) |
| `entry_risk_usdt` | float | pre-computed initial risk in USDT |
| `risk_price` | float | ATR-based stop distance at entry |
| `risk_missing` | int | 1 if entry_risk_usdt could not be computed |
| `sync_timestamp` | unix_ts | last update epoch seconds |

**Validity check:** Skip position if ANY of the following are true:
- `entry_risk_usdt <= 0` OR `risk_missing == "1"`
- `atr_value <= 0`
- `sync_timestamp` is older than `CONFIG.max_position_age_sec` (default 120s)

### 1.2 ATR — Hybrid Provider
Source: `quantum:position:{SYMBOL}` field `atr_value` (live scanning value)  
Rolling window: last `CONFIG.atr_window` values per symbol, maintained in V2 state ring buffer  
Percentile: computed over the ring buffer each tick

**ATR percentile regime thresholds:**

| Regime | Percentile range |
|--------|-----------------|
| `LOW_VOL` | < 20 |
| `MID_VOL` | 20 – 80 |
| `HIGH_VOL` | > 80 |

ATR is considered **invalid** if `atr_value == 0` or position age > 120s.  
On invalid ATR → **skip position entirely**, do not emit, do not update state.

### 1.3 Heat
Source: Redis hash `quantum:capital:state`  
Field: `heat` (HGET quantum:capital:state heat)  
Type: float in range [0.0, 1.0]  
**Default: 0.0 if key or field is missing** (non-negotiable safety rule)

### 1.4 Config
Source: Redis hash `quantum:config:harvest_v2` (HGETALL)  
Loaded once at startup, refreshed every `CONFIG.config_refresh_interval_sec` (default 60s)

Config fields and defaults:

```
# R thresholds
r_stop_base             = 0.5        # Hard stop at R_net <= this
r_target_base           = 3.0        # Full-close target
trailing_step           = 0.3        # Pullback from max_R_seen triggers FULL_CLOSE
r_emit_step             = 0.05       # Minimum R change to force re-emit same decision

# Partial scaling thresholds
partial_25_r            = 1.0        # First partial: close 25%
partial_50_r            = 1.5        # Second partial: close 50% of remainder
partial_75_r            = 2.0        # Third partial: close 75% of remainder

# Vol factors per regime
vol_factor_low          = 0.7        # Tighten stops/targets in low vol
vol_factor_mid          = 1.0        # Baseline
vol_factor_high         = 1.4        # Widen in high vol

# Heat sensitivity
heat_sensitivity        = 0.5        # How much heat reduces R_target

# ATR rolling window
atr_window              = 50         # Ticks to maintain for percentile

# Timing
config_refresh_interval_sec = 60
max_position_age_sec        = 120
scan_interval_sec           = 2.0

# Shadow stream
stream_shadow           = quantum:stream:harvest.v2.shadow

# Metrics key
metrics_key             = quantum:metrics:harvest_v2
```

---

## 2. CORE METRICS

### 2.1 R_net

```
initial_risk = position.entry_risk_usdt          # pre-stored in Redis, USDT
R_net        = position.unrealized_pnl / initial_risk
```

Constraint: if `initial_risk <= 0` → skip position (log `SKIP_INVALID_RISK`).

### 2.2 Regime Detection

On each tick per symbol:
1. Append `position.atr_value` to symbol ring buffer (max `atr_window` entries)
2. Compute percentile rank of current `atr_value` within the ring buffer
3. Assign regime:

```python
if   pct < 20:  regime = "LOW_VOL"
elif pct < 80:  regime = "MID_VOL"
else:           regime = "HIGH_VOL"
```

Ring buffer is **in-memory only** — not persisted to Redis.  
On cold start, ring buffer is empty → regime defaults to `MID_VOL` until `atr_window` samples collected.

### 2.3 Vol Factor

```python
vol_factor = {
    "LOW_VOL":  CONFIG.vol_factor_low,
    "MID_VOL":  CONFIG.vol_factor_mid,
    "HIGH_VOL": CONFIG.vol_factor_high,
}[regime]
```

### 2.4 Heat Factor

```python
heat = redis.hget("quantum:capital:state", "heat") or 0.0
heat = max(0.0, min(1.0, float(heat)))   # clamp always
```

### 2.5 Dynamic Stop

```
R_stop_effective = CONFIG.r_stop_base × vol_factor × heat_factor
```

`heat_factor` is **NOT applied to stop** — stop is pure vol-scaled.  
Final formula:

```
R_stop_effective = CONFIG.r_stop_base × vol_factor
```

> Rationale: stop tightening is via volatility only; heat affects targets.

### 2.6 Dynamic Target

```
R_target_effective = CONFIG.r_target_base × vol_factor × (1 - heat × CONFIG.heat_sensitivity)
```

### 2.7 Trailing Stop

Condition for `FULL_CLOSE` via trailing:

```python
if state.max_R_seen is not None:
    if R_net < (state.max_R_seen - CONFIG.trailing_step):
        decision = "FULL_CLOSE"   # trailing stop triggered
```

`max_R_seen` is updated every tick:

```python
if R_net > (state.max_R_seen or R_net):
    state.max_R_seen = R_net
```

---

## 3. DECISION LOGIC

Evaluated in strict priority order (first match wins):

```
1.  R_net <=  R_stop_effective          → FULL_CLOSE   (stop hit)
2.  Trailing check (see 2.7)            → FULL_CLOSE   (trailing pullback)
3.  R_net >=  r_target_base             → FULL_CLOSE   (target hit)
4.  R_net >= partial_75_r  AND state.partial_stage < 3  → PARTIAL_75
5.  R_net >= partial_50_r  AND state.partial_stage < 2  → PARTIAL_50
6.  R_net >= partial_25_r  AND state.partial_stage < 1  → PARTIAL_25
7.  Otherwise                           → HOLD
```

`state.partial_stage` tracks highest partial already emitted (0=none, 1=25%, 2=50%, 3=75%).  
Partial stages are **monotonically increasing** — never decremented.

---

## 4. EMISSION GUARD

Emit only if **either** condition is true:

```
CONDITION A:  decision != state.last_decision
CONDITION B:  abs(R_net - state.last_emit_R) > CONFIG.r_emit_step
```

If neither condition is true → `HOLD_SUPPRESSED` (counted in metrics, not emitted).

---

## 5. REDIS STATE CONTRACT

Key: `quantum:harvest_v2:state:{SYMBOL}` (Redis Hash)  
TTL: none (persistent — cleared only on full reset)

| Field | Type | Description |
|-------|------|-------------|
| `max_R_seen` | float | Peak R ever seen for this position lifecycle |
| `partial_stage` | int | 0/1/2/3 — highest partial tier emitted |
| `last_decision` | str | Last emitted decision string |
| `last_emit_R` | float | R_net at last emission |
| `last_update_ts` | float | Unix timestamp of last write |

**Write rule:** State is written **after** emission guard passes and emission succeeds.  
**Read rule:** On missing key → initialize defaults: `max_R_seen=None, partial_stage=0, last_decision=None, last_emit_R=None`.

---

## 6. SHADOW OUTPUT CONTRACT

Stream: `quantum:stream:harvest.v2.shadow`  
Write with: `XADD quantum:stream:harvest.v2.shadow * field value ...`  
Trim: `MAXLEN ~ 50000`

### Mandatory payload fields:

| Field | Type | Example |
|-------|------|---------|
| `symbol` | str | `AIUSDT` |
| `side` | str | `LONG` |
| `R_net` | float (4dp) | `1.2345` |
| `R_stop` | float (4dp) | `0.5250` |
| `R_target` | float (4dp) | `2.8000` |
| `regime` | str | `MID_VOL` |
| `heat` | float (4dp) | `0.0000` |
| `decision` | str | `PARTIAL_25` |
| `partial_stage` | int | `1` |
| `max_R_seen` | float (4dp) | `1.2345` |
| `initial_risk` | float (4dp) | `10.0054` |
| `unrealized_pnl` | float (4dp) | `12.3456` |
| `atr_value` | float (6dp) | `0.000450` |
| `vol_factor` | float (4dp) | `1.0000` |
| `emit_reason` | str | `DECISION_CHANGE` or `R_STEP` |
| `timestamp` | float | Unix epoch with ms |
| `v2_version` | str | `2.0.0` |

No optional fields. All fields required on every emission.

---

## 7. METRICS CONTRACT

Key: `quantum:metrics:harvest_v2` (Redis Hash)  
Updated atomically with HINCRBY / HSET per tick.

| Field | Increment | Description |
|-------|-----------|-------------|
| `ticks` | +1/tick | Total scan iterations |
| `evaluated` | +1/position evaluated | Positions that passed validity check |
| `skipped_invalid` | +1 | Positions skipped (invalid risk/ATR/staleness) |
| `full_closes` | +1 | FULL_CLOSE decisions emitted |
| `partials` | +1 | PARTIAL_* decisions emitted |
| `holds` | +1 | HOLD decisions (emission guard passed) |
| `hold_suppressed` | +1 | HOLD_SUPPRESSED (guard blocked) |
| `divergence_from_v1` | +1 | When v2 decision != v1 would decide (Phase 5) |
| `avg_R_sum` | +float | Accumulator for avg_R calc |
| `avg_R_count` | +1 | Denominator for avg_R |
| `regime_low` | +1 | LOW_VOL ticks |
| `regime_mid` | +1 | MID_VOL ticks |
| `regime_high` | +1 | HIGH_VOL ticks |
| `last_tick_ts` | SET | Unix timestamp of last tick |
| `start_ts` | SET once | Unix timestamp of service start |

---

## 8. SAFETY RULES — NON-NEGOTIABLE

These rules are enforced by code structure, not by flags.  
There is no runtime switch that can bypass them in V2.

| Rule | Implementation |
|------|---------------|
| **No `apply.plan` writes** | `STREAM_PLANS` not imported; xadd to `apply.plan` is absent from codebase |
| **No governor interaction** | No import of governor client; no reads from governor state |
| **No execution interaction** | No writes to `quantum:stream:trade.intent` or `quantum:stream:execution.*` |
| **No kill-switch bypass** | Harvect V2 does NOT read `quantum:kill`; it has no execution path to gate |
| **ATR must be valid** | `atr_value <= 0` → hard skip, no state update, no emission |
| **Heat defaults to 0** | `heat = float(redis.hget(...) or 0.0)` — no exception can raise heat above 1.0 |
| **Shadow only** | Only output is `quantum:stream:harvest.v2.shadow` |
| **No position mutation** | V2 never writes to `quantum:position:*` keys |
| **Read-only Redis positions** | All position reads are HGETALL; no HSET/HDEL on position keys |

---

## 9. BOUNDARY CONDITIONS

| Condition | Handling |
|-----------|---------|
| `entry_risk_usdt == 0` | Skip, log `SKIP_ZERO_RISK`, increment `skipped_invalid` |
| `atr_value == 0` | Skip, log `SKIP_ZERO_ATR`, increment `skipped_invalid` |
| Position age > `max_position_age_sec` | Skip, log `SKIP_STALE`, increment `skipped_invalid` |
| `quantum:capital:state` missing | `heat = 0.0` — normal operation continues |
| `quantum:config:harvest_v2` missing | Use all defaults, log `CONFIG_MISSING_USING_DEFAULTS` once |
| Ring buffer empty (cold start) | `regime = MID_VOL`, `vol_factor = 1.0` |
| Redis connection lost | Service exits with error code 1, systemd restarts |
| `unrealized_pnl` = 0 | Normal evaluation — R_net = 0, outcome depends on stop level |
| `partial_stage == 3` AND `R_net >= partial_75_r` | Already at max partial → evaluate only HOLD, FULL_CLOSE, trailing |

---

## 10. LOG FORMAT CONTRACT

Log level: `INFO` for all emit/skip decisions.  
Log line format (structured one-liner):

```
[HV2] {SYMBOL} R={R_net:.3f} regime={regime} decision={decision} emit={true|false} reason={reason}
```

Tick-level summary every tick:

```
[HV2_TICK] ts={ts} scanned={n} evaluated={e} emitted={em} skipped_invalid={si} hold_suppressed={hs}
```

---

## 11. PHASE GATES

This spec governs Phases 1–4.  
Phases 5–7 extend but do not modify this spec.

| Phase | Action | Modifies spec? |
|-------|--------|---------------|
| 1 | Write complete code | No |
| 2 | Redis schema + test harness | No |
| 3 | systemd unit + deploy script | No |
| 4 | Shadow run (write to shadow stream, verify) | No |
| 5 | Comparison vs V1 divergence metrics | Addends divergence calc only |
| 6 | Chaos/safety validation | No |
| 7 | Live switch — add apply.plan write, remove shadow flag | Yes — Phase 7 amendment required |

**Phase 7 requires a separate amendment document before any apply.plan write is added.**

---

## 12. REVISION LOG

| Date | Change | Author |
|------|--------|--------|
| 2026-02-22 | Initial spec — PHASE 0 baseline | Rollout sequence |

---

*End of PHASE 0 — SOURCE OF TRUTH*
