# P3.4 Position Reconciliation Engine

**Version:** 1.0.0  
**Status:** Production Ready  
**Purpose:** Evidence-driven position reconciliation with auto-repair and hold gating

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [Auto-Repair Policy](#auto-repair-policy)
5. [Hold Logic](#hold-logic)
6. [Redis Keys](#redis-keys)
7. [Metrics](#metrics)
8. [Configuration](#configuration)
9. [Deployment](#deployment)
10. [Monitoring](#monitoring)
11. [Troubleshooting](#troubleshooting)
12. [Safety Considerations](#safety-considerations)

---

## Overview

P3.4 is the **Position Reconciliation Engine** - a continuous, fail-closed system that:

- **Detects drift** between Exchange position, Internal ledger, and Apply results
- **Auto-repairs** when safe (with evidence from executed orders)
- **Sets HOLDs** when unsafe (no evidence, side mismatch, stale data)
- **Integrates with P3.3** to block execution when reconciliation required
- **Never places orders** - only repairs internal ledger state

### Position in Stack

```
P3.2 Governor       → "Can we trade?" (limits, circuit breakers)
P3.3 Position Brain → "Does state match RIGHT NOW?" (sanity)
P3.4 Reconcile      → "Why doesn't state match - can we repair?" (reconciliation)
```

### Key Principle

**P3.4 NEVER sends orders to exchange.**  
It only repairs `quantum:position:ledger` to match reality when evidence supports it.

---

## Architecture

### Components

1. **Reconciliation Loop** (1s cadence)
   - Iterate through allowlist symbols
   - Read exchange snapshot (from P3.3)
   - Compare with internal ledger
   - Detect drift and take action

2. **Evidence Collection**
   - Read last 50 entries from `quantum:stream:apply.result`
   - Build 5-minute evidence window
   - Match filled_qty to observed drift

3. **Auto-Repair Logic**
   - Allow repair ONLY when:
     * Exchange snapshot fresh (< 10s)
     * NO side mismatch
     * Drift small & plausible (< 1% tolerance)
     * Evidence exists (recent executed order with matching filled_qty)

4. **Hold Logic**
   - Set `quantum:reconcile:hold:<symbol>` (TTL 300s) when:
     * Side mismatch (LONG vs SHORT)
     * Stale exchange data (> 10s old)
     * Qty mismatch without evidence
     * Missing exchange snapshot

5. **P3.3 Integration**
   - P3.3 checks `quantum:reconcile:hold:<symbol>` before issuing ALLOW
   - If hold exists → P3.3 DENY with reason="reconcile_hold_active"

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Exchange Snapshot (from P3.3)                               │
│ quantum:position:snapshot:BTCUSDT                           │
│ {position_amt, side, entry_price, mark_price, ts_epoch}    │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ P3.4 Reconcile Engine (1s loop)                             │
│ • Read exchange snapshot                                     │
│ • Read internal ledger                                       │
│ • Detect drift (side/qty mismatch)                          │
│ • Check snapshot freshness                                   │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Evidence Collection                                          │
│ • Read quantum:stream:apply.result                          │
│ • Find recent executed orders                                │
│ • Match filled_qty to drift                                  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
        ┌────────┴────────┐
        │ Safe to repair? │
        └────────┬────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
    ┌──────┐          ┌──────┐
    │ YES  │          │ NO   │
    └───┬──┘          └───┬──┘
        │                 │
        ▼                 ▼
┌────────────────┐  ┌─────────────────┐
│ Auto-Repair    │  │ Set Hold        │
│ • Update ledger│  │ • TTL 300s      │
│ • Emit event   │  │ • P3.3 blocks   │
│ • Clear hold   │  │ • Emit event    │
└────────────────┘  └─────────────────┘
        │                 │
        ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│ P3.3 Position Brain                                          │
│ • Check quantum:reconcile:hold:BTCUSDT                      │
│ • If hold exists → DENY with "reconcile_hold_active"       │
│ • If no hold → Continue normal permit logic                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Auto-Repair Policy

### When Auto-Repair is ALLOWED

P3.4 will automatically repair the ledger when **ALL** conditions are met:

1. ✅ **Fresh Exchange Snapshot**
   - `age_sec = now - snapshot.ts_epoch`
   - `age_sec <= 10s`
   - Stale data → HOLD

2. ✅ **No Side Mismatch**
   - `exchange.side == ledger.side`
   - Side mismatch (LONG vs SHORT) → HOLD

3. ✅ **Drift is Small & Plausible**
   - `diff_amt = abs(exchange_amt - ledger_amt)`
   - `diff_pct = (diff_amt / max(exchange_amt, ledger_amt)) * 100`
   - `diff_pct <= 1.0%` (configurable)
   - Large unexplained drift → HOLD

4. ✅ **Evidence Exists**
   - Recent executed order in `quantum:stream:apply.result`
   - `order.filled_qty` matches observed drift
   - `order.age <= 300s` (5 minutes)
   - No evidence → HOLD

### Auto-Repair Procedure

When all conditions pass:

1. **Update Ledger**
   ```python
   redis.hset("quantum:position:ledger:BTCUSDT", {
       "ledger_amt": exchange.position_amt,  # Align to exchange
       "ledger_side": exchange.side,
       "ts_epoch": now,
       "source": "p34_auto",
       "version": ledger.version + 1
   })
   ```

2. **Update State**
   ```python
   redis.hset("quantum:reconcile:state:BTCUSDT", {
       "status": "REPAIRED",
       "reason": "auto_fixed",
       "diff_amt": 0,
       "last_seen_ts": now
   })
   ```

3. **Clear Hold**
   ```python
   redis.delete("quantum:reconcile:hold:BTCUSDT")
   ```

4. **Emit Event**
   ```python
   redis.xadd("quantum:stream:reconcile.events", {
       "event": "AUTO_REPAIR",
       "symbol": "BTCUSDT",
       "old_ledger_amt": old_amt,
       "new_ledger_amt": new_amt,
       "evidence_plan_id": plan_id,
       "evidence_order_id": order_id
   })
   ```

5. **Update Metrics**
   ```python
   p34_auto_repair.labels(symbol="BTCUSDT").inc()
   p34_hold_active.labels(symbol="BTCUSDT").set(0)
   ```

---

## Hold Logic

### When HOLD is Set

P3.4 sets a hold (blocks P3.3 execution) when:

1. **Side Mismatch**
   - Exchange: LONG, Ledger: SHORT (or vice versa)
   - Reason: `side_mismatch`
   - Action: Manual intervention required

2. **Stale Exchange Data**
   - Snapshot age > 10s
   - Reason: `stale_exchange`
   - Action: Wait for fresh snapshot or investigate P3.3

3. **Qty Mismatch Without Evidence**
   - Drift detected but no matching executed order
   - Reason: `qty_mismatch_no_evidence`
   - Action: Manual review or wait for evidence

4. **Missing Exchange Snapshot**
   - `quantum:position:snapshot:BTCUSDT` doesn't exist
   - Reason: `missing_exchange`
   - Action: Check P3.3 is running

### Hold Key Structure

```
Key: quantum:reconcile:hold:BTCUSDT
Value: "1"
TTL: 300s (5 minutes)
```

- **Fail-closed:** If P3.4 crashes, holds expire after 5 min
- **P3.3 integration:** P3.3 checks this key before issuing ALLOW
- **Auto-clear:** P3.4 clears hold when reconciliation succeeds

### Manual Hold Management

**Check hold status:**
```bash
redis-cli GET quantum:reconcile:hold:BTCUSDT
# Output: "1" if active, (nil) if inactive
```

**Manually set hold:**
```bash
redis-cli SETEX quantum:reconcile:hold:BTCUSDT 300 1
```

**Manually clear hold:**
```bash
redis-cli DEL quantum:reconcile:hold:BTCUSDT
```

---

## Redis Keys

### Input Keys (Read by P3.4)

| Key | Type | Source | Purpose |
|-----|------|--------|---------|
| `quantum:position:snapshot:<symbol>` | Hash | P3.3 | Exchange position truth |
| `quantum:position:ledger:<symbol>` | Hash | Apply Layer | Internal ledger state |
| `quantum:stream:apply.result` | Stream | Apply Layer | Execution evidence |

**Exchange Snapshot Fields:**
```
position_amt: 0.062        # Exchange position
side: LONG                 # LONG or SHORT
entry_price: 95000.0       # Entry price
mark_price: 96000.0        # Current mark price
ts_epoch: 1735680000       # Snapshot timestamp
```

**Ledger Fields:**
```
ledger_amt: 0.062          # Internal ledger position
ledger_side: LONG          # LONG or SHORT
ts_epoch: 1735680000       # Last update timestamp
source: apply              # apply, p34_auto, p34_manual, init
version: 42                # Increment on each update
```

### Output Keys (Written by P3.4)

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `quantum:reconcile:state:<symbol>` | Hash | ∞ | Current reconciliation status |
| `quantum:reconcile:hold:<symbol>` | String | 300s | Hold flag (blocks P3.3) |
| `quantum:stream:reconcile.events` | Stream | ∞ | Audit trail (maxlen=10000) |

**State Fields:**
```
status: OK|DRIFT|HOLD|REPAIRED|MANUAL_REQUIRED
reason: auto_fixed|side_mismatch|stale_exchange|qty_mismatch_no_evidence|missing_exchange
exchange_amt: 0.062
ledger_amt: 0.062
diff_amt: 0.0
diff_pct: 0.0
last_seen_ts: 1735680000
```

**Event Types:**
- `LEDGER_INIT` - Ledger initialized from exchange
- `AUTO_REPAIR` - Drift auto-repaired
- `HOLD_SET` - Hold activated
- `HOLD_CLEARED` - Hold cleared
- `MANUAL_REQUIRED` - Manual intervention needed

---

## Metrics

**Prometheus Port:** 8046

### Available Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `p34_reconcile_drift_total` | Counter | symbol, reason | Total drifts detected |
| `p34_reconcile_auto_repair_total` | Counter | symbol | Total auto-repairs |
| `p34_reconcile_hold_active` | Gauge | symbol | Hold active (1=yes, 0=no) |
| `p34_reconcile_diff_amt` | Gauge | symbol | Current position diff amount |
| `p34_reconcile_last_fix_age_sec` | Gauge | symbol | Seconds since last auto-fix |

### Example Queries

**Total drifts by reason:**
```promql
sum by (reason) (p34_reconcile_drift_total)
```

**Auto-repair rate (last 5m):**
```promql
rate(p34_reconcile_auto_repair_total[5m])
```

**Symbols with active holds:**
```promql
p34_reconcile_hold_active == 1
```

**Current position drift:**
```promql
p34_reconcile_diff_amt{symbol="BTCUSDT"}
```

---

## Configuration

**File:** `/etc/quantum/reconcile-engine.env`

```bash
# Symbol allowlist (comma-separated)
P34_ALLOWLIST=BTCUSDT

# Loop cadence (seconds)
P34_LOOP_CADENCE_SEC=1.0

# Exchange snapshot freshness limit (seconds)
P34_EXCHANGE_FRESHNESS_SEC=10

# Qty drift tolerance percentage (1.0 = 1%)
P34_QTY_DRIFT_TOLERANCE_PCT=1.0

# Evidence window for apply.result lookback (seconds)
P34_EVIDENCE_WINDOW_SEC=300

# Hold TTL when reconciliation required (seconds)
P34_HOLD_TTL_SEC=300

# Prometheus metrics port
P34_PROMETHEUS_PORT=8046

# Logging level
P34_LOG_LEVEL=INFO
```

### Tuning Guide

**Lower Tolerance (More Conservative):**
```bash
P34_QTY_DRIFT_TOLERANCE_PCT=0.5  # 0.5% instead of 1%
P34_EXCHANGE_FRESHNESS_SEC=5     # 5s instead of 10s
```
Effect: More holds, fewer auto-repairs, safer

**Higher Tolerance (More Aggressive):**
```bash
P34_QTY_DRIFT_TOLERANCE_PCT=2.0  # 2% instead of 1%
P34_EVIDENCE_WINDOW_SEC=600      # 10 minutes instead of 5
```
Effect: More auto-repairs, fewer holds, riskier

**Recommended Production Settings:**
- Tolerance: 1.0% (default)
- Freshness: 10s (default)
- Evidence window: 300s (default)
- Hold TTL: 300s (default)

---

## Deployment

### Quick Deploy (VPS)

```bash
# On VPS, as root or quantum user
cd /opt/quantum
bash ops/p34_deploy_and_proof.sh
```

This script will:
1. Pull latest code
2. Copy config files to /etc/quantum/
3. Install systemd service
4. Restart P3.4 and P3.3
5. Run proof tests

### Manual Deploy

```bash
# 1. Pull code
cd /opt/quantum
git pull origin main

# 2. Copy config
cp deployment/config/reconcile-engine.env /etc/quantum/

# 3. Install systemd service
cp deployment/systemd/quantum-reconcile-engine.service /etc/systemd/system/
systemctl daemon-reload

# 4. Enable and start
systemctl enable quantum-reconcile-engine
systemctl start quantum-reconcile-engine

# 5. Restart P3.3 (for hold check integration)
systemctl restart quantum-position-state-brain

# 6. Verify
systemctl status quantum-reconcile-engine
curl http://localhost:8046/metrics | grep p34_
```

### Systemd Service

**Service Name:** `quantum-reconcile-engine`

**Commands:**
```bash
# Start
systemctl start quantum-reconcile-engine

# Stop
systemctl stop quantum-reconcile-engine

# Restart
systemctl restart quantum-reconcile-engine

# Status
systemctl status quantum-reconcile-engine

# Logs
journalctl -u quantum-reconcile-engine -f
```

---

## Monitoring

### Health Checks

**Service status:**
```bash
systemctl is-active quantum-reconcile-engine
# Output: active
```

**Metrics endpoint:**
```bash
curl http://localhost:8046/metrics | head -10
# Should show p34_* metrics
```

**Recent logs:**
```bash
journalctl -u quantum-reconcile-engine -n 20
```

### Key Indicators

**1. Active Holds**
```bash
redis-cli KEYS "quantum:reconcile:hold:*"
# Empty = no holds (normal)
# Keys = holds active (check reason)
```

**2. Reconcile State**
```bash
redis-cli HGETALL quantum:reconcile:state:BTCUSDT
```
Expected output (healthy):
```
status: OK
reason: (empty)
diff_amt: 0.0
last_seen_ts: 1735680000
```

**3. Event Stream**
```bash
redis-cli XLEN quantum:stream:reconcile.events
# Shows total events processed
```

**4. Recent Events**
```bash
redis-cli XREVRANGE quantum:stream:reconcile.events + - COUNT 5
```

### Alerts (Prometheus)

**Alert on hold active > 5 minutes:**
```yaml
- alert: P34HoldActive
  expr: p34_reconcile_hold_active == 1
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "P3.4 hold active for {{ $labels.symbol }}"
```

**Alert on repeated drift:**
```yaml
- alert: P34RepeatedDrift
  expr: rate(p34_reconcile_drift_total[5m]) > 0.1
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Repeated position drift detected"
```

---

## Troubleshooting

### Issue 1: Service Won't Start

**Symptoms:**
```bash
systemctl status quantum-reconcile-engine
# Output: failed (code=exited, status=1)
```

**Debug:**
```bash
# Check logs
journalctl -u quantum-reconcile-engine -n 50

# Common causes:
# - Redis not running
# - Config file missing
# - Python dependencies missing
# - Permission issues
```

**Fix:**
```bash
# Check Redis
systemctl status redis

# Check config exists
ls -l /etc/quantum/reconcile-engine.env

# Check Python dependencies
/opt/quantum/venv/bin/pip list | grep -i "redis\|prometheus"

# Check permissions
chown -R quantum:quantum /opt/quantum
```

### Issue 2: Holds Not Clearing

**Symptoms:**
```bash
redis-cli GET quantum:reconcile:hold:BTCUSDT
# Output: "1" (persists)
```

**Debug:**
```bash
# Check P3.4 logs for repair attempts
journalctl -u quantum-reconcile-engine | grep -i "BTCUSDT.*repair"

# Check reconcile state
redis-cli HGETALL quantum:reconcile:state:BTCUSDT
# Look at status and reason fields
```

**Reasons:**
1. **Side mismatch** - Manual fix required
2. **No evidence** - Wait for new order or manual fix
3. **Stale exchange data** - Check P3.3

**Manual Fix:**
```bash
# If you know the correct position, update ledger manually:
redis-cli HSET quantum:position:ledger:BTCUSDT ledger_amt <correct_amt>

# Hold will clear on next reconciliation loop (< 1s)
```

### Issue 3: Metrics Not Responding

**Symptoms:**
```bash
curl http://localhost:8046/metrics
# Connection refused
```

**Debug:**
```bash
# Check if P3.4 is running
systemctl status quantum-reconcile-engine

# Check if prometheus_client is installed
/opt/quantum/venv/bin/pip list | grep prometheus

# Check logs for metrics server errors
journalctl -u quantum-reconcile-engine | grep -i prometheus
```

**Fix:**
```bash
# Install prometheus_client if missing
/opt/quantum/venv/bin/pip install prometheus-client

# Restart service
systemctl restart quantum-reconcile-engine
```

### Issue 4: P3.3 Ignoring Holds

**Symptoms:**
- P3.4 sets hold
- P3.3 still issues ALLOW permits

**Debug:**
```bash
# Check if P3.3 has hold check code
grep -A 5 "quantum:reconcile:hold" /opt/quantum/microservices/position_state_brain/main.py
# Should show hold check in _grant_permit() function

# Check P3.3 restart after P3.4 deployment
systemctl status quantum-position-state-brain
```

**Fix:**
```bash
# Restart P3.3 to load updated code
systemctl restart quantum-position-state-brain

# Verify integration
journalctl -u quantum-position-state-brain | grep -i "reconcile_hold"
```

---

## Safety Considerations

### Fail-Closed Design

1. **Missing Data → HOLD**
   - No exchange snapshot → HOLD
   - Stale snapshot → HOLD
   - No ledger → Initialize from exchange

2. **Uncertain Drift → HOLD**
   - No evidence in apply.result → HOLD
   - Large unexplained drift → HOLD

3. **Dangerous Mismatch → HOLD**
   - Side mismatch → HOLD (NEVER auto-repair)

4. **Hold Expiry**
   - TTL 300s (5 minutes)
   - If P3.4 crashes, holds expire automatically
   - Fail-closed: No hold = assume safe (if P3.3 passes sanity checks)

### What P3.4 NEVER Does

- ❌ Never places orders on exchange
- ❌ Never modifies exchange position
- ❌ Never auto-repairs side mismatch
- ❌ Never ignores stale data
- ❌ Never repairs without evidence

### What P3.4 ONLY Does

- ✅ Reads exchange snapshot (from P3.3)
- ✅ Reads internal ledger
- ✅ Compares and detects drift
- ✅ Updates ledger when safe (with evidence)
- ✅ Sets holds when unsafe
- ✅ Emits audit trail

### Manual Override

If you need to force execution despite hold:

**Option 1: Clear hold manually**
```bash
redis-cli DEL quantum:reconcile:hold:BTCUSDT
```

**Option 2: Fix root cause**
```bash
# Side mismatch: Fix ledger manually
redis-cli HSET quantum:position:ledger:BTCUSDT ledger_side LONG

# Qty mismatch: Fix ledger manually
redis-cli HSET quantum:position:ledger:BTCUSDT ledger_amt 0.062

# P3.4 will detect fix and clear hold automatically
```

---

## Summary

**P3.4 Position Reconciliation Engine** is a production-grade, fail-closed system that:

- ✅ Continuously monitors position drift (1s cadence)
- ✅ Auto-repairs when safe (evidence-based)
- ✅ Blocks execution when unsafe (hold gating)
- ✅ Integrates with P3.3 (permit denial)
- ✅ Provides audit trail (event stream)
- ✅ Exposes Prometheus metrics (port 8046)
- ✅ Never places orders (only repairs ledger)

**Deploy:** `bash ops/p34_deploy_and_proof.sh`  
**Monitor:** `journalctl -u quantum-reconcile-engine -f`  
**Metrics:** `curl http://localhost:8046/metrics`  
**Status:** `redis-cli HGETALL quantum:reconcile:state:BTCUSDT`

---

**Version:** 1.0.0  
**Last Updated:** 2025-01-31  
**Maintainer:** Quantum Trading Team
