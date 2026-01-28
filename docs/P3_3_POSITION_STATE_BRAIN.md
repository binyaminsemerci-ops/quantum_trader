# P3.3 Position State Brain

**Status**: ✅ DEPLOYED  
**Port**: 8045 (metrics)  
**Purpose**: 3-way sanity check before reduceOnly execution  

---

## Executive Summary

P3.3 Position State Brain is a critical safety layer that validates position state **before** Apply Layer executes any reduceOnly order. It reconciles three sources of truth:

1. **Exchange Truth** (Binance testnet): Real position from `/fapi/v2/positionRisk`
2. **Redis Truth** (pipeline): Harvest/risk proposals in quantum streams
3. **Internal Ledger** (shadow): What we think position is from our executed orders

**Integration**: Apply Layer testnet mode requires **BOTH** permits:
- ✅ P3.2 Governor permit (limits OK)
- ✅ P3.3 Position permit (exchange state OK)

**Fail-Closed**: Missing either permit → execution BLOCKED

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   P3.3 Position State Brain                  │
│                                                               │
│  ┌─────────────┐     ┌──────────────┐    ┌───────────────┐  │
│  │  Exchange   │     │    Redis     │    │    Internal   │  │
│  │   Snapshot  │────▶│   Proposals  │◀───│    Ledger     │  │
│  │  (Binance)  │     │              │    │  (apply.res.) │  │
│  └─────────────┘     └──────────────┘    └───────────────┘  │
│         │                    │                     │          │
│         └────────────────────┼─────────────────────┘          │
│                              │                                │
│                      ┌───────▼──────┐                         │
│                      │  6-Point     │                         │
│                      │  Sanity      │                         │
│                      │  Check       │                         │
│                      └───────┬──────┘                         │
│                              │                                │
│               ┌──────────────┴──────────────┐                 │
│               │                             │                 │
│        ┌──────▼──────┐             ┌────────▼─────┐           │
│        │   ALLOW     │             │     DENY     │           │
│        │  + safe_qty │             │   + reason   │           │
│        └──────┬──────┘             └────────┬─────┘           │
│               │                             │                 │
│               └──────────────┬──────────────┘                 │
│                              │                                │
│                      ┌───────▼──────┐                         │
│                      │  P3.3 Permit │                         │
│                      │  (Redis key) │                         │
│                      └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
                    ┌─────────▼─────────┐
                    │   Apply Layer     │
                    │                   │
                    │  1. Check Gov     │
                    │  2. Check P3.3    │
                    │  3. Use safe_qty  │
                    │  4. Execute       │
                    └───────────────────┘
```

---

## Redis Key Schemas

### 1. Exchange Snapshots
**Key**: `quantum:position:snapshot:<symbol>`  
**Type**: HASH  
**TTL**: 3600s (1 hour)  
**Updated**: Every 5s by P3.3 service

**Fields**:
```
position_amt          # From Binance /fapi/v2/positionRisk
side                  # LONG | SHORT
entry_price           # Average entry price
mark_price            # Current mark price
leverage              # Current leverage
unrealized_pnl        # Current PnL
ts_epoch              # Snapshot timestamp (MUST be fresh)
ts_iso                # Human-readable timestamp
```

**Purpose**: Real-time exchange position state for sanity checks

---

### 2. Internal Ledger
**Key**: `quantum:position:ledger:<symbol>`  
**Type**: HASH  
**TTL**: 86400s (24 hours)  
**Updated**: When Apply Layer executes orders (executed=True)

**Fields**:
```
last_known_amt        # Our shadow position amount
last_known_side       # Our shadow position side
last_order_id         # Last executed order ID
last_order_time_epoch # When last order executed
last_update_ts        # When ledger last updated
```

**Purpose**: Shadow ledger to detect exchange/ledger mismatches

---

### 3. P3.3 Permits
**Key**: `quantum:permit:p33:<plan_id>`  
**Type**: STRING (JSON)  
**TTL**: 60s  
**Created**: When P3.3 evaluates EXECUTE plans

**Allow Permit**:
```json
{
  "plan_id": "abc123",
  "symbol": "BTCUSDT",
  "allow": true,
  "safe_close_qty": 0.025,
  "exchange_position_amt": 0.038,
  "ledger_position_amt": 0.038,
  "snapshot_age_sec": 3.2,
  "ts_epoch": 1737672000
}
```

**Deny Permit**:
```json
{
  "plan_id": "abc123",
  "symbol": "BTCUSDT",
  "allow": false,
  "reason": "stale_exchange_state",
  "details": {
    "snapshot_age_sec": 15.7,
    "threshold_sec": 10
  },
  "ts_epoch": 1737672000
}
```

**Purpose**: Single-use permit for Apply Layer execution

---

## 6 Sanity Rules

### Rule 1: Stale Snapshot
**Trigger**: Exchange snapshot age > 10s  
**Action**: DENY with `reason=stale_exchange_state`  
**Rationale**: Old data may not reflect current position

**Example**:
```
Exchange snapshot: 15s old
Result: DENY (stale_exchange_state)
```

---

### Rule 2: No Position
**Trigger**: `abs(exchange_position_amt) < 0.0001`  
**Action**: DENY with `reason=no_position`  
**Rationale**: Cannot close if no position exists

**Example**:
```
Exchange: position_amt = 0.0000
Plan: CLOSE_PARTIAL_75
Result: DENY (no_position)
```

---

### Rule 3: Side Mismatch
**Trigger**: Exchange side ≠ Ledger side  
**Action**: DENY with `reason=reconcile_required_side_mismatch`  
**Rationale**: Position flipped unexpectedly → reconcile needed

**Example**:
```
Exchange: LONG (+0.05)
Ledger: SHORT (-0.03)
Result: DENY (reconcile_required_side_mismatch)
```

---

### Rule 4: Qty Mismatch
**Trigger**: `abs(exchange_amt - ledger_amt) / exchange_amt > tolerance`  
**Action**: DENY with `reason=reconcile_required_qty_mismatch`  
**Rationale**: Position quantity differs > 1% → reconcile needed

**Example**:
```
Exchange: 0.050 BTC
Ledger: 0.045 BTC
Diff: 10% (> 1% tolerance)
Result: DENY (reconcile_required_qty_mismatch)
```

---

### Rule 5: Cooldown
**Trigger**: Time since last order < 15s  
**Action**: DENY with `reason=cooldown_in_flight`  
**Rationale**: Recent order may not be reflected in position yet

**Example**:
```
Last order: 8s ago
Cooldown: 15s
Result: DENY (cooldown_in_flight)
```

---

### Rule 6: Safe Close Qty
**Computation**:
```python
requested_qty = exchange_amt * CLOSE_FACTOR  # 0.5, 0.75, or 1.0
safe_close_qty = min(requested_qty, abs(exchange_amt))
safe_close_qty = round_to_step_size(safe_close_qty, step_size)
```

**Purpose**: Ensure we never try to close more than actual position

**Example**:
```
Exchange: 0.038 BTC
Plan: CLOSE_PARTIAL_75
Requested: 0.038 * 0.75 = 0.0285
Safe: min(0.0285, 0.038) = 0.0285
Rounded (step=0.001): 0.028
Result: ALLOW (safe_close_qty=0.028)
```

---

## Configuration

**File**: `/etc/quantum/position-state-brain.env`

```bash
# Service
P33_LOG_LEVEL=INFO

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Polling
P33_POLL_SEC=5                    # How often to update snapshots
P33_ALLOWLIST=BTCUSDT             # Symbols to track

# Thresholds
P33_STALE_THRESHOLD_SEC=10        # Max snapshot age
P33_COOLDOWN_SEC=15               # Min time between orders
P33_QTY_TOLERANCE_PCT=1.0         # Max % difference (ledger vs exchange)

# Permits
P33_PERMIT_TTL=60                 # Permit lifetime (seconds)

# Metrics
P33_METRICS_PORT=8045

# Binance (copied from testnet.env)
BINANCE_API_KEY=xxx
BINANCE_API_SECRET=xxx
BINANCE_TESTNET=true
```

---

## Metrics

**Endpoint**: `http://localhost:8045/metrics`

### Counters
- `p33_snapshot_total{symbol, status}` - Exchange snapshot updates
- `p33_ledger_update_total{symbol}` - Ledger updates from apply.result
- `p33_permit_total{symbol, decision}` - Permits issued (allow/deny)

### Gauges
- `p33_last_snapshot_ts{symbol}` - Last snapshot timestamp
- `p33_position_amt{symbol}` - Current exchange position
- `p33_ledger_amt{symbol}` - Current ledger position
- `p33_snapshot_age_sec{symbol}` - Current snapshot age

### Histograms
- `p33_evaluation_duration_seconds` - Sanity check evaluation time

---

## Integration with Apply Layer

### Testnet Mode (ENABLED)

Apply Layer `execute_testnet()` requires **BOTH** permits:

```python
# 1. CHECK P3.2 GOVERNOR PERMIT
gov_permit = redis.get(f"quantum:permit:governor:{plan_id}")
if not gov_permit or not gov_permit['granted']:
    return BLOCKED

redis.delete(gov_permit_key)  # Consume

# 2. CHECK P3.3 POSITION PERMIT
p33_permit = redis.get(f"quantum:permit:p33:{plan_id}")
if not p33_permit or not p33_permit['allow']:
    return BLOCKED

safe_close_qty = p33_permit['safe_close_qty']  # USE THIS
redis.delete(p33_permit_key)  # Consume

# 3. EXECUTE with safe_close_qty
order = place_market_order(
    symbol=symbol,
    side=order_side,
    quantity=safe_close_qty,
    reduce_only=True
)
```

**Fail-Closed Behavior**:
- Missing Governor permit → `error=missing_permit_or_redis`
- Missing P3.3 permit → `error=missing_or_denied_p33_permit`
- P3.3 denied → `error=missing_or_denied_p33_permit`

---

## Operational Guide

### Starting P3.3

```bash
# Start service
systemctl start quantum-position-state-brain

# Check status
systemctl status quantum-position-state-brain

# View logs
journalctl -u quantum-position-state-brain -f
```

### Monitoring

**Check metrics**:
```bash
curl -s http://localhost:8045/metrics | grep "^p33_"
```

**Check snapshots**:
```bash
redis-cli KEYS "quantum:position:snapshot:*"
redis-cli HGETALL "quantum:position:snapshot:BTCUSDT"
```

**Check ledger**:
```bash
redis-cli KEYS "quantum:position:ledger:*"
redis-cli HGETALL "quantum:position:ledger:BTCUSDT"
```

**Check permits**:
```bash
redis-cli KEYS "quantum:permit:p33:*"
redis-cli GET "quantum:permit:p33:<plan_id>"
```

### Troubleshooting

**Problem**: P3.3 service not starting

```bash
# Check logs
journalctl -u quantum-position-state-brain -n 50

# Common issues:
# - Missing config: /etc/quantum/position-state-brain.env
# - Missing Binance credentials
# - Redis not accessible
# - Port 8045 already in use
```

**Problem**: Snapshots not updating

```bash
# Check last snapshot
redis-cli HGET "quantum:position:snapshot:BTCUSDT" ts_epoch

# Compare to now
date +%s

# If age > 10s, check:
# - Binance API credentials valid
# - Network connectivity
# - Service logs for errors
```

**Problem**: All permits denied

```bash
# Check recent denials
journalctl -u quantum-position-state-brain --since "5 minutes ago" | grep "DENY"

# Common reasons:
# - stale_exchange_state: Snapshots not updating
# - no_position: No position on exchange
# - cooldown_in_flight: Recent order executed
# - reconcile_required: Exchange/ledger mismatch
```

**Problem**: Apply Layer not checking P3.3

```bash
# Check Apply Layer logs
journalctl -u quantum-apply-layer --since "5 minutes ago" | grep "P3.3"

# Should see:
# - "P3.3 permit consumed ✓"
# - "Using P3.3 safe_close_qty=..."

# If not:
# - Restart Apply Layer: systemctl restart quantum-apply-layer
# - Check code integration in execute_testnet()
```

---

## Deployment

**One-command deployment**:
```bash
bash /home/qt/quantum_trader/ops/p33_deploy_and_proof.sh
```

**Manual steps**:
```bash
# 1. Pull code
cd /root/quantum_trader
git pull origin main

# 2. Sync to working dir
rsync -av /root/quantum_trader/ /home/qt/quantum_trader/

# 3. Install config
cp /home/qt/quantum_trader/deployment/config/position-state-brain.env /etc/quantum/
# Add Binance credentials from testnet.env

# 4. Install service
cp /home/qt/quantum_trader/deployment/systemd/quantum-position-state-brain.service /etc/systemd/system/
systemctl daemon-reload

# 5. Start
systemctl enable --now quantum-position-state-brain

# 6. Restart Apply Layer
systemctl restart quantum-apply-layer

# 7. Verify
bash /home/qt/quantum_trader/ops/p33_proof.sh
```

---

## Testing

### Test 1: Normal Execution
**Setup**: Healthy position, fresh snapshot, no cooldown  
**Expected**: P3.3 permit issued with safe_close_qty  
**Verify**: Apply Layer executes with safe_close_qty

### Test 2: Stale Snapshot
**Setup**: Stop P3.3 service for 15s  
**Expected**: P3.3 denies with `stale_exchange_state`  
**Verify**: Apply Layer blocks with `missing_or_denied_p33_permit`

### Test 3: No Position
**Setup**: Close all positions on exchange  
**Expected**: P3.3 denies with `no_position`  
**Verify**: Apply Layer blocks

### Test 4: Cooldown
**Setup**: Execute order, immediately try another  
**Expected**: P3.3 denies with `cooldown_in_flight`  
**Verify**: Apply Layer blocks

### Test 5: Qty Mismatch
**Setup**: Manually modify ledger to differ > 1%  
**Expected**: P3.3 denies with `reconcile_required_qty_mismatch`  
**Verify**: Apply Layer blocks

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Stale exchange data** | Snapshots refreshed every 5s, denied if > 10s old |
| **Race conditions** | 15s cooldown after each order |
| **Qty computation errors** | P3.3 computes safe_close_qty (clamped + rounded) |
| **Exchange/ledger divergence** | 1% tolerance, reconcile if exceeded |
| **Missing permits** | Fail-closed: no permit → no execution |
| **Service failure** | Apply Layer blocks without P3.3 permit |

---

## Future Enhancements

1. **Reconciliation Service**: Auto-sync ledger when mismatches detected
2. **Multi-Symbol Support**: Parallel snapshot updates for multiple symbols
3. **Advanced Cooldowns**: Per-symbol, adaptive based on volatility
4. **Permit Caching**: Pre-issue permits for predicted plans
5. **ML Integration**: Predict position changes, pre-validate

---

## References

- P3.2 Governor: [P3_2_GOVERNOR.md](P3_2_GOVERNOR.md)
- Apply Layer: [P3_APPLY_LAYER.md](P3_APPLY_LAYER.md)
- Binance API: https://binance-docs.github.io/apidocs/futures/en/

---

**Last Updated**: 2026-01-23  
**Version**: 1.0.0  
**Status**: Production Ready
