# P3.2 Governor - Fund-Grade Limits + Auto-Disarm

**Version:** 2.0.0 (Hardened)  
**Date:** 2026-01-24  
**Status:** Production Ready (Hardened with Real Binance Data)

---

## Executive Summary

P3.2 Governor is a safety-critical microservice that enforces fund-grade rate limits using **real-time Binance testnet position and price data**. It can automatically disarm the Apply Layer under unsafe conditions. It acts as the final gate before trade execution, issuing **single-use execution permits** (60s TTL) via Redis that the Apply Layer checks before placing real orders.

**P3.2 Hardening (v2.0):**
- ✅ Fetches REAL `positionAmt` from Binance `/fapi/v2/positionRisk`
- ✅ Fetches REAL `markPrice` from Binance `/fapi/v1/premiumIndex`
- ✅ Computes `close_qty` based on position and action type
- ✅ Enforces limits using computed notional (`qty × markPrice`)
- ✅ Single-use permits with 60s TTL (down from 3600s)
- ✅ Apply Layer fail-closed enforcement (blocks if permit missing or Redis error)

---

## Architecture

```
Harvest Proposals → Apply Layer → Governor → Execution Permits
                                     ↓
                            Auto-Disarm (if unsafe)
```

### Integration Pattern

1. **Apply Layer** publishes plans to `quantum:stream:apply.plan`
2. **Governor** reads plans, evaluates against limits
3. **Governor** writes permit keys: `quantum:permit:<plan_id>`
4. **Apply Layer** checks permit before executing real trades
5. **Governor** can force Apply Layer to `dry_run` mode if limits breached

---

## Real Limit Accounting (P3.2 Hardening)

**Problem:** Original implementation relied on placeholder values (`plan.close_qty=0.0`, `plan.price=None`) that made daily limits meaningless.

**Solution:** Governor now fetches real data from Binance testnet:

### 1. Position Data
```python
# GET /fapi/v2/positionRisk?symbol=BTCUSDT
position = binance_client.get_position(symbol)
pos_amt = abs(position['positionAmt'])  # Current position size
```

### 2. Price Data
```python
# GET /fapi/v1/premiumIndex?symbol=BTCUSDT
mark_price = binance_client.get_mark_price(symbol)
```

### 3. Computed Quantities
```python
# Based on action type
if action == 'FULL_CLOSE_PROPOSED':
    computed_qty = pos_amt  # 100% of position
elif action == 'PARTIAL_75':
    computed_qty = pos_amt * 0.75  # 75% of position
elif action == 'PARTIAL_50':
    computed_qty = pos_amt * 0.50  # 50% of position
```

### 4. Computed Notional
```python
computed_notional = computed_qty * mark_price
```

### 5. Limit Enforcement
- Daily notional limit checked against `computed_notional`
- Daily qty limit used as fallback if Binance unavailable
- Permits embed `computed_qty` and `computed_notional` for audit

### Binance Credentials
```bash
# /etc/quantum/governor.env
BINANCE_TESTNET_API_KEY=<your_key>
BINANCE_TESTNET_API_SECRET=<your_secret>
```

**Fallback:** If credentials missing, Governor uses qty-only limits (no notional enforcement).

---

## Safety Gates

### 1. Hourly Rate Limit (per symbol)
- **Default:** 3 executions/hour
- **Config:** `GOV_MAX_EXEC_PER_HOUR=3`
- **Behavior:** Blocks 4th+ execution in rolling 1-hour window

### 2. Burst Protection (per symbol)
- **Default:** 2 executions/5min
- **Config:** `GOV_MAX_EXEC_PER_5MIN=2`
- **Behavior:** Blocks 3rd+ execution in rolling 5-minute window
- **Auto-Disarm:** Can trigger system-wide disarm if breached

### 3. Daily Notional Limit (per symbol)
- **Default:** $5,000 USD notional reduced per day
- **Config:** `GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD=5000`
- **Data Source:** Real-time `markPrice` from Binance `/fapi/v1/premiumIndex`
- **Calculation:** `computed_notional = computed_qty × markPrice`
- **Fallback:** If Binance unavailable, uses qty limit
- **Behavior:** Blocks execution if daily total would exceed limit

### 4. Daily Quantity Limit (per symbol)
- **Default:** 0.02 BTC reduced per day
- **Config:** `GOV_MAX_REDUCE_QTY_PER_DAY=0.02`
- **Behavior:** Fallback when price data unavailable

### 5. Kill Score Critical Gate
- **Default:** 0.8 (80%)
- **Config:** `GOV_KILL_SCORE_CRITICAL=0.8`
- **Behavior:** If kill_score ≥ 0.8, only CLOSE actions allowed (blocks all others)

---

## Auto-Disarm

### Trigger Conditions

1. **Burst Breach:** If burst limit exceeded and `GOV_DISARM_ON_BURST_BREACH=true`
2. **Repeated Errors:** If error count ≥ `GOV_DISARM_ON_ERROR_COUNT` (default: 5)

### Disarm Action

When triggered:
1. Backs up `/etc/quantum/apply-layer.env` to `.bak.<timestamp>`
2. Sets `APPLY_MODE=dry_run` in config
3. Restarts `quantum-apply-layer.service`
4. Emits event to `quantum:stream:governor.events`
5. Increments `quantum_govern_disarm_total` metric

### Idempotency

- Uses Redis key: `quantum:governor:disarm:<date>` with 24h TTL
- Only one disarm per day (prevents repeated restarts)

---

## Configuration

### File: `/etc/quantum/governor.env`

```bash
# Redis connection
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Rate limits (per symbol)
GOV_MAX_EXEC_PER_HOUR=3
GOV_MAX_EXEC_PER_5MIN=2

# Daily limits (per symbol)
GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD=5000
GOV_MAX_REDUCE_QTY_PER_DAY=0.02

# Kill score gate
GOV_KILL_SCORE_CRITICAL=0.8

# Auto-disarm
GOV_ENABLE_AUTO_DISARM=true
GOV_DISARM_ON_ERROR_COUNT=5
GOV_DISARM_ON_BURST_BREACH=true

# Apply Layer config path (for disarm action)
APPLY_CONFIG_PATH=/etc/quantum/apply-layer.env

# Prometheus metrics
METRICS_PORT=8044
```

---

## Redis Keys

### Permit Keys
- **Pattern:** `quantum:permit:<plan_id>`
- **TTL:** 60 seconds (single-use, race-safe)
- **Format:** JSON:
  ```json
  {
    "granted": true,
    "symbol": "BTCUSDT",
    "decision": "EXECUTE",
    "computed_qty": 0.015,
    "computed_notional": 1425.50,
    "created_at": 1737753600,
    "consumed": false
  }
  ```
- **Consumption:** Apply Layer DELETEs permit after validation (atomic)
- **Purpose:** Short TTL prevents stale permits, `computed_*` fields provide audit trail

### Block Records
- **Pattern:** `quantum:governor:block:<plan_id>`
- **TTL:** 1 hour
- **Format:** JSON: `{"reason": "<reason>", "timestamp": <unix>, "symbol": "<symbol>"}`

### Execution Tracking
- **Pattern:** `quantum:governor:exec:<symbol>`
- **Type:** Redis List (timestamps)
- **TTL:** 24 hours
- **Max Length:** 100 entries

### Daily Limits
- **Pattern:** `quantum:governor:daily:<symbol>:<date>`
- **Type:** String (float value)
- **TTL:** 24 hours
- **Value:** Cumulative notional or qty reduced today

### Disarm Lock
- **Pattern:** `quantum:governor:disarm:<date>`
- **TTL:** 24 hours
- **Format:** JSON: `{"reason": "<reason>", "context": {...}, "timestamp": <unix>}`

---

## Prometheus Metrics

**Endpoint:** `http://localhost:8044/metrics`

### Counters
- `quantum_govern_allow_total{symbol}` - Plans allowed
- `quantum_govern_block_total{symbol,reason}` - Plans blocked
- `quantum_govern_disarm_total{reason}` - Auto-disarm events

### Gauges
- `quantum_govern_exec_count_hour{symbol}` - Executions in last hour
- `quantum_govern_exec_count_5min{symbol}` - Executions in last 5 minutes

---

## Event Stream

**Stream:** `quantum:stream:governor.events`

### Event Format

```
event: AUTO_DISARM
reason: burst_limit_breach
context: {"symbol": "BTCUSDT", "exec_5min": 3, "limit": 2}
timestamp: 1769207000
action_taken: APPLY_MODE=dry_run + restart
```

---

## Deployment

### Prerequisites
- Redis running
- Python 3.8+
- `redis-py` and `prometheus-client` installed
- Apply Layer deployed (P3.0/P3.1)

### Deploy Script (P3.2 Hardened)

```bash
# Single idempotent deployment command
cd /root/quantum_trader && git pull && bash ops/p32_vps_deploy_and_proof.sh
```

**Steps:**
1. Pulls latest code from GitHub
2. Syncs code to `/home/qt/quantum_trader`
3. Installs config to `/etc/quantum/governor.env` (with Binance credentials)
4. Installs systemd unit (`quantum-governor.service`)
5. Starts Governor service
6. Restarts Apply Layer (to pick up fail-closed enforcement)
7. Runs comprehensive proof tests
8. Saves proof output to `docs/P3_2_VPS_PROOF.txt`

**Config Auto-Install:**
- Copies `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET` from `/etc/quantum/testnet.env`
- Falls back to qty-only limits if credentials not found

### Verify

```bash
bash ops/p32_proof_governor.sh
```

**Checks:**
- Service status
- Metrics endpoint (port 8044)
- Permit keys (allow behavior)
- Block records
- Execution tracking
- Event stream
- Config file
- Recent logs

---

## Integration with Apply Layer

### P3.2 Hardened Integration (Fail-Closed Enforcement)

**In `execute_testnet()` (after position check):**

```python
# CHECK GOVERNOR PERMIT (P3.2) - FAIL-CLOSED
permit_key = f"quantum:permit:{plan.plan_id}"

# Fail-closed: Redis error blocks execution
try:
    permit_data = self.redis.get(permit_key)
except Exception as e:
    logger.error(f"{symbol}: Redis error checking permit: {e}")
    return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")

# Fail-closed: Missing permit blocks execution
if not permit_data:
    logger.warning(f"{symbol}: No execution permit from Governor (blocked)")
    return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")

# Parse and validate permit
try:
    permit = json.loads(permit_data)
    
    if not permit.get('granted'):
        logger.warning(f"{symbol}: Governor permit denied")
        return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")
    
    # Check if already consumed (race protection)
    if permit.get('consumed'):
        logger.warning(f"{symbol}: Permit already consumed (race detected)")
        return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")
    
    # CONSUME PERMIT ATOMICALLY (single-use semantics)
    self.redis.delete(permit_key)
    logger.info(f"{symbol}: Permit consumed ✓ (qty={permit['computed_qty']:.4f}, notional=${permit['computed_notional']:.2f})")
    
except json.JSONDecodeError as e:
    logger.error(f"{symbol}: Invalid permit format: {e}")
    return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")
except Exception as e:
    logger.error(f"{symbol}: Failed to consume permit: {e}")
    return ApplyResult(..., decision="BLOCKED", error="missing_permit_or_redis")

# ... proceed with execution
```

**Key Changes:**
- Decision changed from `"SKIP"` to `"BLOCKED"` for missing permits
- Single error code: `"missing_permit_or_redis"` (fail-closed)
- Atomic permit consumption (DELETE before execution)
- Redis errors block execution (no fallback)
- Logs computed qty/notional from permit

**In `execute_dry_run()` (logged only):**

```python
# CHECK GOVERNOR PERMIT (P3.2) - LOG ONLY in dry_run
permit_key = f"quantum:permit:{plan.plan_id}"
permit_data = self.redis.get(permit_key)

if not permit_data:
    logger.info(f"{symbol}: [DRY_RUN] No Governor permit (would block in testnet)")
else:
    logger.info(f"{symbol}: [DRY_RUN] Governor permit granted ✓")
```

---

## Operational Guide

### Monitor Governor

```bash
# Service status
systemctl status quantum-governor

# Real-time logs
journalctl -u quantum-governor -f

# Metrics
curl http://localhost:8044/metrics | grep quantum_govern
```

### Check Limits

```bash
# Hourly execution count for BTCUSDT
curl -s http://localhost:8044/metrics | grep "quantum_govern_exec_count_hour.*BTCUSDT"

# Block reasons
curl -s http://localhost:8044/metrics | grep "quantum_govern_block_total"
```

### Manual Disarm Test

```bash
# Simulate burst breach by sending many plans quickly
# (Governor will auto-disarm if GOV_DISARM_ON_BURST_BREACH=true)

# Check if disarmed
redis-cli GET "quantum:governor:disarm:$(date +%Y-%m-%d)"

# Check event stream
redis-cli XREVRANGE quantum:stream:governor.events + - COUNT 5
```

### Re-Enable After Disarm

```bash
# Review disarm reason
redis-cli GET "quantum:governor:disarm:$(date +%Y-%m-%d)" | jq

# If safe to proceed, manually set testnet mode
sudo sed -i 's/^APPLY_MODE=.*/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer
```

---

## Tuning Recommendations

### Conservative (Default)
```bash
GOV_MAX_EXEC_PER_HOUR=3
GOV_MAX_EXEC_PER_5MIN=2
GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD=5000
```

### Moderate
```bash
GOV_MAX_EXEC_PER_HOUR=5
GOV_MAX_EXEC_PER_5MIN=3
GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD=10000
```

### Aggressive (High-Frequency)
```bash
GOV_MAX_EXEC_PER_HOUR=10
GOV_MAX_EXEC_PER_5MIN=5
GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD=20000
```

**⚠️ Warning:** Always test on testnet before mainnet!

---

## Troubleshooting

### Governor Not Issuing Permits

**Symptom:** Apply Layer logs show "No execution permit from Governor"

**Checks:**
1. `systemctl status quantum-governor` - Is service running?
2. `journalctl -u quantum-governor -n 50` - Any errors?
3. `redis-cli XLEN quantum:stream:apply.plan` - Are plans being published?
4. `redis-cli --scan --pattern "quantum:permit:*" | wc -l` - Any permits?

**Fix:** Restart Governor: `systemctl restart quantum-governor`

### All Plans Blocked

**Symptom:** High `quantum_govern_block_total` count

**Checks:**
1. `curl -s http://localhost:8044/metrics | grep block_total` - What reasons?
2. `redis-cli --scan --pattern "quantum:governor:block:*" | head -5 | xargs -I {} redis-cli GET {}` - Sample blocks

**Common Reasons:**
- `hourly_limit_exceeded` - Too many executions in last hour
- `burst_limit_exceeded` - Too many in last 5 minutes
- `daily_limit_exceeded` - Daily notional/qty cap reached
- `kill_score_critical_non_close` - High kill score, non-CLOSE action

**Fix:** Wait for limit window to pass, or adjust config and restart

### Auto-Disarm Not Working

**Symptom:** Breach occurred but Apply Layer still in testnet mode

**Checks:**
1. `grep GOV_ENABLE_AUTO_DISARM /etc/quantum/governor.env` - Enabled?
2. `redis-cli GET "quantum:governor:disarm:$(date +%Y-%m-%d)"` - Already disarmed today?
3. `journalctl -u quantum-governor | grep -i disarm` - Was disarm attempted?

**Fix:** Check logs for errors in disarm logic, ensure Governor has sudo permissions (runs as root)

---

## Security Considerations

1. **Governor Must Run as Root** (for systemd restart capability)
   - Or grant `qt` user sudo permissions for specific commands
2. **Redis Access Control** - Ensure only trusted services can write to `quantum:permit:*`
3. **Metrics Port** - Expose 8044 only to monitoring systems (not public)
4. **Config Permissions** - `/etc/quantum/governor.env` should be readable only by `qt` user

---

## Future Enhancements

- [ ] Multi-symbol aggregate limits (e.g., max $10k/day across all symbols)
- [ ] Dynamic limit adjustment based on market volatility
- [ ] Permit expiration (currently 1h, could be configurable)
- [ ] Email/SMS alerts on disarm events
- [ ] Web dashboard for real-time limit monitoring
- [ ] Whitelist for bypass (e.g., emergency manual trades)

---

**Document Version:** 1.0.0  
**Last Updated:** 2026-01-23  
**Maintainer:** Quantum Trading Team
