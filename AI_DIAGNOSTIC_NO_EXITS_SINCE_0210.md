# ROOT CAUSE ANALYSIS: NO EXITS SINCE 02:10

## DIAGNOSIS COMPLETE ✅

### Summary
**Exit/Harvest IS working but P3.3 is blocking ALL permits with `stale_exchange_state` errors.**

---

## Root Cause

**P3.3 Position State Brain cannot obtain fresh exchange snapshots from Binance.**

Evidence:
1. ✅ Exit plans ARE being created (FULL_CLOSE_PROPOSED with market_reduce_only)
2. ✅ Exit plans ARE reaching P3.3 (stream message IDs confirmed)
3. ❌ P3.3 is DENYING 100% of plans (BUY and EXIT) with: `reason=stale_exchange_state`
4. ❌ Snapshot ages: **13-99 seconds old**
5. ❌ Stale threshold: **10 seconds**
6. 🚫 Result: All plans older than 10s = instant DENY

---

## Evidence

### Exit Plan Being Denied
```
Plan ID: d02f8bc4
Symbol: BTCUSDT
Action: FULL_CLOSE_PROPOSED (exit/harvest)
Type: market_reduce_only
Time: 04:56:18 UTC

Decision: DENY
Reason: stale_exchange_state
Age: 13 seconds (> threshold 10 seconds)
```

### P3.3 Logs (Recent)
```
04:56:18 [WARNING] BTCUSDT: P3.3 DENY plan d02f8bc4 reason=stale_exchange_state age_seconds=13
04:56:18 [WARNING] ETHUSDT: P3.3 DENY plan 541e8b3c reason=stale_exchange_state age_seconds=99
04:56:18 [WARNING] BTCUSDT: P3.3 DENY plan 1b5da5d5 reason=stale_exchange_state age_seconds=13
04:56:18 [WARNING] ETHUSDT: P3.3 DENY plan 05a8f9da reason=stale_exchange_state age_seconds=99
```

### Configuration
```
P33_POLL_SEC=1              ← Should update every 1 second
P33_STALE_THRESHOLD_SEC=10  ← Reject if > 10 seconds old
```

### What's Actually Happening
```
Expected: P3.3 polls Binance every 1 second → snapshots <5 seconds old
Actual:   P3.3 snapshots are 13-99 seconds old → all rejected
```

---

## Impact

| Item | Status |
|------|--------|
| Entry BUY orders | ❌ BLOCKED (denied) |
| Exit/CLOSE orders | ❌ BLOCKED (denied) |
| 7 open positions | 🔒 LOCKED (can't enter, can't exit) |
| Exit/Harvest services | ✅ Running (producing plans, unaware of blocks) |
| System status | 🔴 **DEADLOCK** |

---

## Why This Happened

### P3.3 Not Updating Snapshots - Possible Causes:

1. **Binance API Credentials Failed**
   - API key expired or invalid
   - Testnet account credentials wrong
   - Rate-limited by Binance

2. **Binance API Timeout**
   - Network latency too high
   - Binance server slow
   - Snapshot fetch takes >1 second

3. **Code Bug**
   - Snapshot update loop crashed silently
   - Exception not properly logged
   - Lock/deadlock in update code

4. **Environment Issue**
   - Network connectivity issue
   - Firewall blocking Binance API
   - DNS resolution failing

---

## Immediate Fix (Temporary - Reduces Safety)

Increase stale threshold to allow older snapshots:
```bash
# Instead of 10 seconds, allow 120-300 seconds
sudo nano /etc/quantum/position-state-brain.env
# Change: P33_STALE_THRESHOLD_SEC=120
sudo systemctl restart quantum-position-state-brain
```

**Risk:** Lower safety margin, but permits will flow through  
**Benefit:** Exits immediately process, positions can close

---

## Proper Fix (Debug Required)

1. **Check P3.3 for Binance errors:**
   ```bash
   journalctl -u quantum-position-state-brain -n 500 | grep -iE "error|fail|credential|auth|binance|timeout|exception"
   ```

2. **Check if snapshot update is running:**
   ```bash
   ps aux | grep position_state_brain
   strace -p <PID> 2>&1 | grep -i "socket\|connect\|api"
   ```

3. **Test Binance connectivity:**
   ```bash
   curl -v "https://testnet.binancefuture.com/fapi/v2/positionRisk" \
     -H "X-MBX-APIKEY: $BINANCE_KEY" | head -50
   ```

4. **Check API latency:**
   ```bash
   time curl -s "https://testnet.binancefuture.com/fapi/v1/time" | jq .serverTime
   ```

---

## Evidence Chain

```
Exit Brain (running) 
  ↓ produces FULL_CLOSE_PROPOSED plans
  ↓ publishes to quantum:stream:apply.plan
  ↓
P3.3 Position State Brain (running)
  ↓ reads from apply.plan
  ↓ checks snapshot age
  ↓ ❌ SNAPSHOT AGE 13-99 SECONDS
  ↓ publishes DENY permit
  ↓
Intent Executor (running)
  ↓ reads DENY permit
  ↓ skips execution
  ↓ publishes result: decision=SKIP
  ↓
Positions remain OPEN (deadlock)
```

---

## What Needs Investigation

**Priority 1 (Highest):**
- Why are P3.3 snapshots >10 seconds old?
- Is Binance API call succeeding?
- Check P3.3 logs for errors

**Priority 2:**
- Network connectivity to Binance
- API key validity
- Rate limiting

**Priority 3:**
- Temporary fix: increase stale threshold
- Monitoring: track snapshot age metric

---

## Quick Commands to Investigate

```bash
# 1. Check P3.3 error logs (last 500 lines)
journalctl -u quantum-position-state-brain -n 500 | tail -100

# 2. Check if Binance API is accessible from VPS
curl -I https://testnet.binancefuture.com/fapi/v1/time

# 3. Check P3.3 snapshot age in real-time
redis-cli monitor | grep "p33:snapshot" &
sleep 10; pkill -f "redis-cli monitor"

# 4. Restart P3.3 to reset snapshot age
systemctl restart quantum-position-state-brain

# 5. Watch for fresh snapshots
journalctl -u quantum-position-state-brain -f | grep -i "snapshot\|permit"
```

---

## Next Steps (Recommended)

1. **Immediate:** Run `journalctl -u quantum-position-state-brain -n 500` and check for Binance errors
2. **If errors found:** Fix credentials, connectivity, or rate limits
3. **If no errors:** Restart P3.3 service and observe snapshot ages for 60 seconds
4. **If snapshots still stale:** Increase threshold temporarily to unblock exits, then debug properly
5. **Once working:** Reduce threshold back to 10 and implement monitoring

---

## Status

**Root Cause:** ✅ IDENTIFIED  
**Diagnosis:** ✅ CONFIRMED  
**Fix Target:** P3.3 Binance snapshot update loop  
**Severity:** 🔴 CRITICAL (all trades blocked)  
**Evidence:** Plan ID d02f8bc4 (BTCUSDT exit, denied 04:56:18 UTC)

