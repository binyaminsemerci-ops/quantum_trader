# Chaos-Test Runbook: Emergency Exit System

## Overview

This runbook describes how to test the Emergency Exit System's fail-closed behavior by simulating Exit Brain failure. The test validates that:

1. Watchdog detects Exit Brain death within 3-5 seconds
2. `system:panic_close` event is published
3. Emergency Exit Worker closes all positions within 5-10 seconds

---

## Prerequisites

### Services Required
```bash
# Check services are running
systemctl status quantum-exit-brain.service
systemctl status quantum-exit-brain-watchdog.service  
systemctl status quantum-emergency-exit-worker.service
```

### Redis Access
```bash
# Verify Redis connectivity
redis-cli ping
```

### Test Environment
- **TESTNET ONLY** - Never run this on production with real positions
- Verify `BINANCE_TESTNET=true` in environment
- Create 1-2 small test positions before running

---

## Test Procedure

### Step 1: Baseline Health Check (T-60s)

```bash
# Verify heartbeat is flowing
redis-cli XREVRANGE exit_brain:heartbeat + - COUNT 5

# Should see recent entries with status=ALIVE:
# 1) 1) "1707840123456-0"
#    2) 1) "status" 2) "ALIVE" 3) "active_positions" 4) "2" 5) "ts" 6) "1707840123456"
```

```bash
# Check EEW is listening
journalctl -u quantum-emergency-exit-worker -n 5 --no-pager
# Should see: "Listening for panic_close events..."
```

### Step 2: Create Test Positions (T-30s)

```bash
# Via Binance testnet CLI or Python
# Create 1-2 small BTCUSDT positions (~$100 notional)
```

### Step 3: Kill Exit Brain (T=0)

```bash
# Method 1: Graceful kill (SIGTERM)
sudo systemctl stop quantum-exit-brain.service

# Method 2: Hard kill (SIGKILL) - more realistic chaos test
sudo kill -9 $(pgrep -f "main_with_watchdog.py")
```

**Start timer now.**

### Step 4: Monitor Watchdog Detection (T+0 to T+5s)

```bash
# Watch watchdog logs in real-time
journalctl -u quantum-exit-brain-watchdog -f
```

**Expected within 3-5 seconds:**
```
🚨 FAILURE DETECTED: Heartbeat missing (5.1s > 5.0s)
🚨 TRIGGERING PANIC CLOSE 🚨
Reason: EXIT_BRAIN_HEARTBEAT_MISSING_(5.1S_>_5.0S)
✅ Panic close published to system:panic_close
Event ID: 550e8400-e29b-41d4-a716-446655440000
```

### Step 5: Verify Panic Close Published (T+5s)

```bash
# Check panic_close stream
redis-cli XREVRANGE system:panic_close + - COUNT 1

# Should see:
# 1) 1) "1707840128456-0"
#    2) 1) "event_id"
#       2) "550e8400-e29b-41d4-a716-446655440000"
#       3) "reason"
#       4) "EXIT_BRAIN_HEARTBEAT_MISSING_(5.1S_>_5.0S)"
#       5) "severity"
#       6) "CRITICAL"
#       7) "issued_by"
#       8) "watchdog"
#       9) "ts"
#       10) "1707840128456"
```

### Step 6: Verify EEW Execution (T+5s to T+10s)

```bash
# Watch EEW logs
journalctl -u quantum-emergency-exit-worker -f
```

**Expected:**
```
============================================================
🚨 PANIC CLOSE EVENT RECEIVED 🚨
============================================================
Event ID: 550e8400-e29b-41d4-a716-446655440000
Source: watchdog
Reason: EXIT_BRAIN_HEARTBEAT_MISSING_(5.1S_>_5.0S)
✅ Trigger validated (source=watchdog, age=0.5s)
Found 2 open positions
Total notional: $207.50
Closing BTCUSDT: SELL 0.002 (reduce-only)
✅ BTCUSDT closed - Order ID: 12345678
Closing ETHUSDT: SELL 0.05 (reduce-only)
✅ ETHUSDT closed - Order ID: 12345679
Closed: 2/2
Execution time: 1847ms
Published to system:panic_close:completed
============================================================
🔒 PANIC CLOSE COMPLETE - SYSTEM HALTED 🔒
============================================================
```

### Step 7: Verify Completion Event (T+10s)

```bash
# Check completion stream
redis-cli XREVRANGE system:panic_close:completed + - COUNT 1

# Should show:
# 1) 1) "1707840130303-0"
#    2) 1) "event_id" 2) "550e8400..." 
#       3) "positions_total" 4) "2"
#       5) "positions_closed" 6) "2"
#       7) "positions_failed" 8) "0"
#       9) "failed_symbols" 10) "[]"
#       11) "ts_started" 12) "1707840128456"
#       13) "ts_completed" 14) "1707840130303"
#       15) "execution_time_ms" 16) "1847"
```

### Step 8: Verify Trading Halted

```bash
# Check halt state
redis-cli HGETALL system:state:trading

# Should show:
# halted: true
# reason: EXIT_BRAIN_HEARTBEAT_MISSING_...
# source: watchdog
# positions_closed: 2
# requires_manual_reset: true
```

---

## Pass/Fail Criteria

### PASS ✅

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Heartbeat last seen → panic_close published | < 5.5s | Watchdog threshold is 5s + 0.5s tolerance |
| panic_close → all positions closed | < 10s | EEW must be fast |
| Total: kill → positions closed | < 15s | End-to-end safety |
| positions_failed | 0 | All positions must close |
| system:state:trading.halted | true | Trading must be halted |

### FAIL ❌

- Watchdog does not detect death within 10 seconds
- panic_close not published
- EEW does not execute
- Positions remain open
- positions_failed > 0 (any position failed to close)
- system:state:trading.halted != true

---

## Recovery Procedure

After test completes:

```bash
# 1. Clear halt state (MANUAL ONLY)
redis-cli DEL system:state:trading

# 2. Restart Exit Brain
sudo systemctl start quantum-exit-brain.service

# 3. Verify heartbeat resumes
redis-cli XREVRANGE exit_brain:heartbeat + - COUNT 1

# 4. Verify watchdog re-attaches
journalctl -u quantum-exit-brain-watchdog -n 5 --no-pager
```

---

## Timeline Summary

```
T=0       Kill Exit Brain (SIGKILL)
T+1-2s    Last heartbeat ages out
T+3-5s    Watchdog detects (hb > 5s threshold)
T+3-5s    system:panic_close published
T+3-6s    EEW receives event
T+5-10s   All positions closed
T+5-10s   system:panic_close:completed published
T+5-10s   system:state:trading.halted = true
```

**Total safety window: < 15 seconds**

---

## Critical Notes

### DO NOT

- ❌ Run this on production with real money
- ❌ Disable the watchdog "for testing"
- ❌ Add retry logic to EEW
- ❌ Add grace periods to watchdog

### DO

- ✅ Run monthly chaos tests
- ✅ Alert on test failures
- ✅ Review logs after each test
- ✅ Keep this runbook updated

---

## Automated Script

```bash
#!/bin/bash
# chaos_test_exit_brain.sh
# Run with: sudo ./chaos_test_exit_brain.sh

set -e

echo "=== CHAOS TEST: Exit Brain Kill ==="
echo "WARNING: This will close ALL test positions"
read -p "Continue? (yes/no): " confirm
[ "$confirm" != "yes" ] && exit 1

# Pre-check
echo "Checking services..."
systemctl is-active quantum-exit-brain.service || { echo "Exit Brain not running"; exit 1; }
systemctl is-active quantum-exit-brain-watchdog.service || { echo "Watchdog not running"; exit 1; }
systemctl is-active quantum-emergency-exit-worker.service || { echo "EEW not running"; exit 1; }

# Record start time
START=$(date +%s%3N)

# Kill Exit Brain
echo "Killing Exit Brain at $(date)"
kill -9 $(pgrep -f "main_with_watchdog.py") 2>/dev/null || true

# Wait for completion
echo "Waiting for panic_close:completed..."
for i in {1..30}; do
    RESULT=$(redis-cli XLEN system:panic_close:completed 2>/dev/null || echo "0")
    if [ "$RESULT" != "0" ]; then
        END=$(date +%s%3N)
        ELAPSED=$((END - START))
        echo "✅ Panic close completed in ${ELAPSED}ms"
        exit 0
    fi
    sleep 1
done

echo "❌ FAIL: No completion event within 30 seconds"
exit 1
```

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-02-14 | 1.0 | AI Agent | Initial creation |
