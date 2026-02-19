# FASE A - SKIP_FLAT_SELL FIX - DEPLOYMENT REPORT
**Date**: 2026-02-18 04:06 UTC  
**Action**: Disable SKIP_FLAT_SELL filter in Intent Bridge  
**Status**: ✅ **ALREADY DEPLOYED & ACTIVE**

---

## 🎯 Deployment Summary

### Configuration Status

**File**: `/etc/quantum/intent-bridge.env`  
**Setting**: `INTENT_BRIDGE_SKIP_FLAT_SELL=false`

**Service**: `quantum-intent-bridge.service`  
**Status**: Active (running)  
**Started**: 2026-02-18 04:06:22 UTC (restarted for verification)  
**PID**: 637393

### Verification Results

✅ **Config Loaded**: Confirmed in service logs
```
Skip flat SELL: False
```

✅ **CLOSE Orders Flowing**: 134 CLOSE entries found in last 100 apply.plan messages
```bash
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -c 'CLOSE'
# Result: 134
```

✅ **No Filtering Active**: 0 SELL orders skipped in last hour
```bash
journalctl -u quantum-intent-bridge --since "1 hour ago" --no-pager | grep -c "Skip publish.*SELL"
# Result: 0
```

---

## 📊 What This Means

### Before Fix (SKIP_FLAT_SELL=true)
```
Exit Monitor detects TP/SL
  → Publishes SELL to trade.intent ✅
    → Intent Bridge checks ledger
      → Ledger missing/flat
        → ❌ SELL DROPPED (never reaches apply.plan)
          → Position stays open indefinitely
```

### After Fix (SKIP_FLAT_SELL=false)
```
Exit Monitor detects TP/SL
  → Publishes SELL to trade.intent ✅
    → Intent Bridge receives SELL
      → ✅ PUBLISHES to apply.plan (no ledger check)
        → Governor processes ✅
          → Apply Layer executes ✅
            → Binance receives close order ✅
              → Position closes at TP/SL
```

---

## 🔬 FASE A Test Plan - ACTIVE NOW

### Observation Period
**Duration**: Next 30-50 trades  
**Start**: 2026-02-18 04:06 UTC  
**End**: ~2026-02-19 (24-48 hours depending on trade frequency)

### Key Metrics to Monitor

#### 1. Exit Execution Rate
**Before**: Very low (exits filtered out)  
**Expected After**: Exits execute when TP/SL hit

**How to Check**:
```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Count CLOSE orders in last 4 hours
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 500 | grep -c "CLOSE"

# Check if they're being executed
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 100 | grep -c "SELL"
```

#### 2. Win Rate
**Before**: 6.1% (from expectancy audit)  
**Expected After**: 25-35%+ if exits work properly

**How to Check**:
```bash
# On local machine or VPS
# Use existing expectancy audit script
python audit_binance_testnet.py
```

#### 3. Position Drift
**Before**: Positions "drift" past TP without closing  
**Expected After**: Positions close at or near TP level

**How to Check**:
```bash
# Check Exit Monitor logs
journalctl -u quantum-exit-monitor -f | grep "EXIT TRIGGERED"

# Example expected output:
# 🎯 EXIT TRIGGERED: BTCUSDT LONG | Reason: TAKE_PROFIT | Entry=$71,234 | Exit=$71,890 | PnL=+0.92%
```

#### 4. reduceOnly Orders on Binance
**Before**: Very few or none  
**Expected After**: Each position should close with reduceOnly=true order

**How to Check**:
- Login to Binance Testnet: https://testnet.binancefuture.com
- Go to Orders → Order History
- Filter: Recent 24h
- Look for: "Reduce-Only" tag on SELL/BUY orders

#### 5. Filtered Orders (Should Be ZERO)
**Expected**: 0 SELL orders filtered

**How to Check**:
```bash
# Check for "Skip publish" events
journalctl -u quantum-intent-bridge --since "1 hour ago" --no-pager | grep "Skip publish.*SELL"

# Expected: No output (0 matches)
```

---

## 📈 Expected Improvements

If SKIP_FLAT_SELL was the primary break point, you should see:

### Immediate Effects (within 10-20 trades)
- ✅ CLOSE orders appear in `quantum:stream:apply.plan`
- ✅ Exit Monitor logs show "EXIT TRIGGERED" events
- ✅ Binance shows reduceOnly SELL/BUY orders
- ✅ Positions close within seconds/minutes of TP hit

### Medium-Term Effects (30-50 trades)
- 📈 Win Rate increases to 25-35%+
- 📈 Profit Factor improves (positive expectancy)
- 📉 Max drawdown per trade decreases
- 📉 Average hold time normalizes (no drift)

### Statistical Validation
Run expectancy audit after 30+ trades:
```python
python audit_binance_testnet.py
```

**Expected Results**:
- Win Rate: 25-35% (up from 6.1%)
- Expectancy: Positive (up from -$0.60)
- R-multiple distribution: More 1R, 2R, 3R wins
- Fewer catastrophic losses (drift prevented)

---

## 🚨 What If Nothing Changes?

If after 30-50 trades you see:
- Win rate still ~6-10%
- No CLOSE orders in logs
- Positions still drifting
- No reduceOnly orders on Binance

**Then**: SKIP_FLAT_SELL was NOT the only break point.

**Next Steps**:
1. Check if Exit Monitor is actually running:
   ```bash
   systemctl status quantum-exit-monitor
   ```

2. Check if Exit Monitor is detecting exit conditions:
   ```bash
   journalctl -u quantum-exit-monitor -n 100 | grep "check_exit"
   ```

3. Check if trade.intent stream is receiving EXIT messages:
   ```bash
   redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 100 | grep -A 5 "EXIT"
   ```

4. Check Alternative Exit Path (HarvestBrain):
   ```bash
   # Is HarvestBrain in LIVE mode?
   grep HARVEST_MODE /etc/quantum/harvest-brain.env
   
   # Should show: HARVEST_MODE=live
   # If shows: HARVEST_MODE=shadow → Enable live mode
   ```

---

## 🎛️ Monitoring Commands

### Quick Status Check
```bash
# Copy/paste this on VPS:
echo "=== SKIP_FLAT_SELL Status ===" && \
grep SKIP_FLAT_SELL /etc/quantum/intent-bridge.env && \
echo "" && \
echo "=== CLOSE Orders (last 100 entries) ===" && \
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -c "CLOSE" && \
echo "" && \
echo "=== Filtered SELL (last hour) ===" && \
journalctl -u quantum-intent-bridge --since "1 hour ago" --no-pager | grep -c "Skip publish.*SELL"
```

**Expected Output**:
```
=== SKIP_FLAT_SELL Status ===
INTENT_BRIDGE_SKIP_FLAT_SELL=false

=== CLOSE Orders (last 100 entries) ===
50-150

=== Filtered SELL (last hour) ===
0
```

### Live Exit Monitoring
```bash
# Watch for exit triggers in real-time
journalctl -u quantum-exit-monitor -f | grep "EXIT TRIGGERED"

# Watch for CLOSE orders being published
journalctl -u quantum-intent-bridge -f | grep "Published plan.*SELL"

# Watch apply.plan stream for CLOSE
redis-cli --csv XREAD BLOCK 5000 STREAMS quantum:stream:apply.plan $ | grep "CLOSE"
```

### Daily Summary
```bash
# Count exits in last 24h
journalctl -u quantum-exit-monitor --since "24 hours ago" --no-pager | grep -c "EXIT TRIGGERED"

# Count CLOSE orders published
journalctl -u quantum-intent-bridge --since "24 hours ago" --no-pager | grep -c "Published plan.*SELL"

# Check win rate (run expectancy audit)
python /home/qt/quantum_trader/audit_binance_testnet.py
```

---

## ✅ Action Items

### ✅ COMPLETED
- [x] Verify SKIP_FLAT_SELL=false in config file
- [x] Restart quantum-intent-bridge service
- [x] Confirm config loaded in service logs
- [x] Verify no SELL orders being filtered
- [x] Confirm CLOSE orders flowing in apply.plan

### 🔄 IN PROGRESS (Observation Phase)
- [ ] Monitor next 30-50 trades (24-48h)
- [ ] Track win rate improvement
- [ ] Verify exits executing on Binance
- [ ] Check for position drift reduction

### 📝 TODO (Post-Observation)
- [ ] Run expectancy audit after 30+ trades
- [ ] Document win rate change
- [ ] If successful: Update architecture docs
- [ ] If unsuccessful: Deep-dive into alternative failure modes

---

## 🔍 Troubleshooting

### Intent Bridge Not Starting
```bash
# Check logs for errors
journalctl -u quantum-intent-bridge -n 50 --no-pager

# Check env file syntax
cat /etc/quantum/intent-bridge.env | grep SKIP

# Restart service
systemctl restart quantum-intent-bridge
```

### SELL Orders Still Being Filtered
```bash
# Check current config in running process
journalctl -u quantum-intent-bridge -n 200 | grep "Skip flat SELL"

# Should show: Skip flat SELL: False
# If shows True: Config not loaded, check EnvironmentFile in service
```

### No CLOSE Orders in apply.plan
```bash
# Check if Exit Monitor is running
systemctl status quantum-exit-monitor

# Check if positions have TP/SL set
redis-cli HGETALL "quantum:position:snapshot:BTCUSDT"

# Check trade.intent stream for EXIT messages
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50
```

---

## 📚 References

- **Exit Pipeline Diagnostic**: [exit_pipeline_diagnostic.md](../exit_pipeline_diagnostic.md)
- **Service Config**: `/etc/quantum/intent-bridge.env`
- **Service File**: `/etc/systemd/system/quantum-intent-bridge.service`
- **Code**: `microservices/intent_bridge/main.py` lines 866-884

---

**Report Generated**: 2026-02-18 04:06 UTC  
**Next Review**: After 30+ trades (24-48h)  
**Success Criteria**: Win rate > 25%, positive expectancy, positions close at TP/SL
