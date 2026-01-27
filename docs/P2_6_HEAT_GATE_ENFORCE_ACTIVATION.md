# P2.6 Heat Gate - Enforce Mode Activation

**Status**: Ready to activate (2026-01-27)  
**Risk Level**: LOW (fail-safe design + hash-only writes)

---

## ✅ Pre-Activation Verification (COMPLETED)

1. **Shadow Mode is Processing** ✅
   ```bash
   curl http://localhost:8056/metrics | grep p26_proposals_processed_total
   # Result: p26_proposals_processed_total 2.0
   ```

2. **Streams Have Data** ✅
   ```bash
   redis-cli XLEN quantum:stream:harvest.proposal    # Result: 15
   redis-cli XLEN quantum:stream:harvest.calibrated  # Result: 2
   ```

3. **Heat Calculation Working** ✅
   ```bash
   curl http://localhost:8056/metrics | grep p26_heat_value
   # Result: p26_heat_value 0.8825 (HOT)
   ```

4. **New Metrics Deployed** ✅
   ```bash
   curl http://localhost:8056/metrics | grep p26_hash_write
   # Result: p26_hash_writes_total, p26_hash_write_fail_total, p26_enforce_mode exposed
   ```

---

## 🚀 Activation Procedure

### Step 1: Set Enforce Mode (30 seconds)

```bash
# On VPS
ssh root@46.224.116.254

# Edit config
nano /etc/quantum/portfolio-heat-gate.env
# Change: P26_MODE=shadow
# To:     P26_MODE=enforce

# Restart service
systemctl restart quantum-portfolio-heat-gate

# Verify startup
systemctl status quantum-portfolio-heat-gate
# Should show: Mode: ENFORCE
```

### Step 2: Verify Hash Writes (1 minute)

```bash
# Check enforce mode metric
curl -s http://localhost:8056/metrics | grep p26_enforce_mode
# Expected: p26_enforce_mode 1.0

# Wait for a proposal to be processed
sleep 30

# Check hash write counter
curl -s http://localhost:8056/metrics | grep p26_hash_writes_total
# Expected: p26_hash_writes_total > 0

# Verify hash key exists for an active symbol
redis-cli KEYS "quantum:harvest:proposal:*"
# Pick one symbol, e.g., BTCUSDT

redis-cli HGETALL quantum:harvest:proposal:BTCUSDT
# Expected fields:
#   calibrated: "1"
#   calibrated_by: "p26_heat_gate"
#   original_action: "FULL_CLOSE"
#   action: "PARTIAL_75" (or similar)
#   heat_value: "0.8825"
#   heat_bucket: "HOT"
#   downgrade_reason: "portfolio_heat_warm"
```

### Step 3: Monitor Apply Layer (2 minutes)

```bash
# Watch Apply Layer logs for calibrated proposals
journalctl -u quantum-apply-layer -f -n 50
# Look for signs Apply Layer is using calibrated proposals:
#   - Different exit_type than expected
#   - Different close_qty

# Check Apply Layer metrics
curl -s http://localhost:8002/metrics | grep apply_
# Watch for changes in apply counts after heat gate processes proposals
```

### Step 4: Verify End-to-End Flow (5 minutes)

```bash
# Inject test proposal (HOT scenario - should allow FULL_CLOSE)
redis-cli XADD quantum:stream:harvest.proposal "*" \
  plan_id "test-$(date +%s)" \
  symbol "TESTUSDT" \
  action "FULL_CLOSE" \
  reason "tp_hit" \
  trace_id "$(uuidgen)"

# Check Heat Gate processed it
journalctl -u quantum-portfolio-heat-gate -n 20 | grep ENFORCE
# Expected: "📤 ENFORCE: test-... | TESTUSDT FULL_CLOSE→..."

# Check hash was written
redis-cli HGETALL quantum:harvest:proposal:TESTUSDT
# Should have calibrated=1

# Check stream was also written (for analysis)
redis-cli XREVRANGE quantum:stream:harvest.calibrated + - COUNT 1
# Should show test entry
```

---

## 🔍 Monitoring Checklist

**Key Metrics to Watch:**

```bash
# Heat Gate metrics
curl -s http://localhost:8056/metrics | egrep "p26_(enforce_mode|hash_writes_total|hash_write_fail|proposals_processed)" | grep -v "#"

# Expected:
p26_enforce_mode 1.0
p26_hash_writes_total 10+  (increasing)
p26_hash_write_fail_total 0  (should stay 0)
p26_proposals_processed_total 12+  (increasing)
```

**Service Health:**

```bash
systemctl status quantum-portfolio-heat-gate
# Should show: Active: active (running)

journalctl -u quantum-portfolio-heat-gate -n 50 --no-pager
# Look for:
#   - "Mode: ENFORCE"
#   - "📤 ENFORCE:" messages
#   - NO "❌ HASH WRITE FAILED" errors
```

---

## 🔴 Rollback Procedure (30 seconds)

If anything looks wrong:

```bash
# On VPS
nano /etc/quantum/portfolio-heat-gate.env
# Change: P26_MODE=enforce
# To:     P26_MODE=shadow

systemctl restart quantum-portfolio-heat-gate

# Verify rollback
curl -s http://localhost:8056/metrics | grep p26_enforce_mode
# Expected: p26_enforce_mode 0.0

journalctl -u quantum-portfolio-heat-gate -n 10
# Should show: Mode: SHADOW
```

**Optional: Clear calibrated hash keys**

```bash
# If you want to "reset" and remove all calibrated proposals
redis-cli KEYS "quantum:harvest:proposal:*" | while read key; do
  redis-cli HDEL "$key" calibrated calibrated_by original_action heat_value heat_bucket downgrade_reason
done
```

---

## 🧪 Test Scenarios

### Test 1: COLD Portfolio → Downgrade to PARTIAL_25

```bash
# Set low equity to force COLD state
redis-cli HMSET quantum:state:portfolio equity_usd 50000

# Create small position
redis-cli XADD quantum:stream:position.snapshot "*" \
  symbol "ETHUSDT" \
  position_notional_usd "5000" \
  sigma "0.15"

# Wait for heat calculation
sleep 5

# Inject FULL_CLOSE proposal
redis-cli XADD quantum:stream:harvest.proposal "*" \
  plan_id "test-cold-$(date +%s)" \
  symbol "ETHUSDT" \
  action "FULL_CLOSE" \
  reason "tp_hit"

# Verify downgrade
redis-cli HGET quantum:harvest:proposal:ETHUSDT action
# Expected: "PARTIAL_25"

journalctl -u quantum-portfolio-heat-gate -n 10 | grep DOWNGRADE
# Expected: "⚠️ DOWNGRADE: ETHUSDT FULL_CLOSE → PARTIAL_25"
```

### Test 2: WARM Portfolio → Downgrade to PARTIAL_75

```bash
# Set moderate equity
redis-cli HMSET quantum:state:portfolio equity_usd 10000

# Keep same position (heat ~0.30)
# Inject FULL_CLOSE
redis-cli XADD quantum:stream:harvest.proposal "*" \
  plan_id "test-warm-$(date +%s)" \
  symbol "ETHUSDT" \
  action "FULL_CLOSE"

# Verify
redis-cli HGET quantum:harvest:proposal:ETHUSDT action
# Expected: "PARTIAL_75"
```

### Test 3: HOT Portfolio → Allow FULL_CLOSE

```bash
# Set low equity (high heat)
redis-cli HMSET quantum:state:portfolio equity_usd 3000

# Inject FULL_CLOSE
redis-cli XADD quantum:stream:harvest.proposal "*" \
  plan_id "test-hot-$(date +%s)" \
  symbol "ETHUSDT" \
  action "FULL_CLOSE"

# Verify PASS
redis-cli HGET quantum:harvest:proposal:ETHUSDT action
# Expected: "FULL_CLOSE" (unchanged)

journalctl -u quantum-portfolio-heat-gate -n 10 | grep PASS
# Expected: "✅ PASS: ETHUSDT FULL_CLOSE"
```

---

## 📊 Success Criteria

**Enforce mode is working correctly when:**

1. ✅ `p26_enforce_mode = 1.0`
2. ✅ `p26_hash_writes_total > 0` and increasing
3. ✅ `p26_hash_write_fail_total = 0`
4. ✅ Hash keys exist: `quantum:harvest:proposal:{symbol}` with `calibrated=1`
5. ✅ Logs show `📤 ENFORCE:` messages
6. ✅ NO `❌ HASH WRITE FAILED` errors
7. ✅ Apply Layer uses calibrated proposals (different exit behavior)

---

## 🛡️ Safety Features

**Fail-Safe Design:**

1. **Fail-Open on Hash Write Error**: If hash write fails, error is logged but original proposal remains intact
2. **Fail-Closed on Missing Data**: Missing equity/positions → downgrade to PARTIAL_25 (safest)
3. **Monotonic Downgrades**: Only reduces aggressiveness, never increases
4. **Stream Preserved**: Stream write continues in both modes for analysis
5. **Instant Rollback**: Switch `P26_MODE=shadow` → no hash writes, no impact

---

## 🎯 When to Activate

**Activate NOW if:**

- ✅ Shadow mode has been processing successfully for 24+ hours
- ✅ Portfolio heat calculations are stable
- ✅ Metrics show reasonable heat values (not stuck at 0 or NaN)
- ✅ No errors in service logs

**Wait if:**

- ❌ Shadow mode shows errors or failures
- ❌ Heat values seem incorrect (always 0, always >10, etc.)
- ❌ Service is unstable (frequent restarts)

---

## 📝 Post-Activation

**Monitor for 24 hours:**

```bash
# Every hour, check:
watch -n 3600 'curl -s http://localhost:8056/metrics | egrep "p26_(enforce_mode|hash_writes_total|hash_write_fail)" | grep -v "#"'

# Daily report
curl -s http://localhost:8056/metrics | egrep "p26_" | grep -v "# " > /tmp/p26_metrics_$(date +%Y%m%d).txt
```

**Document activation:**

```bash
# Record activation in system
echo "$(date): P2.6 Heat Gate activated to ENFORCE mode" >> /var/log/quantum/activations.log
```
