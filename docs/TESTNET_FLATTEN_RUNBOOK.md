# Testnet Flatten Runbook

## Purpose
Safely flatten all open positions in testnet for emergency scenarios or complete position reset.

## ⚠️ DANGER: TESTNET ONLY

This feature is **DISABLED BY DEFAULT** and requires multiple safety confirmations. It will **CLOSE ALL OPEN POSITIONS** using market orders.

## When to Use

- Emergency: Need to close all positions immediately
- Testing: Reset to zero positions for clean test runs
- Risk Management: Position sizes have grown beyond acceptable limits
- Post-Incident: Clean up after a malfunction

## Prerequisites

### Hard Requirements (ALL must be met)

1. **ESS Active**: Emergency Stop System must be running
   - Latch file `/var/run/quantum/ESS_ON` must exist
   - Stops new trades from opening

2. **Config Flags Set** in `/etc/quantum/governor.env`:
   ```bash
   GOV_TESTNET_FORCE_FLATTEN=true
   GOV_TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW
   ```

3. **Redis Arm Key**: One-shot trigger
   ```bash
   redis-cli SET quantum:gov:testnet:flatten:arm 1 EX 60
   ```

4. **Rate Limit**: Maximum 1 flatten per 60 seconds
   - Prevents accidental rapid re-execution

### Automatic Checks (No-Op if Failed)

If any requirement is missing:
- Governor logs WARNING
- Increments `gov_testnet_flatten_noop_total` metric
- Does NOT execute flatten
- Does NOT crash

## Safety Mechanisms

### Triple Confirmation Required

1. **Environment Variable**: `GOV_TESTNET_FORCE_FLATTEN=true`
2. **Confirmation String**: `GOV_TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW`
3. **ESS Latch File**: `/var/run/quantum/ESS_ON` exists

### Fail-Safe Behavior

- Errors do NOT crash Governor
- Failed closes are logged but don't stop iteration
- Metrics track all attempts/failures
- Cooldown prevents rapid re-execution

### Rate Limiting

- Cooldown: 60 seconds between flatten operations
- Tracked via Redis key: `quantum:gov:testnet:flatten:last_ts`
- TTL: 1 hour

## Procedure

### Step 1: Activate ESS

```bash
bash /home/qt/quantum_trader/ops/ess_controller.sh activate
```

**Expected Output:**
```
✓ ESS ACTIVATED
  All trading operations STOPPED
```

### Step 2: Set Config Flags

```bash
# Add flags to Governor env file
echo "GOV_TESTNET_FORCE_FLATTEN=true" >> /etc/quantum/governor.env
echo "GOV_TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW" >> /etc/quantum/governor.env
```

### Step 3: Restart Governor

```bash
systemctl restart quantum-governor
```

**Verify in logs:**
```bash
journalctl -u quantum-governor --since "10 seconds ago" | grep flatten
```

**Expected:**
```
[WARNING] Testnet flatten ARMED (GOV_TESTNET_FORCE_FLATTEN=true + CONFIRM=FLATTEN_NOW)
[WARNING] Testnet flatten requires ESS active + Redis arm key
```

### Step 4: Arm Flatten (One-Shot Trigger)

```bash
redis-cli SET quantum:gov:testnet:flatten:arm 1 EX 60
```

**What Happens:**
1. Governor detects arm key on next periodic check (~1-2 seconds)
2. Deletes arm key immediately (prevents re-trigger)
3. Checks all safety requirements
4. Fetches open positions from Binance
5. Places reduceOnly MARKET close orders for each position
6. Logs completion: `TESTNET_FLATTEN done symbols=N orders=M`

### Step 5: Monitor Execution

```bash
# Follow logs
journalctl -u quantum-governor -f | grep -i flatten

# Check metrics
watch -n 2 'curl -s localhost:8044/metrics | grep gov_testnet_flatten'
```

**Expected Log:**
```
[WARNING] Testnet flatten: ALL SAFETY CHECKS PASSED - executing flatten
[INFO] Testnet flatten: Closing BTCUSDT SELL qty=0.001
[INFO] Testnet flatten: BTCUSDT close order placed: 123456789
[WARNING] TESTNET_FLATTEN done symbols=2 orders=2 errors=0
```

### Step 6: Verify Positions Closed

```bash
# Check Binance positions
redis-cli --scan --pattern "quantum:position:snapshot:*"

# Or check Governor logs for position checks
journalctl -u quantum-governor --since "1 minute ago" | grep "P2.9"
```

### Step 7: Cleanup

```bash
# Remove flatten flags
sed -i '/GOV_TESTNET_FORCE_FLATTEN/d' /etc/quantum/governor.env

# Restart Governor
systemctl restart quantum-governor

# Deactivate ESS
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

## Automated Proof Script

```bash
bash /home/qt/quantum_trader/scripts/proof_testnet_flatten.sh
```

**What it does:**
1. Activates ESS
2. Sets config flags
3. Restarts Governor
4. Arms flatten
5. Waits for execution (60s timeout)
6. Verifies logs and metrics
7. Cleans up config
8. Deactivates ESS
9. Prints PASS/FAIL summary

## Metrics

### Gauge Metrics

- `gov_testnet_flatten_enabled`: 1 if armed, 0 if disabled

### Counter Metrics

- `gov_testnet_flatten_attempt_total`: Total flatten attempts
- `gov_testnet_flatten_noop_total`: Attempts blocked by safety checks
- `gov_testnet_flatten_orders_total`: Close orders placed
- `gov_testnet_flatten_errors_total`: Errors during execution

### Monitoring Query

```bash
curl -s localhost:8044/metrics | grep gov_testnet_flatten
```

## Troubleshooting

### Issue: Flatten Armed but Not Executing

**Check arm key:**
```bash
redis-cli GET quantum:gov:testnet:flatten:arm
```

**If "1"**: Governor hasn't processed it yet (wait 1-2s)  
**If nil**: Already processed or expired

### Issue: "NO-OP" in Logs

**Check requirements:**
```bash
# 1. Config flags
grep FLATTEN /etc/quantum/governor.env

# 2. ESS latch
ls -l /var/run/quantum/ESS_ON

# 3. Cooldown
redis-cli GET quantum:gov:testnet:flatten:last_ts
```

### Issue: Orders Failed

**Check Binance connection:**
```bash
journalctl -u quantum-governor -n 100 | grep -i "binance\|order"
```

**Check position data:**
- Positions might already be zero
- Binance API might be returning errors
- Network connectivity issues

### Issue: Partial Flatten (Some Orders Failed)

**Normal behavior:**
- Governor continues through all positions even if some fail
- Check `errors` count in completion log
- Review individual order errors in logs

## Rollback

If flatten needs to be disabled immediately:

```bash
# Method 1: Remove flags and restart
sed -i '/GOV_TESTNET_FORCE_FLATTEN/d' /etc/quantum/governor.env
systemctl restart quantum-governor

# Method 2: Delete arm key (if not yet processed)
redis-cli DEL quantum:gov:testnet:flatten:arm

# Method 3: Deactivate ESS (prevents execution even if armed)
bash /home/qt/quantum_trader/ops/ess_controller.sh deactivate
```

## Production Safety

### Why This Won't Run in Production

1. **Testnet Detection**: Feature only in testnet mode
2. **ESS Requirement**: Production ESS activation is rare and deliberate
3. **Double Confirmation**: Requires exact string "FLATTEN_NOW"
4. **Manual Arming**: Requires explicit Redis command
5. **Rate Limiting**: Can't be spammed

### Additional Production Safeguards

- Separate production config file (no flatten flags)
- Production deployment scripts should audit env files
- Alerts on `gov_testnet_flatten_enabled` gauge in production (should be 0)

## Examples

### Example 1: Clean Slate Reset

```bash
# 1. Stop trading
bash ops/ess_controller.sh activate

# 2. Arm flatten
echo "GOV_TESTNET_FORCE_FLATTEN=true" >> /etc/quantum/governor.env
echo "GOV_TESTNET_FORCE_FLATTEN_CONFIRM=FLATTEN_NOW" >> /etc/quantum/governor.env
systemctl restart quantum-governor
redis-cli SET quantum:gov:testnet:flatten:arm 1 EX 60

# 3. Wait for completion
sleep 10
journalctl -u quantum-governor --since "30 seconds ago" | grep "TESTNET_FLATTEN done"

# 4. Resume clean
sed -i '/GOV_TESTNET_FORCE_FLATTEN/d' /etc/quantum/governor.env
systemctl restart quantum-governor
bash ops/ess_controller.sh deactivate
```

### Example 2: Emergency Flatten

```bash
# If positions are out of control and need immediate closure:
bash /home/qt/quantum_trader/scripts/proof_testnet_flatten.sh
```

## Related Documentation

- [Testnet Clean Slate Runbook](TESTNET_CLEAN_SLATE_RUNBOOK.md)
- [ESS Controller](../ops/ess_controller.sh)
- [Governor P2.9 Gate](../microservices/governor/)

## Quick Reference

```bash
# Arm flatten (one-liner after config set)
redis-cli SET quantum:gov:testnet:flatten:arm 1 EX 60

# Check if armed
redis-cli GET quantum:gov:testnet:flatten:arm

# Monitor execution
journalctl -u quantum-governor -f | grep flatten

# Check metrics
curl -s localhost:8044/metrics | grep gov_testnet_flatten

# Disarm immediately
redis-cli DEL quantum:gov:testnet:flatten:arm
```
