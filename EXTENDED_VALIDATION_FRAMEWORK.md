# Extended LIVE Testnet Validation - Operational Framework

**Start Date**: February 1, 2026  
**Duration**: 2-4 hours minimum  
**Objective**: Validate AI-driven position sizing under live market conditions  
**Status**: READY FOR EXECUTION

---

## Validation Objectives

### Primary Goals
1. ✅ **RL Position Sizing**: Confirm AI determines position count dynamically (not hardcoded)
2. ✅ **Portfolio Exposure**: Verify 80% MAX_EXPOSURE_PCT limit gates new entries correctly
3. ✅ **Leverage Flow**: Validate leverage/TP/SL metadata reaches execution layer
4. ✅ **Market Adaptation**: Monitor system behavior as market moves and exposure changes
5. ✅ **Permit Chain**: Confirm Governor + P2.6 + P3.3 gates process RL metadata

### Success Criteria

| Criterion | Target | Validation Method |
|-----------|--------|-------------------|
| Leverage in apply.plan | 100% | Redis XREVRANGE on entries |
| Position count emergence | <8 positions at any time | Query ledger, verify non-hardcoded |
| Exposure limit enforcement | ≤80% portfolio | Calculate (notional/equity)*100 |
| TP/SL presence | 100% of entries | Check apply.plan fields |
| Fallback signal leverage | 10.0x minimum | Verify trade.intent payloads |
| Permit gate throughput | No delays | Monitor execution latency |

---

## Pre-Validation Checklist

### System Health
- [ ] All services active: `systemctl status quantum-*`
- [ ] Redis connected and responsive: `redis-cli ping`
- [ ] Streams have data: `redis-cli DBSIZE`
- [ ] Logging at DEBUG level: `INTENT_BRIDGE_LOG_LEVEL=DEBUG`
- [ ] Allowlist contains test symbols: Verify WAVESUSDT in allowlist

### Configuration Verification
```bash
# Check KEY settings on VPS
cat /etc/quantum/intent-bridge.env | grep -E 'MAX_EXPOSURE|ALLOWLIST|LOG_LEVEL'
```

**Expected Output**:
```
MAX_EXPOSURE_PCT=80.0
INTENT_BRIDGE_ALLOWLIST=31 symbols (WAVESUSDT included)
INTENT_BRIDGE_LOG_LEVEL=DEBUG
```

### Baseline Capture
```bash
# Before starting validation, capture initial state
redis-cli INFO stats > baseline_stats.txt
redis-cli XLEN quantum:stream:trade.intent > baseline_intents.txt
redis-cli XLEN quantum:stream:apply.plan > baseline_plans.txt
journalctl -u quantum-trading_bot -n 100 > baseline_logs_bot.txt
journalctl -u quantum-intent-bridge -n 100 > baseline_logs_bridge.txt
```

---

## Validation Phases

### Phase 1: Stabilization (First 15 minutes)
**Goal**: System reaches steady state, generates first entries

**Monitoring**:
- Watch for BUY signals on allowlisted symbols
- Verify leverage=10.0 in logs: `grep "leverage=10" logs`
- Check first positions created: `redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 5`
- Monitor trading-bot health: `journalctl -u quantum-trading_bot -f`

**Expected Behavior**:
- WAVESUSDT generates BUY signals (high momentum +1.04%)
- WAVESUSDT entries published with leverage=10.0
- Intent Bridge accepts and publishes plans
- No errors in service logs

**Pass Criteria**: 
- At least 1 entry in apply.plan within 2 minutes
- All entries have leverage field populated

---

### Phase 2: Accumulation (15-45 minutes)
**Goal**: Build portfolio positions, test exposure limiting

**Monitoring**:
```bash
# Every 5-10 minutes, check:
# 1. Current positions
redis-cli --raw HGETALL quantum:ledger:latest

# 2. Portfolio exposure
redis-cli --raw GET quantum:portfolio:exposure_pct

# 3. Recent entries
redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 10

# 4. Position count
redis-cli --raw HLEN quantum:ledger:latest
```

**Key Metrics to Track**:
- **Position Count**: Should start at 1, grow gradually (not jump to 8)
- **Portfolio Exposure**: Should climb toward 80% (not spike over)
- **Entry Distribution**: Verify multiple symbols (not just WAVESUSDT)
- **Leverage Consistency**: All entries should have leverage=10.0

**Expected Behavior**:
- First entry: 1 position, ~15% exposure
- Second entry: 2 positions, ~30% exposure
- Third entry: 3 positions, ~50% exposure
- Continues until exposure approaches 80%

**Pass Criteria**:
- Position count increases gradually (1→2→3...)
- Exposure climbs linearly
- No sudden position spikes
- All new entries have leverage/TP/SL fields

---

### Phase 3: Exposure Limiting (45-90 minutes)
**Goal**: Test that system blocks new entries when exposure ≥80%

**Monitoring**:
```bash
# Watch Intent Bridge logs for rejection messages
journalctl -u quantum-intent-bridge -f | grep -E 'exposure|rejected|BUY rejected'

# Calculate real-time exposure
calculate_exposure() {
  TOTAL_NOTIONAL=$(redis-cli --raw GET quantum:portfolio:notional_usd)
  EQUITY=$(redis-cli --raw GET quantum:account:equity_usd)
  if [ ! -z "$TOTAL_NOTIONAL" ] && [ ! -z "$EQUITY" ]; then
    echo "scale=1; ($TOTAL_NOTIONAL / $EQUITY) * 100" | bc
  fi
}
```

**Expected Behavior**:
- As exposure approaches 80%, BUY signals continue but are NOT published
- Intent Bridge logs show: "Skip publish: {SYMBOL} BUY rejected (exposure=79.5% >= MAX=80.0%)"
- Position count plateaus at current level
- SELL signals still processed (to reduce exposure)

**Pass Criteria**:
- BUY rejection messages appear in logs
- Rejection triggered when exposure ≥ 80%
- No NEW positions created above 80%
- Position count stabilizes

---

### Phase 4: Stress Conditions (90-120 minutes)
**Goal**: Monitor system under volatility, test permit chain stability

**Monitoring**:
```bash
# Watch for rapid market moves
# Check permit gate throughput
journalctl -u quantum-governor -f | tail -20
journalctl -u quantum-p26-permit -f | tail -20
journalctl -u quantum-p33-permit -f | tail -20

# Monitor execution layer
redis-cli --raw XREVRANGE quantum:stream:execution.result 0 + COUNT 20
```

**Expected Behavior**:
- Governor processes entries within SLA
- P2.6 validates plans with RL metadata
- P3.3 executes with correct leverage/TP/SL
- No metadata loss through permit chain
- Positions adjust TP/SL based on volatility

**Pass Criteria**:
- No execution errors or timeouts
- All permits process entries cleanly
- Metadata preserved through chain
- Execution latency < 500ms per entry

---

## Real-Time Monitoring Dashboard

### Command 1: System Overview (Run every 30 seconds)
```bash
watch -n 30 '
echo "=== QUANTUM TRADER - LIVE STATUS ===";
echo "Time: $(date)";
echo "";
echo "SERVICES:";
systemctl status quantum-trading-bot quantum-intent-bridge quantum-ai-engine --no-pager | grep -E "Active:|●";
echo "";
echo "POSITIONS:";
redis-cli --raw HGETALL quantum:ledger:latest | head -20;
echo "";
echo "EXPOSURE:";
redis-cli --raw GET quantum:portfolio:exposure_pct;
echo "";
echo "RECENT ENTRIES:";
redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 5 | grep -E "symbol|leverage|side";
'
```

### Command 2: Entry Flow Tracking (Real-time logs)
```bash
# Terminal 1: Trading Bot signals
journalctl -u quantum-trading_bot -f | grep -E "BUY|SELL|confidence"

# Terminal 2: Intent Bridge bridging
journalctl -u quantum-intent-bridge -f | grep -E "Parsed|Published|Added leverage"

# Terminal 3: Permit chain
journalctl -u quantum-governor -f | grep -E "decision|leverage"
```

### Command 3: Metrics Collection (Every 2 minutes)
```bash
log_metrics() {
  TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
  POSITIONS=$(redis-cli --raw HLEN quantum:ledger:latest)
  EXPOSURE=$(redis-cli --raw GET quantum:portfolio:exposure_pct)
  INTENTS=$(redis-cli XLEN quantum:stream:trade.intent)
  PLANS=$(redis-cli XLEN quantum:stream:apply.plan)
  
  echo "$TIMESTAMP | Positions: $POSITIONS | Exposure: $EXPOSURE% | Intents: $INTENTS | Plans: $PLANS" >> validation_metrics.log
}

while true; do
  log_metrics
  sleep 120
done
```

---

## Validation Checklist - During Test

### Every 15 Minutes
- [ ] All services still running
- [ ] No error logs in any service
- [ ] Position count tracking matches expectations
- [ ] Exposure % within reasonable bounds
- [ ] Leverage field in all new entries
- [ ] Stop-loss and take-profit fields present

### Every 30 Minutes
- [ ] Review Intent Bridge logs for anomalies
- [ ] Check permit gate throughput (no delays)
- [ ] Verify execution layer processing entries
- [ ] Compare expected vs actual position count
- [ ] Confirm exposure calculation accuracy

### Every 60 Minutes
- [ ] Full system metrics capture
- [ ] Review RL Agent sizing decisions
- [ ] Analyze market impact on leverage/TP/SL
- [ ] Check for log file size growth (rotation working?)
- [ ] Verify Redis memory usage reasonable

---

## Failure Scenarios & Response

### Scenario 1: Leverage Field Missing
**Symptom**: apply.plan entries have no leverage field
**Root Cause**: Parse/publish logic failure or trading-bot regression
**Response**:
1. Stop validation test
2. Check trading-bot logs: `journalctl -u quantum-trading_bot -n 50`
3. Verify fallback signal: `grep "leverage" microservices/trading_bot/simple_bot.py | head -3`
4. Restart trading-bot if needed
5. Resume validation

### Scenario 2: Position Count Exceeds Limit
**Symptom**: More than expected positions created despite <80% exposure
**Root Cause**: Exposure calculation error or allowlist issue
**Response**:
1. Capture current state: `redis-cli HGETALL quantum:ledger:latest`
2. Calculate manual exposure: `(notional / equity) * 100`
3. Check allowlist size: `redis-cli --raw GET INTENT_BRIDGE_ALLOWLIST | wc -w`
4. Review Intent Bridge calculation logic
5. Check if exposure check being bypassed

### Scenario 3: Exposure Blocks All BUY Signals
**Symptom**: No new entries after reaching 60% exposure
**Root Cause**: Exposure limit too conservative or incorrectly blocking
**Response**:
1. Check exposure calc: `redis-cli --raw GET quantum:portfolio:exposure_pct`
2. Verify MAX_EXPOSURE_PCT value: `cat /etc/quantum/intent-bridge.env | grep MAX`
3. Check if ALL symbols hitting exposure gate: `journalctl -u quantum-intent-bridge | grep "exposure="`
4. May need to adjust MAX_EXPOSURE_PCT if test is too conservative

### Scenario 4: Execution Layer Not Processing RL Metadata
**Symptom**: Positions created but leverage/TP/SL not respected
**Root Cause**: Permit gates ignoring metadata fields
**Response**:
1. Check permit gate logs: `journalctl -u quantum-governor | tail -30`
2. Verify metadata present in apply.plan: `redis-cli --raw XREVRANGE quantum:stream:apply.plan -1 +1 | grep leverage`
3. Check if permits consuming apply.plan correctly
4. Test permit gate with manual test message

---

## Data Collection for Analysis

### Collect Every Hour
```bash
# Create hourly snapshot
HOUR=$(date +%Y%m%d_%H00)

# Position history
redis-cli --raw HGETALL quantum:ledger:latest > validation_positions_$HOUR.json

# Exposure history
echo "$(date): $(redis-cli --raw GET quantum:portfolio:exposure_pct)%" >> validation_exposure_history.txt

# Entry count
echo "$(date): Intents=$(redis-cli XLEN quantum:stream:trade.intent), Plans=$(redis-cli XLEN quantum:stream:apply.plan)" >> validation_streams_history.txt

# Log excerpts
journalctl -u quantum-trading_bot --since "1 hour ago" > validation_logs_bot_$HOUR.txt
journalctl -u quantum-intent-bridge --since "1 hour ago" > validation_logs_bridge_$HOUR.txt
```

### Analysis Templates
```python
# Python analysis script (post-validation)
import json
from datetime import datetime

def analyze_validation():
    metrics = []
    
    # Parse metrics log
    with open('validation_metrics.log') as f:
        for line in f:
            # Extract: TIMESTAMP | Positions: X | Exposure: Y% | ...
            parts = line.split('|')
            timestamp = parts[0].strip()
            positions = int(parts[1].split(':')[1].strip())
            exposure = float(parts[2].split(':')[1].strip().rstrip('%'))
            metrics.append({
                'timestamp': timestamp,
                'positions': positions,
                'exposure': exposure
            })
    
    # Analysis
    print(f"Test Duration: {len(metrics) * 2} minutes")
    print(f"Max Positions: {max(m['positions'] for m in metrics)}")
    print(f"Max Exposure: {max(m['exposure'] for m in metrics)}%")
    print(f"Position Count Progression: {[m['positions'] for m in metrics[::3]]}")
```

---

## Expected Results After 2-4 Hours

### Success Indicators
- ✅ Position count grew from 1 to 4-6 (not hardcoded 8)
- ✅ Exposure climbed gradually to ~75-80%
- ✅ Multiple symbols had positions (not just WAVESUSDT)
- ✅ All entries had leverage=10.0, correct TP/SL
- ✅ Portfolio exposure limit prevented >80% positions
- ✅ No leverage metadata lost through pipeline
- ✅ Permit gates processed all entries without errors
- ✅ System remained stable under market volatility

### Metrics to Report
```
VALIDATION REPORT
================
Duration: 2-4 hours
Start Time: [timestamp]
End Time: [timestamp]
Status: PASS/FAIL

Key Results:
- Positions Created: X (range: 4-6 expected)
- Max Exposure Reached: Y% (target: 75-80%)
- Entries Processed: Z (all should have leverage)
- Leverage=10.0 Entries: Z (should be 100%)
- TP/SL Present: W% (should be 100%)
- Exposure Rejections: V (expected >0 if reached 80%)
- Error Count: 0 (target)
- Execution Latency: <500ms avg

Position Distribution:
- WAVESUSDT: A positions
- [Other symbols]: B positions each
- Total Notional: $XXX
- Portfolio Equity: $YYY

Conclusions:
1. AI-driven position sizing: [CONFIRMED/FAILED]
2. Exposure limiting: [CONFIRMED/FAILED]
3. Leverage metadata flow: [CONFIRMED/FAILED]
4. Market adaptation: [CONFIRMED/FAILED]
```

---

## Abort Criteria

Stop validation immediately if:
1. **Critical Error**: Service crashes (non-restart)
2. **Data Loss**: RL metadata missing from apply.plan
3. **Runaway Positions**: Position count exceeds 10
4. **Exposure Exceeded**: Portfolio exposure >100%
5. **Permit Chain Failure**: Execution layer not processing entries
6. **Fund Loss**: Unexpected account equity drop >5%

---

## Post-Validation Actions

### If Successful (PASS)
1. Document results in `VALIDATION_RESULTS.md`
2. Archive logs and metrics
3. Update system status to "VALIDATED FOR PRODUCTION"
4. Begin LIVE testnet ramp (SUM_MODE=CONTROLLED, SOLUSDT, position_size_usd=150)
5. Move to production decision review

### If Partial Success (NEEDS REVIEW)
1. Identify specific failures
2. Create bug fixes or improvements
3. Deploy fixes to testnet
4. Re-validate specific components
5. Document lessons learned

### If Failed (CRITICAL)
1. Immediately capture full diagnostic state
2. Disable LIVE trading (set to TESTNET_SHADOW)
3. Review root cause analysis
4. Make necessary fixes
5. Plan re-validation cycle

---

## Pre-Validation System Checklist

```bash
# Run this before starting validation
#!/bin/bash

echo "=== PRE-VALIDATION SYSTEM CHECK ==="
echo ""

# 1. Services
echo "[1] Services Status:"
systemctl status quantum-trading-bot quantum-intent-bridge quantum-ai-engine | grep Active

# 2. Redis
echo "[2] Redis Connection:"
redis-cli ping

# 3. Configuration
echo "[3] Configuration:"
cat /etc/quantum/intent-bridge.env | grep -E "MAX_EXPOSURE|ALLOWLIST|LOG_LEVEL"

# 4. Baseline metrics
echo "[4] Baseline Metrics:"
echo "  - Intents in stream: $(redis-cli XLEN quantum:stream:trade.intent)"
echo "  - Plans in stream: $(redis-cli XLEN quantum:stream:apply.plan)"
echo "  - Redis memory: $(redis-cli INFO memory | grep used_memory_human)"

# 5. Log health
echo "[5] Recent Logs (last 10 seconds):"
journalctl -u quantum-trading-bot --since "10s ago" | tail -3

echo ""
echo "✅ System ready for validation"
```

---

## Execution Plan

1. **T+0 min**: Start all monitoring terminals
2. **T+0 min**: Capture baseline metrics
3. **T+5 min**: Verify first WAVESUSDT entry created
4. **T+15 min**: Check position accumulation starting
5. **T+45 min**: Monitor exposure approaching 80%
6. **T+75 min**: Verify exposure gate rejecting new BUY signals
7. **T+120 min**: Review collected data and metrics
8. **T+120+ min**: Extend to 4 hours if no issues

---

**Status**: VALIDATION FRAMEWORK READY FOR EXECUTION 🚀

Next: Execute extended LIVE testnet validation with full monitoring
