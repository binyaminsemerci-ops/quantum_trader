# Extended Validation - Quick Start Guide

**Ready to Execute**: February 1, 2026

---

## 30-Second System Check

```bash
# Copy/paste this to verify system is ready
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
echo "=== QUANTUM TRADER - PRE-VALIDATION CHECK ===";
echo "";
echo "1. Services:";
systemctl status quantum-trading-bot quantum-intent-bridge quantum-ai-engine | grep Active;
echo "";
echo "2. Redis:";
redis-cli ping;
echo "";
echo "3. Configuration:";
cat /etc/quantum/intent-bridge.env | grep -E "MAX_EXPOSURE|LOG_LEVEL";
echo "";
echo "4. Initial Metrics:";
echo "   Intents: $(redis-cli XLEN quantum:stream:trade.intent)";
echo "   Plans: $(redis-cli XLEN quantum:stream:apply.plan)";
echo "   Positions: $(redis-cli --raw HLEN quantum:ledger:latest || echo 0)";
echo "";
echo "✅ System ready for validation"
'
```

---

## Execution Phases

### Phase A: Launch Monitoring (T+0)

**Terminal 1**: Real-time metrics
```bash
watch -n 30 'ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "
echo \"Time: \$(date '+%H:%M:%S')\";
echo \"Pos: \$(redis-cli --raw HLEN quantum:ledger:latest || echo 0) | Exp: \$(redis-cli --raw GET quantum:portfolio:exposure_pct || echo N/A)%\";
echo \"Intents: \$(redis-cli XLEN quantum:stream:trade.intent) | Plans: \$(redis-cli XLEN quantum:stream:apply.plan)\"
"'
```

**Terminal 2**: Trading Bot logs
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-trading-bot -f | grep -E "BUY|SELL|confidence"'
```

**Terminal 3**: Intent Bridge logs
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-intent-bridge -f | grep -E "Parsed|Published|Added leverage"'
```

### Phase B: Wait for Entries (T+5 to T+15)

**What to Watch**:
- ✅ First WAVESUSDT BUY signal appears in trading-bot logs
- ✅ Intent Bridge parses it: "✓ Parsed WAVESUSDT BUY: leverage=10.0"
- ✅ apply.plan populated: position_count goes 0→1
- ✅ Logs show no errors

**Expected Timeline**:
- T+3-5 min: First signal generated
- T+5-10 min: First entry in apply.plan
- T+10-15 min: First position confirmed

### Phase C: Monitor Accumulation (T+15 to T+45)

**Every 5 Minutes**:
```bash
redis-cli --raw HGETALL quantum:ledger:latest | head -30  # See positions
redis-cli --raw GET quantum:portfolio:exposure_pct         # Check exposure
```

**What to Expect**:
- Positions: 1 → 2 → 3 → 4 (gradual)
- Exposure: 15% → 30% → 50% → 65% (linear growth)
- Multiple symbols entering
- All entries with leverage=10.0

### Phase D: Test Exposure Gate (T+45 to T+90)

**Watch for Rejection Messages**:
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-intent-bridge -f | grep "exposure.*rejected"'
```

**What to Expect**:
- As exposure approaches 80%, BUY signals still generated
- But Intent Bridge logs: "Skip publish: {SYMBOL} BUY rejected (exposure=79.5% >= MAX=80.0%)"
- Position count plateaus
- SELL signals still process (to reduce exposure)

### Phase E: Extended Monitoring (T+90 to T+120+)

**Continuous Checks**:
- Service stability (no crashes)
- Error rate (should be 0)
- Metadata flow (leverage/TP/SL present)
- Permit chain processing

---

## Quick Reference: Key Commands

### Check Current State
```bash
VPS="root@46.224.116.254"
KEY="~/.ssh/hetzner_fresh"

# Positions and exposure
ssh -i $KEY $VPS 'redis-cli --raw HGETALL quantum:ledger:latest'

# Exposure percentage
ssh -i $KEY $VPS 'redis-cli --raw GET quantum:portfolio:exposure_pct'

# Recent entries
ssh -i $KEY $VPS 'redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 5'

# Error count (last hour)
ssh -i $KEY $VPS 'journalctl -u quantum-intent-bridge --since "1 hour ago" | grep -i error | wc -l'

# Current leverage verification
ssh -i $KEY $VPS 'redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 1 | grep -A 1 leverage'
```

### Check Services
```bash
# All services
ssh -i $KEY $VPS 'systemctl status quantum-trading-bot quantum-intent-bridge quantum-ai-engine --no-pager'

# Restart if needed
ssh -i $KEY $VPS 'systemctl restart quantum-trading-bot'
```

### View Logs
```bash
# Trading Bot (last 20 lines)
ssh -i $KEY $VPS 'journalctl -u quantum-trading-bot -n 20'

# Intent Bridge (last 20 lines)
ssh -i $KEY $VPS 'journalctl -u quantum-intent-bridge -n 20'

# Real-time (follows)
ssh -i $KEY $VPS 'journalctl -u quantum-intent-bridge -f'
```

### Collect Data
```bash
# Save current state
ssh -i $KEY $VPS 'redis-cli BGSAVE'

# Export metrics
ssh -i $KEY $VPS 'journalctl -u quantum-intent-bridge --since "30 minutes ago" > /tmp/intent_bridge_log.txt'

# Download logs
scp -i $KEY $VPS:/tmp/*.txt ./validation_logs/
```

---

## Troubleshooting Quick Fixes

### Issue: No entries after 15 minutes
```bash
# Check trading-bot generating signals
ssh -i $KEY $VPS 'journalctl -u quantum-trading-bot -n 50 | grep -i "buy\|sell"'

# Check AI engine is responding
ssh -i $KEY $VPS 'curl -s http://localhost:8001/health'

# Restart trading-bot if needed
ssh -i $KEY $VPS 'systemctl restart quantum-trading-bot'
```

### Issue: Leverage field missing
```bash
# Verify trading-bot code (should be leverage=10.0)
ssh -i $KEY $VPS 'grep -n "leverage" /home/qt/quantum_trader/microservices/trading_bot/simple_bot.py | head -5'

# Check recent entries
ssh -i $KEY $VPS 'redis-cli --raw XREVRANGE quantum:stream:apply.plan 0 + COUNT 1 | grep leverage'

# If missing, git reset and restart
ssh -i $KEY $VPS 'cd /home/qt/quantum_trader && git fetch origin && git reset --hard origin/main && systemctl restart quantum-trading-bot'
```

### Issue: Exposure not increasing
```bash
# Check exposure calculation
ssh -i $KEY $VPS 'redis-cli --raw GET quantum:portfolio:notional_usd'
ssh -i $KEY $VPS 'redis-cli --raw GET quantum:account:equity_usd'

# Manual calc: (notional / equity) * 100

# Check exposure is being stored
ssh -i $KEY $VPS 'redis-cli --raw GET quantum:portfolio:exposure_pct'
```

### Issue: Positions not created
```bash
# Check if plans are in apply.plan
ssh -i $KEY $VPS 'redis-cli XLEN quantum:stream:apply.plan'

# Check if permits are processing
ssh -i $KEY $VPS 'journalctl -u quantum-governor -n 20 | tail -10'

# Check execution layer
ssh -i $KEY $VPS 'redis-cli XLEN quantum:stream:execution.result'
```

---

## Success Indicators

✅ **First 5 Minutes**: System generating signals  
✅ **15 Minutes**: First entry in apply.plan with leverage=10.0  
✅ **30 Minutes**: 2-3 positions created, exposure 30-50%  
✅ **60 Minutes**: 4-5 positions, exposure 60-80%  
✅ **90 Minutes**: Exposure gate active, BUY rejections logged  
✅ **120 Minutes**: System stable, no crashes, metrics consistent  

---

## Go/No-Go Decision

### GO if:
- ✅ All services running
- ✅ Leverage=10.0 in all entries
- ✅ Position count emerged gradually (1→2→3→...)
- ✅ Exposure climbed to 75-80%
- ✅ BUY rejections occurred at ≥80%
- ✅ Zero critical errors
- ✅ Metadata present through entire pipeline

### NO-GO if:
- ❌ Services crashed or unreachable
- ❌ Leverage field missing or wrong value
- ❌ Position count jumped unexpectedly
- ❌ Exposure exceeded 100%
- ❌ Permit chain not processing entries
- ❌ Critical errors in logs
- ❌ Metadata lost between streams

---

## After Validation

### If Successful (GO)
```bash
1. Fill out VALIDATION_TEST_REPORT_TEMPLATE.md
2. Archive logs and metrics
3. Update system status
4. Schedule production readiness review
```

### If Issues Found (NO-GO)
```bash
1. Document in report
2. Identify root causes
3. Create fix tickets
4. Schedule re-validation
```

---

## Summary

1. **Start**: Run 3 monitoring terminals (metrics, bot logs, bridge logs)
2. **Wait**: Observe entries accumulating over 90 minutes
3. **Verify**: Check leverage, exposure, position count progression
4. **Document**: Fill report with actual metrics
5. **Decide**: GO/NO-GO based on success criteria

**Estimated Duration**: 2-4 hours  
**Resource Required**: SSH access to VPS, monitoring terminals  
**Documentation**: VALIDATION_TEST_REPORT_TEMPLATE.md

---

**Status**: READY TO EXECUTE 🚀

Next: Start validation with all monitoring active
