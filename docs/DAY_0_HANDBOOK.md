# Day-0 Handbook

**Purpose**: Guide for the first day of live trading  
**Audience**: Founders & Operators  

---

## Before Starting

### Prerequisites Checklist

```
□ All tests passing (failure_scenarios, stress_tests, kill_switch_tests)
□ Shadow mode validated (minimum 50 trades)
□ Constitution documents reviewed and signed off
□ Scaling at Level 1 (PROOF_MODE)
□ $1,000 capital allocated (no more)
□ Alert channels configured and tested
□ Emergency contacts documented
□ Kill-switch tested within last 7 days
```

### Mental Preparation

Remember:
- Day-0 is about validation, not profit
- Expect small trades, small movements
- Watch for system behavior, not P&L
- Success = system working as designed
- Failure = learn and iterate, no harm done

---

## Day-0 Timeline

### T-60 minutes: Preparation

```
□ Login to monitoring dashboard
□ Check all services healthy (green)
□ Verify system in PROOF_MODE
□ Confirm max position size ($50)
□ Verify stop-loss system operational
□ Test Telegram alerts (send test)
□ Have emergency contact list ready
```

### T-30 minutes: Pre-Flight

```
□ Run pre-flight check
□ Verify all 10 checks pass
□ Review no-trade calendar (any blocks today?)
□ Check current market volatility
□ Confirm exchange API connectivity
□ Verify balance matches expectation
```

### T-0: Trading Begins

```
System is now live. Your job:
↓
Watch and document, do not intervene
↓
Note any unexpected behavior
↓
Trust the system unless critical issue
```

### During Trading

| If You See | Action |
|------------|--------|
| Trade entered | Note entry price, size, stop |
| Trade exits profitable | Note and log |
| Trade exits at stop | Note and log (normal) |
| Warning alert | Assess, usually watch |
| Critical alert | Investigate immediately |
| Kill-switch | Verify positions closed |

### T+8 hours (or End of Day): Review

```
□ Document all trades
□ Calculate actual vs expected P&L
□ Check for any policy violations
□ Review system logs for anomalies
□ Assess: Continue tomorrow or pause?
```

---

## What to Expect

### Normal Day-0 Outcomes

| Outcome | Frequency | Response |
|---------|-----------|----------|
| 0 trades | Very common | System is selective - this is good |
| 1-2 trades | Common | Expected for PROOF_MODE |
| Small loss | Common | Within limits is fine |
| Small profit | Normal | Great, continue |
| No issues | Ideal | Ready for Day-1 |

### Warning Signs (Not Critical)

| Sign | Meaning |
|------|---------|
| Multiple rejected trades | Policy working as designed |
| Higher than expected volatility | Size automatically reduced |
| Slippage on entry | Log and monitor |

### Critical Signs (Investigate)

| Sign | Action |
|------|--------|
| Kill-switch activated | Verify positions closed, review cause |
| System unresponsive | Check services, restart if needed |
| Position mismatch | Reconcile immediately |
| Unexpected large trade | Investigate - should not happen in Level 1 |

---

## Day-0 Success Criteria

### Must Have

- [ ] System operated without crashing
- [ ] All trades within policy limits
- [ ] Stop-losses set for all positions
- [ ] Alerts delivered properly
- [ ] No manual intervention needed

### Nice to Have

- [ ] At least one complete trade cycle
- [ ] Positive P&L (but not required)
- [ ] No warning alerts

### Automatic Fail

- [ ] Any trade exceeds 2% risk → Investigate
- [ ] Daily loss > 5% → System should have halted
- [ ] Position without stop → Critical bug
- [ ] System continues after kill-switch → Critical bug

---

## After Day-0

### If Successful

```
Day 0: $1,000, max $50/trade
Day 1-7: Continue Level 1
Day 8-30: If 15+ trades successful, consider Level 2 evaluation
```

### If Issues Found

```
1. Halt trading
2. Document all issues
3. Root cause analysis
4. Fix and re-test
5. Run shadow mode again
6. Repeat Day-0
```

---

## Emergency Quick Reference

### Kill-Switch (if needed)

```
Dashboard: Click red KILL button
Telegram: /killswitch
API: POST /api/kill-switch
```

### Manual Position Close (if automated fails)

```
1. Login to exchange
2. Close position manually
3. Document action
4. Report incident
```

### Contact Escalation

```
Level 1: Check dashboard, logs
Level 2: Contact other founder
Level 3: Exchange support
```

---

## Day-0 Log Template

```
Date: YYYY-MM-DD
Scaling Level: 1 (PROOF_MODE)
Capital: $1,000

Pre-Flight: [PASSED/FAILED]
Trading Start Time: HH:MM UTC
Trading End Time: HH:MM UTC

Trades:
| # | Symbol | Side | Entry | Exit | P&L |
|---|--------|------|-------|------|-----|
| 1 |        |      |       |      |     |

Total P&L: $X.XX
Daily Return: X.XX%

Incidents: [None / Describe]

Issues Found: [None / Describe]

Tomorrow Plan: [Continue / Pause / Investigate]

Signed: ____________
```

---

**END OF DAY-0 HANDBOOK**
