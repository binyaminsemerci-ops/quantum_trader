# GO-LIVE Rollback Procedure

**Version**: 1.0  
**Date**: December 4, 2025  
**Purpose**: Emergency deactivation of real trading

---

## When to Use This Procedure

Execute this rollback when:
- ✅ ESS triggered with critical issue
- ✅ Multiple system failures detected
- ✅ Unexpected losses exceed thresholds
- ✅ Exchange connectivity unstable
- ✅ Configuration error discovered
- ✅ Team decision to pause trading

---

## Rollback Steps (Execute in Order)

### STEP 1: Deactivate GO-LIVE Flag

```bash
# Remove activation flag immediately
rm go_live.active

# Verify deletion
ls go_live.active  # Should show: file not found
```

**Effect**: New orders will be skipped. Existing positions remain open.

---

### STEP 2: Disable Activation in Config

```bash
# Edit config file
# config/go_live.yaml

# Change from:
activation_enabled: true

# To:
activation_enabled: false
```

**Effect**: Prevents accidental re-activation without flag file.

---

### STEP 3: Set Accounts to Testnet Mode

For each configured exchange account:

1. **Edit exchange config** (e.g., `config/exchange.yaml`):
   ```yaml
   exchanges:
     binance:
       testnet: true  # Change from false to true
       api_key: "testnet_key"
       api_secret: "testnet_secret"
   ```

2. **OR use environment variable**:
   ```bash
   export BINANCE_TESTNET=true
   export BYBIT_TESTNET=true
   export OKX_TESTNET=true
   ```

**Effect**: All future orders route to testnet endpoints, not production.

---

### STEP 4: Restart Microservices

```bash
# Restart backend service to apply config changes
kubectl rollout restart deployment/quantum-trader-backend -n trading

# OR if running locally
pkill -f "uvicorn.*main:app"
python -m uvicorn backend.main:app --reload

# Wait for services to stabilize (30-60 seconds)
sleep 60
```

**Effect**: New configuration loaded, all connections reset.

---

### STEP 5: Verify Rollback

Run verification checks:

```bash
# 1. Check activation flag is gone
ls go_live.active
# Expected: No such file or directory

# 2. Check config shows disabled
grep "activation_enabled" config/go_live.yaml
# Expected: activation_enabled: false

# 3. Run pre-flight check (should PASS but show GO-LIVE disabled)
python scripts/preflight_check.py

# 4. Check logs for ORDER_SKIPPED messages
kubectl logs -n trading deployment/quantum-trader-backend --tail=50 | grep "ORDER_SKIPPED"
# Expected: Recent log entries showing orders skipped

# 5. Verify ESS status
curl http://localhost:8000/api/risk/ess/status
# Expected: {"active": true, "reason": "manual_deactivation"} OR {"active": false}
```

---

### STEP 6: Handle Open Positions

Choose one of the following strategies:

#### Option A: Let TP/SL Close Positions Naturally
- **Best for**: Normal rollback, no urgent market risk
- **Action**: Wait for TP/SL orders to trigger
- **Monitor**: Dashboard for position status

#### Option B: Manual Position Closure
- **Best for**: Emergency situations, unstable markets
- **Action**: Close positions via exchange UI or API
- **Steps**:
  1. Log into exchange (Binance/Bybit/OKX)
  2. Navigate to futures positions
  3. Close each position at market price
  4. Verify all positions closed

#### Option C: Cancel All Orders and Close
- **Best for**: Critical failures, need immediate exit
- **Action**: Cancel all open orders, then close positions
- **Steps**:
  ```bash
  # Use exchange API to cancel all orders
  curl -X DELETE "https://fapi.binance.com/fapi/v1/allOpenOrders" \
       -H "X-MBX-APIKEY: $BINANCE_API_KEY"
  
  # Then manually close positions (see Option B)
  ```

---

### STEP 7: Post-Rollback Checklist

After rollback completes, verify:

- [ ] `go_live.active` file deleted
- [ ] `config/go_live.yaml` shows `activation_enabled: false`
- [ ] All accounts set to testnet mode
- [ ] Microservices restarted successfully
- [ ] No new real orders placed (check logs)
- [ ] Open positions handled (closed or monitored)
- [ ] ESS status appropriate (active if emergency, inactive if normal)
- [ ] RiskGate decisions returning to normal
- [ ] Dashboard shows expected state

---

### STEP 8: Root Cause Analysis

Document the rollback event:

1. **Create incident report**:
   - Date/time of rollback
   - Trigger reason (ESS, loss, failure, etc.)
   - Systems affected
   - Positions at time of rollback
   - Total P&L impact

2. **Investigate root cause**:
   - Review logs for errors
   - Check metrics for anomalies
   - Interview operators for context
   - Identify configuration issues

3. **Action items**:
   - Fix bugs if discovered
   - Update configuration if needed
   - Improve monitoring/alerts
   - Update runbooks with learnings

4. **Share with team**:
   - Incident report document
   - Timeline of events
   - Lessons learned
   - Preventive measures

---

## Recovery from Rollback

To re-enable trading after rollback:

### Prerequisites

- [ ] Root cause identified and resolved
- [ ] Configuration errors fixed
- [ ] System stability verified (24h+ without issues)
- [ ] Testnet validation completed (5+ successful trades)
- [ ] Team consensus to proceed
- [ ] Risk limits reviewed and adjusted if needed

### Re-activation Steps

1. **Set accounts back to production mode**:
   ```yaml
   # config/exchange.yaml
   exchanges:
     binance:
       testnet: false  # Change from true to false
       api_key: "production_key"
       api_secret: "production_secret"
   ```

2. **Enable activation in config**:
   ```yaml
   # config/go_live.yaml
   activation_enabled: true
   ```

3. **Restart services**:
   ```bash
   kubectl rollout restart deployment/quantum-trader-backend -n trading
   ```

4. **Run pre-flight check**:
   ```bash
   python scripts/preflight_check.py
   # Must return exit code 0
   ```

5. **Run activation script**:
   ```bash
   python scripts/go_live_activate.py
   # Expected: GO-LIVE ACTIVATED SUCCESSFULLY
   ```

6. **Monitor closely for first 1 hour**:
   - Watch dashboard continuously
   - Verify first trade executes correctly
   - Check position sizing appropriate
   - Confirm TP/SL orders placed

---

## Emergency Contacts

| Severity | Contact | Response Time |
|----------|---------|---------------|
| **CRITICAL** | On-call Engineer | < 15 minutes |
| **HIGH** | Senior Operator | < 30 minutes |
| **MEDIUM** | Reliability Team | < 2 hours |
| **LOW** | Standard Support | Next business day |

### Contact Methods

- **On-call Engineer**: [oncall@quantum-trader.com] or [+1-555-0199]
- **Senior Operator**: [operator@quantum-trader.com] or [+1-555-0188]
- **Reliability Team**: [sre@quantum-trader.com]

---

## Rollback History

Keep a log of all rollback events:

| Date | Time (UTC) | Trigger | Duration | Impact | Resolved By |
|------|------------|---------|----------|--------|-------------|
| 2025-12-04 | 14:30 | ESS - Capital Loss | 2h 15m | -$150 | Operator-1 |
| _Add new rows as rollbacks occur_ | | | | | |

---

## Related Documentation

- **OPERATOR_MANUAL.md** - Daily operations and procedures
- **PROMPT_10_PREFLIGHT_CHECKLIST.md** - GO-LIVE checklist
- **BUILD_CONSTITUTION_AUDIT.md** - System architecture review
- **EPIC_GOLIVE_001_SUMMARY.md** - GO-LIVE implementation details

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Next Review**: January 4, 2026
