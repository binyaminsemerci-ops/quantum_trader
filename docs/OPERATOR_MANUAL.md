# Quantum Trader v2.0 - Operator Manual

**Version**: 1.0  
**Date**: December 4, 2025  
**Audience**: Trading Operators, System Reliability Engineers

---

## Quick Reference

| Action | Command |
|--------|---------|
| **Pre-flight check** | `python scripts/preflight_check.py` |
| **Activate GO-LIVE** | `python scripts/go_live_activate.py` |
| **Deactivate trading** | `rm go_live.active` or edit `config/go_live.yaml` |
| **Check system health** | Open Risk & Resilience Dashboard (Grafana) |
| **Emergency stop** | ESS triggers automatically OR `rm go_live.active` |

---

## Section 1: Daily Routine

### Start of Day (Before Market Open)

1. **Check dashboards**
   - Open Risk & Resilience Dashboard in Grafana
   - Verify all panels showing green/normal status
   - Check for any overnight alerts or incidents

2. **Check risk state**
   - Global Risk status should be `OK` (not `CRITICAL`)
   - No active positions with excessive drawdown
   - Capital profiles correctly assigned to all accounts

3. **Check ESS status**
   - Emergency Stop System should be INACTIVE
   - If ESS is active, investigate root cause before resuming trading
   - Review ESS trigger history for patterns

4. **Run pre-flight check**
   ```bash
   python scripts/preflight_check.py
   ```
   - Must return exit code 0 (all checks passed)
   - If exit != 0, investigate failures before proceeding

### During Trading Hours

- **Monitor dashboard every 30 minutes** (minimum)
- **Check for anomalies**:
  - Unusual position sizes
  - Unexpected exchange failovers
  - High RiskGate block rate
  - Rising stress scenario severity
- **Review recent trades**:
  - Verify execution prices are reasonable
  - Check TP/SL orders placed correctly
  - Confirm position sizing within limits

### End of Day

- **Review daily performance**:
  - Total P&L vs expectations
  - Number of trades executed
  - ESS trigger count (should be zero or near-zero)
  - Exchange failover count
- **Check overnight positions**:
  - All positions have TP/SL orders
  - Margin utilization within limits
  - No positions exceeding max size
- **Update operational log**:
  - Document any incidents or anomalies
  - Record ESS triggers with root cause
  - Note any manual interventions

---

## Section 2: Activating Real Trading

### Prerequisites

Before activating GO-LIVE, ensure:

- ✅ All EPICs completed (Risk3, RiskGate, ESS, Profiles, Observability, Stress, Preflight, K8s)
- ✅ Backend services running and healthy
- ✅ Risk & Resilience Dashboard imported to Grafana
- ✅ All accounts configured with MICRO profile
- ✅ Testnet validation completed (minimum 3 successful test trades)
- ✅ Team trained on dashboard, runbooks, emergency procedures

### Activation Steps

1. **Run pre-flight check**
   ```bash
   python scripts/preflight_check.py
   ```
   - If exit code = 0 → Continue
   - If exit code != 0 → Investigate failures, DO NOT activate

2. **Edit GO-LIVE config**
   ```bash
   # Edit config/go_live.yaml
   # Change: activation_enabled: false
   # To:     activation_enabled: true
   ```

3. **Run activation script**
   ```bash
   python scripts/go_live_activate.py
   ```
   - Expected output: `✅ GO-LIVE ACTIVATED SUCCESSFULLY`
   - If activation fails, check logs and config

4. **Verify activation**
   ```bash
   # Check activation flag exists
   ls go_live.active
   
   # Should see: go_live.active
   ```

5. **Monitor first hour**
   - Watch dashboard continuously
   - Verify first trade executes correctly
   - Check positions page shows accurate data
   - Confirm TP/SL orders placed on Binance

---

## Section 3: Stopping Trading

### Normal Deactivation

To stop trading in controlled manner:

1. **Remove activation flag**
   ```bash
   rm go_live.active
   ```

2. **OR edit config**
   ```yaml
   # config/go_live.yaml
   activation_enabled: false
   ```

3. **Verify deactivation**
   - New orders will be skipped (check logs: `ORDER_SKIPPED`)
   - Existing positions remain open
   - TP/SL orders remain active

4. **Optional: Close all positions**
   - Use exchange UI or API to manually close
   - Or wait for TP/SL to trigger naturally

### Emergency Deactivation

If immediate stop required:

1. **Delete activation flag**
   ```bash
   rm go_live.active
   ```

2. **Trigger ESS (if needed)**
   - ESS will automatically block all new orders
   - Existing positions will be closed by TP/SL
   - See docs/GO_LIVE_ROLLBACK.md for full procedure

---

## Section 4: Emergency Procedures

### ESS Triggered

**Symptoms**: ESS status = ACTIVE, all orders blocked

**Actions**:
1. Check dashboard for trigger reason (capital loss, drawdown, etc.)
2. Review recent trades for cause
3. If legitimate risk event:
   - Let ESS remain active
   - Close positions manually if needed
   - Investigate root cause
4. If false positive:
   - Review ESS thresholds in risk config
   - Reset ESS (requires manual intervention)
   - Monitor closely after reset

### Exchange Outage

**Symptoms**: Exchange failover count increasing, errors in logs

**Actions**:
1. Check exchange status page (Binance/Bybit/OKX)
2. Verify fallback exchange is healthy
3. If outage is prolonged:
   - Consider deactivating trading
   - Close positions if market conditions worsen
4. Document outage duration and impact

### RiskGate Blocking Everything

**Symptoms**: All orders blocked by RiskGate, no execution

**Actions**:
1. Check Global Risk status (likely CRITICAL)
2. Review RiskGate decision logs
3. Common causes:
   - Portfolio margin exceeded
   - Single-symbol risk limit hit
   - Correlation risk too high
4. Resolution:
   - Close positions to reduce risk
   - Adjust risk limits if appropriate
   - Wait for risk state to return to OK

### Unexpected Position Sizes

**Symptoms**: Positions larger or smaller than expected

**Actions**:
1. Check position sizing calculations in logs
2. Verify capital profile settings (MICRO/SMALL/MEDIUM/LARGE)
3. Review RL agent decisions (if RL position sizing enabled)
4. If incorrect:
   - Manually adjust positions via exchange UI
   - Review config for errors
   - File incident report

---

## Section 5: Configuration Management

### Capital Profiles

Defined in `config/trading_profile.yaml`:

| Profile | Max Position | Max Leverage | Use Case |
|---------|--------------|--------------|----------|
| **MICRO** | $100 | 2x | Initial GO-LIVE, testing |
| **SMALL** | $500 | 5x | Week 1-2, validation phase |
| **MEDIUM** | $2,000 | 10x | Month 1, stable performance |
| **LARGE** | $10,000 | 20x | Month 2+, proven track record |

**Profile Advancement Criteria**:
- MICRO → SMALL: 1 week, 90%+ uptime, zero ESS triggers
- SMALL → MEDIUM: 2 weeks, 80%+ win rate, max 2% daily drawdown
- MEDIUM → LARGE: 1 month, consistent profitability, risk metrics stable

### Risk Limits

Defined in `config/risk_v3.yaml`:

- **Max portfolio margin**: 80% of available capital
- **Max single-symbol exposure**: 30% of portfolio
- **Max correlation risk**: 0.7 (7 highly correlated positions)
- **ESS trigger thresholds**:
  - Capital loss: -10% total
  - Daily drawdown: -5%
  - Consecutive losses: 5 trades

### Updating Configuration

1. **Create PR with changes**
2. **Peer review** (at least one approval)
3. **Test on TESTNET** (minimum 3 trades)
4. **Run pre-flight check** (must pass)
5. **Deploy during low-volume period** (Asian session hours)
6. **Monitor for 1 hour post-change**

---

## Section 6: Monitoring & Alerts

### Key Metrics to Watch

| Metric | Target | Red Flag |
|--------|--------|----------|
| **System uptime** | 99.5%+ | < 95% |
| **ESS triggers** | 0 per day | > 1 per day |
| **Exchange failover rate** | < 1% | > 5% |
| **RiskGate block rate** | < 10% | > 30% |
| **Order execution success** | > 95% | < 90% |
| **TP/SL placement success** | 100% | < 95% |

### Dashboard Panels

1. **Global Risk Status** - Current risk level (OK/WARNING/CRITICAL)
2. **ESS Trigger Count** - Cumulative emergency stops
3. **RiskGate Decisions** - ALLOW vs BLOCK ratio
4. **Exchange Failover Events** - Fallback activations
5. **Stress Scenario Severity** - Current stress test results
6. **Position Overview** - Active positions, margin utilization
7. **Recent Trades** - Last 50 trades with P&L
8. **System Health** - Service status, connectivity

### Prometheus Alerts (Recommended)

```yaml
# Example alert rules (add to Prometheus)
- alert: ESSTriggered
  expr: ess_triggers_total > 0
  for: 1m
  labels:
    severity: critical
  annotations:
    summary: "Emergency Stop System activated"

- alert: HighRiskGateBlockRate
  expr: rate(risk_gate_decisions_total{decision="blocked"}[5m]) > 0.3
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "RiskGate blocking >30% of orders"

- alert: ExchangeFailover
  expr: increase(exchange_failover_events_total[10m]) > 0
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Exchange failover detected"
```

---

## Section 7: Troubleshooting

### Pre-flight Check Fails

**Problem**: `python scripts/preflight_check.py` returns exit code 1

**Solutions**:
1. Read output to identify failed check
2. Common failures:
   - **Health endpoints**: Backend service not running → Start service
   - **Risk system**: Global Risk CRITICAL → Close positions to reduce risk
   - **Observability**: Metrics endpoint down → Check Prometheus/metrics service
   - **Stress scenarios**: Not all registered → Check stress scenario config
3. Resolve issue and re-run pre-flight check

### GO-LIVE Activation Fails

**Problem**: `python scripts/go_live_activate.py` fails after preflight passes

**Solutions**:
1. Check `config/go_live.yaml`:
   - Ensure `activation_enabled: true`
   - Verify `require_risk_state: "OK"` matches current risk state
2. Check logs for specific error
3. Verify testnet history requirement met (if enabled)

### Orders Not Executing

**Problem**: Signals generated but no orders placed

**Solutions**:
1. Check `go_live.active` file exists → If not, run activation script
2. Check logs for `ORDER_SKIPPED` messages
3. Verify RiskGate not blocking all orders
4. Check exchange connectivity (API keys valid, network accessible)

### Positions Missing TP/SL

**Problem**: Positions opened without stop-loss or take-profit orders

**Solutions**:
1. Check logs for TP/SL placement errors
2. Verify Binance API permissions (need futures trading enabled)
3. Manually place TP/SL via exchange UI as temporary fix
4. File bug report with position details

---

## Section 8: Operational Checklist

### Daily Operations

- [ ] Start of day: Run pre-flight check
- [ ] Start of day: Review dashboard for overnight activity
- [ ] Start of day: Check ESS status (should be INACTIVE)
- [ ] During trading: Monitor dashboard every 30 minutes
- [ ] During trading: Review new positions within 5 minutes of opening
- [ ] End of day: Review daily P&L and performance metrics
- [ ] End of day: Verify all positions have TP/SL
- [ ] End of day: Update operational log

### Weekly Operations

- [ ] Review week's performance vs targets
- [ ] Check for recurring issues or patterns
- [ ] Assess capital profile advancement criteria
- [ ] Team debrief on incidents and learnings
- [ ] Update documentation with new procedures
- [ ] Backup configuration files and logs

### Before Major Changes

- [ ] Create PR with detailed description
- [ ] Get peer review (at least 1 approval)
- [ ] Test on TESTNET (minimum 3 successful trades)
- [ ] Run pre-flight check (must pass)
- [ ] Schedule deployment during low-volume period
- [ ] Notify team of change window
- [ ] Monitor continuously for 1 hour post-change

---

## Support & Escalation

### Contact Information

| Role | Contact | Availability |
|------|---------|--------------|
| **Senior Operator** | [operator@quantum-trader.com] | 24/7 |
| **Reliability Engineer** | [sre@quantum-trader.com] | Business hours |
| **On-call Engineer** | [oncall@quantum-trader.com] | 24/7 |

### Escalation Criteria

**Immediate escalation** (page on-call):
- ESS triggered with unknown cause
- Multiple exchange outages simultaneously
- Positions > 2x expected size
- Loss > 5% in single day

**Standard escalation** (email during business hours):
- Pre-flight check fails repeatedly
- RiskGate block rate > 30% for > 1 hour
- Configuration questions
- Feature requests

---

**Document Version**: 1.0  
**Last Updated**: December 4, 2025  
**Next Review**: January 4, 2026
