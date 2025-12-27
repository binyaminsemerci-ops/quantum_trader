# PROMPT 10 GO-LIVE PRODUCTION PLAYBOOK

**EPIC-P10: Prompt 10 GO-LIVE Program**  
**Date:** December 4, 2025  
**Version:** 1.0.0  
**Status:** Production Ready (Staged Rollout)

---

## 1. Capital Profiles Overview

Quantum Trader v2.0 uses **4 capital profiles** with explicit risk limits:

| Profile | Daily DD | Weekly DD | Trade Risk | Max Positions | Leverage | Use Case |
|---------|----------|-----------|------------|---------------|----------|----------|
| **Micro** | -0.5% | -2.0% | 0.2% | 2 | 1x | Testnet graduation, ultra-safe |
| **Low** | -1.0% | -3.5% | 0.5% | 3 | 2x | Conservative growth |
| **Normal** | -2.0% | -7.0% | 1.0% | 5 | 3x | Standard operations |
| **Aggressive** | -3.5% | -12.0% | 2.0% | 8 | 5x | Experienced accounts only |

**Default:** All new accounts start at **Micro** profile.

---

## 2. Current Production Setup

### Active Profiles (Private Multi-Account)

```bash
# Account 1: Main Binance (Normal profile)
export QT_ACCOUNT_MAIN_BINANCE_CAPITAL_PROFILE=normal
export QT_ACCOUNT_MAIN_BINANCE_EXCHANGE=binance
export QT_ACCOUNT_MAIN_BINANCE_API_KEY=xxx
export QT_ACCOUNT_MAIN_BINANCE_API_SECRET=yyy

# Account 2: Friend 1 Firi (Micro profile - testing)
export QT_ACCOUNT_FRIEND1_FIRI_CAPITAL_PROFILE=micro
export QT_ACCOUNT_FRIEND1_FIRI_EXCHANGE=firi
export QT_ACCOUNT_FRIEND1_FIRI_API_KEY=xxx
export QT_ACCOUNT_FRIEND1_FIRI_API_SECRET=yyy
export QT_ACCOUNT_FRIEND1_FIRI_CLIENT_ID=zzz
```

### Profile Assignment Strategy

- **Main account:** Normal profile (proven track record)
- **New accounts:** Start at Micro (4-week evaluation period)
- **Friend accounts:** Micro → Low after 4 weeks no breach
- **Aggressive:** Reserved for expert users after 8+ weeks at Normal

---

## 3. Pre-Launch Checklist

**Before enabling real trading, verify:**

### ✅ Infrastructure Health

- [ ] All exchanges reachable (ping health endpoints)
- [ ] Multi-exchange failover tested (EPIC-EXCH-FAIL-001)
- [ ] Account routing verified (EPIC-MT-ACCOUNTS-001)
- [ ] API keys valid and funded
- [ ] Testnet→production transition completed

### ✅ Risk Systems

- [ ] Global Risk v3 operational
- [ ] Capital profiles configured per account
- [ ] Strategy whitelist/blacklist defined
- [ ] Daily/weekly DD limits understood
- [ ] Emergency shutdown procedures tested

### ✅ Observability

- [ ] Grafana dashboards configured
- [ ] Prometheus metrics flowing
- [ ] Log aggregation working (structured logs)
- [ ] Alert channels tested (email, Slack)
- [ ] PnL tracking per account

### ✅ Backtests & Validation

- [ ] All strategies backtested on historical data
- [ ] Walk-forward validation completed
- [ ] Sharpe ratio > 0.8 for Normal profile strategies
- [ ] Win rate > 48% for production strategies
- [ ] Slippage/commission assumptions realistic

### ✅ Strategy Configuration

- [ ] Strategies assigned to correct accounts
- [ ] Leverage settings match profile limits
- [ ] Position sizing calibrated to risk limits
- [ ] Stop-loss/take-profit rules defined

---

## 4. Daily Routine

### Morning Check (Before Market Open)

**Time:** 30 minutes before main trading session

1. **Review Overnight Activity**
   ```bash
   # Check if any trades executed overnight
   python scripts/check_overnight_trades.py
   ```

2. **Check System Health**
   - [ ] Backend service running (health endpoint)
   - [ ] All exchanges connected (no failover active)
   - [ ] Global Risk status = NORMAL (not CAUTION/EMERGENCY)
   - [ ] No unread alerts in logs

3. **Review Dashboards**
   - [ ] Daily PnL per account (vs profile limits)
   - [ ] Open position count per account
   - [ ] Strategy performance (win rate, PnL)
   - [ ] Exchange latency metrics

4. **Confirm Profile Compliance**
   ```python
   # Run profile status check
   from backend.reports.review_jobs import generate_daily_risk_report
   report = await generate_daily_risk_report()
   print(report["summary"])
   ```

### Evening Review (After Market Close)

**Time:** 30 minutes after main trading session

1. **Daily PnL Summary**
   - [ ] Total PnL per account
   - [ ] Compare vs daily DD limits
   - [ ] Flag any accounts approaching limits

2. **Position Review**
   - [ ] All positions intentional (no leaks)
   - [ ] Stop-losses placed correctly
   - [ ] Position sizes within limits

3. **Log Review**
   - [ ] Check for ERROR/WARNING logs
   - [ ] Review any StrategyNotAllowedError incidents
   - [ ] Review any ProfileLimitViolationError incidents
   - [ ] Verify exchange failover events (if any)

4. **Update Daily Notes**
   - Notable trades (big wins/losses)
   - System issues encountered
   - Manual interventions taken

---

## 5. Weekly Routine

### Weekly Review (Sunday Evening)

**Time:** 1-2 hours for comprehensive review

1. **Generate Weekly Report**
   ```python
   from backend.reports.review_jobs import generate_weekly_risk_report
   report = await generate_weekly_risk_report()
   ```

2. **Performance Analysis**
   - [ ] Weekly PnL per account
   - [ ] Compare vs weekly DD limits
   - [ ] Calculate Sharpe ratio per strategy
   - [ ] Review win rate trends
   - [ ] Analyze losing trades (root cause)

3. **Profile Promotion/Downgrade Decisions**
   
   **Check Promotion Eligibility:**
   - Micro → Low: No breach 4 weeks, 20+ trades, win rate >45%, Sharpe >0.5
   - Low → Normal: No breach 6 weeks, 50+ trades, win rate >48%, Sharpe >0.8
   - Normal → Aggressive: No breach 8 weeks, 100+ trades, win rate >50%, Sharpe >1.0
   
   **Check Downgrade Triggers:**
   - Weekly DD limit breached → Downgrade one level
   - 2+ consecutive weeks negative → Consider downgrade
   - Sharpe ratio < 0.3 → Downgrade one level

4. **Strategy Review**
   - [ ] Disable underperforming strategies
   - [ ] Update whitelist/blacklist per profile
   - [ ] Adjust position sizing if needed

5. **Risk Limit Review**
   - [ ] Are current limits appropriate?
   - [ ] Any limit breaches this week?
   - [ ] Should limits be tightened/loosened?

6. **Documentation Update**
   - Update weekly notes (Markdown file)
   - Log any profile changes
   - Record major events/incidents

---

## 6. Incident Response & Recovery

### Scenario 1: Global Risk = EMERGENCY

**Trigger:** `GlobalRiskStatus.EMERGENCY` detected

**Immediate Actions:**
1. ⛔ **STOP all new orders** (kill switch activated automatically)
2. 📊 **Review open positions** (dashboard)
3. 🚨 **Assess situation:**
   - Market crash? (check BTC price)
   - Exchange outage? (check failover status)
   - Bug in strategy? (check logs)
4. 🔧 **Manual intervention:**
   - Close risky positions if market crash
   - Wait for exchange recovery if outage
   - Disable buggy strategy if code issue
5. ✅ **Verify system recovery** before re-enabling

**Post-Incident:**
- [ ] Document incident in incident log
- [ ] Review what triggered EMERGENCY
- [ ] Update risk parameters if needed
- [ ] Consider downgrading affected accounts

---

### Scenario 2: Daily DD Breach

**Trigger:** Account exceeds `max_daily_loss_pct`

**Immediate Actions:**
1. ⛔ **Stop trading on affected account** (profile guard blocks new orders)
2. 📊 **Review losing trades:**
   - Which strategies caused losses?
   - Was it market conditions or strategy bug?
3. 🔧 **Root cause analysis:**
   - Backtest strategy on today's data
   - Check for execution issues (slippage, latency)
   - Review logs for anomalies
4. 🛡️ **Risk action:**
   - Close remaining positions if loss accelerating
   - OR hold if DD is temporary market move
5. 📝 **Document:**
   - What happened
   - Why it happened
   - Prevention measures

**Next Day:**
- [ ] Account remains locked until manual review
- [ ] Decide: Continue with same profile OR downgrade
- [ ] If downgrade → update account config
- [ ] Re-enable trading after confirmation

---

### Scenario 3: Weekly DD Breach

**Trigger:** Account exceeds `max_weekly_loss_pct`

**Immediate Actions:**
1. ⛔ **Stop trading on affected account**
2. 📊 **Comprehensive review:**
   - Review all trades this week
   - Calculate per-strategy PnL breakdown
   - Identify root cause (strategy, market, execution)
3. 🔻 **Mandatory downgrade:**
   - Normal → Low
   - Low → Micro
   - Micro → Pause trading for 2 weeks
4. 📝 **Document incident:**
   - Full post-mortem report
   - Strategy adjustments needed
   - Profile change rationale

**Recovery Plan:**
- [ ] 2-week cooling-off period (no live trading)
- [ ] Re-backtest all strategies
- [ ] Paper trade for 1 week at new profile
- [ ] Resume live trading with reduced risk

---

### Scenario 4: Exchange Failover (Multiple Triggers)

**Trigger:** Exchange failover triggered 3+ times in 1 hour

**Immediate Actions:**
1. 🔍 **Investigate primary exchange:**
   - Check exchange status page
   - Review API error logs
   - Test manual API call
2. 🚨 **Assess impact:**
   - Are orders executing on failover exchange?
   - Any stuck orders or positions?
3. 🔧 **Manual intervention:**
   - If primary exchange down: Keep using failover
   - If primary exchange degraded: Monitor closely
   - If both exchanges problematic: Stop trading
4. 📝 **Log event:**
   - Which exchange failed
   - How long failover active
   - Any execution issues

**Post-Recovery:**
- [ ] Test primary exchange thoroughly
- [ ] Review failover performance
- [ ] Update failover config if needed

---

### Scenario 5: Strategy Blocked (StrategyNotAllowedError)

**Trigger:** Strategy blocked by profile whitelist/blacklist

**Expected Behavior:**
- Signal ignored (logged)
- No order placed
- Profile guard working as designed

**Review Actions:**
1. ✅ **Verify it's intentional:**
   - Check strategy_profile_policy.py
   - Is strategy supposed to be blocked?
2. 🔧 **If unintentional:**
   - Update whitelist/blacklist
   - Restart backend service
3. 📝 **If intentional:**
   - No action needed (working as designed)

---

## 7. Safe Rollout Ladder

### Phase 1: Testnet Only (Week 1)

**Goal:** Validate system without real capital

- [ ] Run all strategies on testnet
- [ ] Verify capital profiles enforced
- [ ] Test exchange routing + failover
- [ ] Collect 1 week of performance data
- [ ] Review logs daily for errors

**Success Criteria:**
- ✅ No critical errors
- ✅ All orders executed correctly
- ✅ Profile limits respected
- ✅ Sharpe ratio > 0.5 on testnet

---

### Phase 2: Micro Profile - Single Account (Weeks 2-5)

**Goal:** Prove system with minimal real capital

- [ ] Enable ONE account at Micro profile
- [ ] Start with $500-$1000 capital
- [ ] Run 2-3 conservative strategies only
- [ ] Daily monitoring (manual)
- [ ] Weekly review + adjustments

**Success Criteria (4 weeks):**
- ✅ No daily/weekly DD breaches
- ✅ At least 20 trades executed
- ✅ Win rate > 45%
- ✅ Sharpe ratio > 0.5
- ✅ No critical system errors

**If successful → Promote to Low profile**

---

### Phase 3: Add More Accounts (Weeks 6-10)

**Goal:** Scale to multi-account setup

- [ ] Enable 2-3 additional accounts (Micro profile)
- [ ] Main account promoted to Low profile
- [ ] Add 1-2 more strategies
- [ ] Implement automated daily reports
- [ ] Weekly review + profile adjustments

**Success Criteria (4 weeks):**
- ✅ All accounts positive or neutral PnL
- ✅ No weekly DD breaches
- ✅ System handles multi-account load
- ✅ Observability working well

**If successful → Promote main account to Normal**

---

### Phase 4: Normal Operations (Week 11+)

**Goal:** Full production with multiple profiles

- [ ] Main account at Normal profile
- [ ] 2-3 friend accounts at Low profile
- [ ] 1-2 test accounts at Micro profile
- [ ] Full strategy suite enabled
- [ ] Automated daily/weekly reports
- [ ] Consider Aggressive profile for main account (after 8+ weeks at Normal)

**Ongoing:**
- Weekly reviews (Sunday)
- Profile promotions/downgrades as earned
- Strategy tuning based on performance
- System improvements (observability, risk, execution)

---

### Phase 5: Aggressive Profile (Optional, Week 19+)

**Goal:** Maximum performance for proven system

**Prerequisites:**
- ✅ 8+ weeks at Normal profile with no weekly DD breach
- ✅ 100+ trades executed
- ✅ Win rate > 50%
- ✅ Sharpe ratio > 1.0
- ✅ Extensive backtesting completed
- ✅ Deep understanding of risk management

**Enable Aggressive Profile:**
- [ ] Promote main account to Aggressive
- [ ] Monitor VERY closely (daily)
- [ ] Be ready to downgrade quickly if needed
- [ ] Consider this as "performance mode" (not default)

---

## 8. Emergency Procedures

### Emergency Shutdown

**When to use:**
- Global Risk = EMERGENCY
- Exchange major outage
- Critical bug discovered
- Market black swan event

**Command:**
```bash
# Stop backend service
pkill -f "uvicorn backend.main:app"

# OR use kill switch API
curl -X POST http://localhost:8000/admin/kill-switch
```

**Post-Shutdown:**
1. Close all open positions manually (exchange web UI)
2. Investigate root cause
3. Fix issue
4. Test on testnet
5. Resume trading only when safe

---

### Emergency Position Close

**When to use:**
- Position stuck due to bug
- Need to close position immediately
- Strategy misbehaving

**Command:**
```python
# Use emergency close script
python scripts/emergency_close_position.py --account main_binance --symbol BTC/USDT
```

---

### Emergency Profile Downgrade

**When to use:**
- Weekly DD breach
- Multiple daily breaches in same week
- System behaving erratically

**Command:**
```python
from backend.policies.account_config import set_capital_profile_for_account

# Downgrade account
set_capital_profile_for_account("main_binance", "low")
```

---

## 9. Configuration Files

### Capital Profiles
- **File:** `backend/policies/capital_profiles.py`
- **Modify:** Only during weekly review
- **Restart required:** Yes (reload config)

### Strategy Whitelist/Blacklist
- **File:** `backend/policies/strategy_profile_policy.py`
- **Modify:** When adding/removing strategies
- **Restart required:** Yes

### Account Configs
- **File:** `.env` (environment variables)
- **Modify:** When adding/changing accounts
- **Restart required:** Yes

### Account→Profile Mapping
- **Runtime:** `set_capital_profile_for_account()`
- **Persistent:** Update `.env` for permanence

---

## 10. Monitoring & Alerts

### Critical Alerts (Immediate Action)

- 🚨 **Global Risk = EMERGENCY** → Stop trading, investigate
- 🚨 **Daily DD breach** → Lock account, review
- 🚨 **Weekly DD breach** → Downgrade profile
- 🚨 **Exchange failover (3+ times)** → Investigate exchange health
- 🚨 **Backend service down** → Restart service

### Warning Alerts (Review Within 1 Hour)

- ⚠️ **Approaching daily DD limit (>80%)** → Monitor closely
- ⚠️ **Strategy blocked (StrategyNotAllowedError)** → Verify intentional
- ⚠️ **Exchange API errors (5+ in 10 min)** → Check exchange status
- ⚠️ **Position count at max** → Review open positions

### Info Alerts (Review Daily)

- ℹ️ **Daily PnL report** → Review in evening routine
- ℹ️ **Exchange failover resolved** → Confirm primary exchange back
- ℹ️ **Profile limit check passed** → Normal operation

---

## 11. Success Metrics

### System Health
- ✅ Uptime > 99.5%
- ✅ All exchanges reachable
- ✅ No critical errors in logs
- ✅ Profile guard working correctly

### Performance
- ✅ Sharpe ratio > 0.8 (Normal profile)
- ✅ Win rate > 48% (Normal profile)
- ✅ Monthly PnL positive
- ✅ Max DD within profile limits

### Risk Management
- ✅ Zero daily DD breaches (goal)
- ✅ Zero weekly DD breaches (goal)
- ✅ No emergency shutdowns due to bugs
- ✅ All trades within position limits

### Progression
- ✅ Accounts promoted based on merit
- ✅ No forced downgrades due to breaches
- ✅ Steady capital growth (not exponential spikes)

---

## 12. Contacts & Resources

### Emergency Contacts
- **System Admin:** [Your contact]
- **Exchange Support:** Keep exchange support links handy
- **Risk Manager:** [If team setup]

### Documentation
- Capital profiles: `backend/policies/capital_profiles.py`
- Strategy policies: `backend/policies/strategy_profile_policy.py`
- Profile guard: `backend/services/risk/profile_guard.py`
- Review jobs: `backend/reports/review_jobs.py`
- Multi-account: `EPIC_MT_ACCOUNTS_001_COMPLETION.md`
- Exchange routing: `EPIC_EXCH_ROUTING_001_COMPLETION.md`
- Failover: `EPIC_EXCH_FAIL_001_COMPLETION.md`

### Dashboards
- **Grafana:** http://localhost:3000/d/quantum-trader
- **Backend Health:** http://localhost:8000/health
- **Prometheus:** http://localhost:9090

---

## 13. Sign-Off

**Production Readiness:** ✅ APPROVED (Staged Rollout)

**Constraints:**
- Start with Micro profile only (Phase 2)
- Daily monitoring mandatory (Phases 2-3)
- Weekly reviews mandatory (all phases)
- No Aggressive profile until Week 19+

**Next Milestones:**
1. Week 1: Testnet validation
2. Week 2: Enable Micro profile (1 account)
3. Week 6: Add more accounts, promote to Low
4. Week 11: Promote to Normal profile
5. Week 19: Consider Aggressive profile (optional)

**Approved By:** Senior Quant + Systems Engineer  
**Date:** December 4, 2025  
**Version:** 1.0.0

---

**⚠️ REMEMBER:**
- Start small (Micro profile)
- Monitor daily
- Review weekly
- Promote gradually
- Downgrade quickly if needed
- Document everything

**Safe trading! 🚀**
