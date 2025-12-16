# Prompt 10 Pre-Flight Checklist

**EPIC-PREFLIGHT-001** | Operational Readiness for Real Trading  
**Target**: GO-LIVE with real capital  
**Risk Profile**: MICRO → LOW (gradual progression)

---

## Pre-Flight: Before Enabling REAL Trading

### System Health ✅
- [ ] **Pre-flight script passes**: `python scripts/preflight_check.py` exits 0
- [ ] **All services healthy**: `/health/ready` returns HTTP 200 for all microservices
- [ ] **Metrics endpoint active**: `/metrics` returns Prometheus data with key metrics
- [ ] **Database connected**: PostgreSQL and Redis responding to health checks
- [ ] **No startup errors**: Check logs for CRITICAL/ERROR entries in last 24h

### Risk System Status ✅
- [ ] **Global Risk v3 status**: Not CRITICAL (INFO or WARNING acceptable)
- [ ] **ESS inactive**: Emergency Stop System shows `is_active = False`
- [ ] **RiskGate v3 operational**: Risk gate initialized and enforcing decisions
- [ ] **Capital profiles loaded**: All profiles (micro, low, medium, high, aggressive) defined
- [ ] **Account mappings valid**: Each account mapped to correct capital profile

### Exchange Connectivity ✅
- [ ] **Primary exchange healthy**: Binance (or designated primary) health check passes
- [ ] **Failover chain configured**: Backup exchanges (Bybit, OKX, etc.) defined
- [ ] **API credentials valid**: Test API keys work on TESTNET (not expired)
- [ ] **Rate limits understood**: Know exchange API rate limits for production
- [ ] **Websocket feeds stable**: Market data streams connected and updating

### Accounts & Configuration ✅
- [ ] **TESTNET validation complete**: At least 3 test trades executed successfully on TESTNET
- [ ] **Account configs reviewed**: Verify exchange, capital profile, strategy whitelist per account
- [ ] **MICRO profile enforced**: All real accounts start with MICRO profile (max 1.5x leverage, 0.5% single-trade risk)
- [ ] **API keys secured**: Production API keys stored in secrets manager (not in code)
- [ ] **IP whitelisting**: Exchange API access restricted to known IPs (if applicable)

### Observability & Monitoring ✅
- [ ] **Grafana dashboard imported**: Risk & Resilience Dashboard available at http://grafana:3000
- [ ] **Prometheus scraping**: Metrics endpoint scraped every 15s, data visible in Grafana
- [ ] **Log aggregation active**: Logs flowing to Loki/ELK for search and analysis
- [ ] **Alert rules defined**: Prometheus alerts configured for ESS triggers, failover spikes, high block rate
- [ ] **On-call setup**: Team notified for CRITICAL alerts (ESS trigger, exchange outage)

### Documentation & Runbooks ✅
- [ ] **Build Constitution Audit reviewed**: `docs/BUILD_CONSTITUTION_AUDIT.md` confirms system alignment
- [ ] **Runbooks available**: Incident response procedures for ESS trigger, exchange outage, model failure
- [ ] **Change log updated**: Recent changes documented (last 7 days)
- [ ] **Team trained**: All operators familiar with dashboard, alert procedures, manual ESS reset

---

## GO-LIVE Day: Enabling Real Trading

### Pre-Launch (T-1 hour)
- [ ] **Team assembled**: All operators available for first hour of trading
- [ ] **Communication channel open**: Slack/Teams channel active for real-time updates
- [ ] **Dashboards displayed**: Risk & Resilience Dashboard + Exchange Health visible
- [ ] **Alerts tested**: Send test alert to verify notification delivery
- [ ] **Backup plan ready**: Know how to trigger manual ESS if needed

### Launch Sequence (T-0)
- [ ] **Switch accounts to REAL**: Update account configs from TESTNET → PRODUCTION
- [ ] **Verify MICRO profile active**: Confirm max leverage 1.5x, max single-trade risk 0.5%
- [ ] **Start services**: Restart backend services with production configs
- [ ] **Confirm health checks pass**: All services green after restart
- [ ] **Enable trading**: Set `allow_new_trades = True` in PolicyStore

### First Hour Monitoring (T+0 to T+60m)
- [ ] **Watch RiskGate decisions**: Expect mostly ALLOW with <30% BLOCK rate
- [ ] **Monitor ESS status**: Should remain INACTIVE (if triggers, investigate immediately)
- [ ] **Track exchange failovers**: Should be 0 under normal conditions
- [ ] **Check order execution latency**: P95 < 500ms, P99 < 1000ms
- [ ] **Verify PnL calculation**: Compare internal PnL with exchange API balance
- [ ] **Log review**: No CRITICAL or ERROR logs (WARNING acceptable)

### First Day Completion (T+6h)
- [ ] **PnL summary**: Calculate total PnL across all accounts
- [ ] **Risk review**: Check max drawdown, max leverage used, ESS near-misses
- [ ] **Block reasons**: Top 3 RiskGate block reasons (should be sensible limits, not errors)
- [ ] **Exchange performance**: Any failover events? Average order latency?
- [ ] **Incident log**: Document any issues encountered and resolutions

---

## First Week: Gradual Ramp-Up

### Daily Review (Days 1-7)
- [ ] **Daily PnL per account**: Track profitability, identify losing accounts
- [ ] **Daily drawdown check**: Ensure DD < 5% per account (MICRO profile limit)
- [ ] **Daily risk review**: Global Risk level, ESS activations, RiskGate stats
- [ ] **Daily operational log**: Note any manual interventions, config changes, incidents

### Profile Advancement (Conservative)
- [ ] **Day 1-3**: Keep ALL accounts at MICRO profile (no exceptions)
- [ ] **Day 4-7**: Promote ONLY profitable accounts to LOW profile (max 3x leverage, 1% single-trade risk)
- [ ] **Day 8+**: Consider MEDIUM profile for consistently profitable accounts (require 7+ days positive PnL)
- [ ] **NO automatic promotion**: All profile changes require manual review and approval

### Monitoring Metrics (Week 1 Targets)
- [ ] **ESS triggers**: 0 (any trigger requires investigation)
- [ ] **Exchange failovers**: < 5 per day (spike indicates infrastructure issue)
- [ ] **RiskGate block rate**: 20-40% (too low = too aggressive, too high = too conservative)
- [ ] **Order execution success rate**: > 95% (failed orders should be retried or logged)
- [ ] **System uptime**: > 99.5% (< 30 minutes downtime per week)

### Configuration Stability
- [ ] **No emergency hotfixes**: All changes via git PR and review
- [ ] **Config as code**: Capital profiles, accounts, exchanges defined in version control
- [ ] **Pre-flight before changes**: Run `python scripts/preflight_check.py` after any config change
- [ ] **Rollback plan**: Know how to revert to previous config version if needed

---

## Ongoing: Change Management

### Before ANY Configuration Change
1. [ ] **Create PR**: All config changes in git branch with clear description
2. [ ] **Peer review**: At least one team member reviews change
3. [ ] **Test on TESTNET**: Validate change in test environment first
4. [ ] **Run pre-flight check**: `python scripts/preflight_check.py` must pass
5. [ ] **Deploy during low-volume period**: Avoid major market events or high volatility
6. [ ] **Monitor for 1 hour post-change**: Watch dashboards for unexpected behavior

### After Major Changes (Risk System, Execution, Profiles)
1. [ ] **Re-run stress scenarios**: `python scripts/run_stress_scenarios.py` (when implemented)
2. [ ] **Review risk decisions**: Check RiskGate block reasons match expectations
3. [ ] **Validate PnL impact**: Ensure changes don't degrade profitability
4. [ ] **Update documentation**: Reflect changes in runbooks, audit docs

---

## Emergency Procedures

### If ESS Triggers Unexpectedly
1. **DO NOT DISABLE ESS** - It triggered for a reason
2. Check Global Risk v3 status (likely CRITICAL)
3. Review recent trades for anomalies (excessive losses, leverage spikes)
4. Check exchange connectivity (failover events, API errors)
5. Review logs for ERROR entries in last 15 minutes
6. Only reset ESS after root cause identified and resolved
7. Document incident in operational log

### If Exchange Outage Detected
1. Verify failover activated (check `exchange_failover_events_total` metric)
2. Confirm orders routing to backup exchange (Bybit, OKX)
3. Monitor order execution success rate (should recover within 1 minute)
4. If all exchanges down: Manually trigger ESS to halt trading
5. Notify team via alert channel
6. Document outage duration and impact

### If RiskGate Blocks Everything (100% block rate)
1. Check ESS status (likely active)
2. Check Global Risk v3 level (likely CRITICAL)
3. Review capital profile configs (possible misconfiguration)
4. Check logs for "risk_gate_not_initialized" errors
5. Verify account → profile mappings are valid
6. If config issue: fix, redeploy, re-run pre-flight check

---

## Success Criteria (Week 1)

### Minimum Acceptable Performance
- ✅ **System uptime**: > 99% (< 1 hour downtime)
- ✅ **ESS triggers**: 0-1 (any more requires investigation)
- ✅ **Exchange failovers**: < 35 (< 5/day average)
- ✅ **Order execution success rate**: > 90%
- ✅ **RiskGate false positive rate**: < 50% (blocks are valid risk decisions)

### Target Performance
- 🎯 **System uptime**: 99.9% (< 10 minutes downtime)
- 🎯 **ESS triggers**: 0
- 🎯 **Exchange failovers**: < 7 (< 1/day average)
- 🎯 **Order execution success rate**: > 98%
- 🎯 **Average order latency**: P95 < 300ms

### Red Flags (Immediate Investigation Required)
- 🚨 **ESS triggers > 2 per day**
- 🚨 **Global Risk CRITICAL for > 1 hour**
- 🚨 **Exchange failover rate > 10/day**
- 🚨 **RiskGate block rate > 70%** (too conservative or systemic issue)
- 🚨 **Order execution success rate < 80%**
- 🚨 **Drawdown > 10% on any account** (even with MICRO profile)

---

## Post-Week 1: Production Stabilization

### Week 2-4 Goals
- [ ] Validate all stress scenarios pass with real data
- [ ] Tune Prometheus alert thresholds based on production baselines
- [ ] Implement single-trade risk calculation with real equity
- [ ] Integrate daily/weekly loss limits into RiskGate
- [ ] Document operational patterns (normal vs anomalous behavior)

### Month 2+ Goals
- [ ] Gradually promote accounts to MEDIUM/HIGH profiles (based on consistent profitability)
- [ ] Implement automated stress testing in CI/CD pipeline
- [ ] Add multi-cluster failover (geographic redundancy)
- [ ] Historical analysis of risk decisions vs PnL outcomes
- [ ] Capacity planning for increased order volume

---

**Checklist Version**: 1.0  
**Last Updated**: December 4, 2025  
**Owner**: Senior System Reliability + QA Engineer  
**Review Frequency**: Before each major release
