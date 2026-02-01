# P3.5 Decision Intelligence - Action Mapping Guide

**Purpose:** Translate P3.5 analytics into concrete operational actions.

---

## Overview

P3.5 provides real-time visibility into **why trades are not executing**. This guide maps each gate/reason to specific investigative and remediation actions.

---

## Gate → Action Mapping

### 1. `p33_permit_denied` (P3.3 Universe Gate)

**What it means:**
- P3.3 Universe Source is actively blocking the trade
- Could be: stale permit, leverage exceeded, risk threshold, action_hold, etc.

**Diagnosis:**
```bash
# Check P3.3 permit reasons
redis-cli HGETALL quantum:p33:symbol:BTCUSDT:permit
redis-cli HGETALL quantum:p33:symbol:ETHUSDT:permit

# Look for specific deny reasons
redis-cli HGET quantum:p33:symbol:BTCUSDT:permit reason
```

**Action Steps:**
1. **If reason = `stale`**: P3.3 stream is not receiving fresh permits
   - Check: `redis-cli XINFO STREAM quantum:stream:universe.permits | grep length`
   - Fix: Restart AI Agent 2.0 or Universe Source publisher

2. **If reason = `leverage_exceeded`**: Aggregate leverage too high
   - Check: `redis-cli HGET quantum:p33:aggregate:leverage current`
   - Fix: Close positions or increase leverage limit in P3.3 config

3. **If reason = `risk_threshold_breach`**: Portfolio risk too high
   - Check: `redis-cli HGET quantum:p33:risk:monitor current_risk`
   - Fix: Reduce position sizes or increase risk tolerance

4. **If reason = `action_hold`**: Manual trading pause
   - Check: `redis-cli HGET quantum:p33:global:control action`
   - Fix: Remove hold via Governor UI or `redis-cli HSET quantum:p33:global:control action ALLOW`

**Expected Resolution Time:**
- Stale permits: 1-5 minutes (restart publisher)
- Leverage/risk: 10-30 minutes (position adjustments)
- Action hold: Immediate (manual toggle)

---

### 2. `no_position` (Position Existence Check)

**What it means:**
- AI Engine/Harvest proposed an EXIT/CLOSE action
- Apply Layer checked: position does not exist
- Could be: race condition, position already closed, symbol mismatch, reconciliation lag

**Diagnosis:**
```bash
# Check actual positions
redis-cli HGETALL quantum:positions

# Check recent closes
redis-cli XREVRANGE quantum:stream:position.events - + COUNT 20 | grep CLOSED

# Check for symbol in original intent
redis-cli XREVRANGE quantum:stream:intents - + COUNT 50 | grep BTCUSDT
```

**Action Steps:**
1. **If frequent for same symbol**: Race condition between AI Engine and Apply Layer
   - Fix: Add deduplication in Intent Executor (check position before publishing intent)
   - Code location: `microservices/intent_executor/main.py`
   - Add: Position existence check before `XADD quantum:stream:intents`

2. **If random symbols**: Reconciliation lag (position closed but AI still thinks it's open)
   - Check: `redis-cli HGET quantum:reconciliation:last_run timestamp`
   - Fix: Increase reconciliation frequency (currently 60s → 30s?)

3. **If after manual closes**: Expected behavior (AI catches up in next cycle)
   - No action needed (self-healing)

**Expected Resolution Time:**
- Race condition fix: 1 hour (code + test + deploy)
- Reconciliation tuning: 30 minutes (config change)
- Manual close lag: 1-2 minutes (auto-resolves)

---

### 3. `not_in_allowlist` (Universe Whitelist)

**What it means:**
- Symbol is not in P3.3 allowed symbols list
- AI Engine proposed trade for symbol not in whitelist

**Diagnosis:**
```bash
# Check current allowlist
redis-cli SMEMBERS quantum:p33:allowlist

# Check which symbols are being proposed
bash scripts/p35_dashboard_queries.sh b "$(redis-cli SMEMBERS quantum:p33:allowlist)"
```

**Action Steps:**
1. **If allowlist too narrow**: Strategic decision to expand
   - Check: Which symbols are repeatedly blocked?
   - Decision: Add to allowlist if they have good signals
   - Update: `redis-cli SADD quantum:p33:allowlist SOLUSDT`

2. **If AI proposing wrong symbols**: Signal quality issue
   - Check: Why is AI Engine producing signals for non-allowed symbols?
   - Fix: Update AI Engine symbol filter (should only propose whitelisted symbols)
   - Code location: `microservices/ai_engine/symbol_filter.py`

**Expected Resolution Time:**
- Expand allowlist: Immediate (Redis command)
- Fix AI filter: 1 hour (code + test)

---

### 4. `leverage_ratio_exceeded` (Position-Level Leverage)

**What it means:**
- This specific trade would push symbol leverage above limit
- Different from P3.3 aggregate leverage (which is checked there)

**Diagnosis:**
```bash
# Check current symbol leverage
redis-cli HGET quantum:leverage:BTCUSDT current
redis-cli HGET quantum:leverage:BTCUSDT limit

# Check adaptive leverage settings
redis-cli HGETALL quantum:adaptive_leverage:BTCUSDT
```

**Action Steps:**
1. **If adaptive leverage is working**: Expected behavior (protecting from over-leverage)
   - No action needed (system working as designed)

2. **If static leverage limit too low**: Increase limit
   - Update: `redis-cli HSET quantum:leverage:BTCUSDT limit 15.0`
   - Or: Enable adaptive leverage for symbol

**Expected Resolution Time:**
- Limit adjustment: Immediate (Redis command)
- Enable adaptive: 5 minutes (config change)

---

### 5. `insufficient_confidence` (AI Confidence Check)

**What it means:**
- AI Engine confidence score below threshold (e.g., <0.60)
- Intent was published but Apply Layer rejected due to low confidence

**Diagnosis:**
```bash
# Check recent intent confidence scores
redis-cli XREVRANGE quantum:stream:intents - + COUNT 20 | grep confidence

# Check confidence threshold
redis-cli HGET quantum:config:apply confidence_threshold
```

**Action Steps:**
1. **If confidence consistently low**: Model performance degraded
   - Check: AI Engine model metrics
   - Fix: Trigger retraining or switch to fallback model

2. **If threshold too high**: Conservative setting blocking good trades
   - Current: 0.60 (example)
   - Consider: Lower to 0.50 if missing good opportunities
   - Update: `redis-cli HSET quantum:config:apply confidence_threshold 0.50`

**Expected Resolution Time:**
- Threshold adjustment: Immediate
- Model retraining: 2-6 hours

---

### 6. `action_hold` (Manual Trading Pause)

**What it means:**
- Global or symbol-specific trading pause is active
- Manually set via Governor UI or Redis command

**Diagnosis:**
```bash
# Check global hold
redis-cli HGET quantum:global:control action

# Check symbol-specific hold
redis-cli HGET quantum:symbol:BTCUSDT:control action
```

**Action Steps:**
1. **If intentional**: No action (wait for manual resume)

2. **If stale/forgotten**: Resume trading
   - Global: `redis-cli HSET quantum:global:control action ALLOW`
   - Symbol: `redis-cli HSET quantum:symbol:BTCUSDT:control action ALLOW`

**Expected Resolution Time:**
- Immediate (Redis command)

---

### 7. `reconciliation_mismatch` (Position State Inconsistency)

**What it means:**
- Discrepancy between internal state (Redis) and exchange state (Binance API)
- Apply Layer detected mismatch and blocked trade for safety

**Diagnosis:**
```bash
# Check reconciliation status
redis-cli HGETALL quantum:reconciliation:status

# Check last reconciliation errors
redis-cli XREVRANGE quantum:stream:reconciliation.errors - + COUNT 10
```

**Action Steps:**
1. **If frequent mismatches**: Reconciliation not running or failing
   - Check: Reconciliation service status
   - Fix: Restart reconciliation service

2. **If isolated**: Exchange API lag (normal during high volatility)
   - Wait: 1-2 minutes for next reconciliation cycle

**Expected Resolution Time:**
- Service restart: 2-5 minutes
- API lag: 1-2 minutes (auto-resolves)

---

## Decision Distribution Analysis

### Healthy System (Baseline)

**Expected ratios** (24/7 steady-state trading):
```
EXECUTE:  30-50%  (trades going through)
SKIP:     30-50%  (no position, waiting for setup)
BLOCKED:  10-20%  (gates protecting, expected)
ERROR:    <5%     (occasional transients)
UNKNOWN:  <5%     (legacy/incomplete data)
```

### Alert Conditions

| Condition | Threshold | Severity | Action |
|-----------|-----------|----------|--------|
| EXECUTE < 20% for 15+ min | <20% | HIGH | Check P3.3 permits + top gates |
| SKIP > 70% for 15+ min | >70% | MEDIUM | Check signal generation (AI Engine) |
| BLOCKED > 40% for 15+ min | >40% | HIGH | Check top gate (likely one gate dominating) |
| ERROR > 10% for 5+ min | >10% | CRITICAL | Check logs for exceptions |
| Single gate > 40% | >40% | MEDIUM | Investigate specific gate |
| Single gate > 60% | >60% | HIGH | Immediate investigation |

---

## Operational Playbook

### Morning Check (Daily)

```bash
# 1. Overall health
bash scripts/p35_dashboard_queries.sh a

# 2. Symbol-specific gates
bash scripts/p35_dashboard_queries.sh b

# 3. Check for drift
bash scripts/p35_dashboard_queries.sh d
```

### Alert Response (When gate explodes)

1. **Identify dominant gate** (>40% share):
   ```bash
   bash scripts/p35_dashboard_queries.sh c
   ```

2. **Look up action mapping** (see sections above)

3. **Execute diagnosis commands** (from relevant section)

4. **Apply fix** (immediate or scheduled)

5. **Verify resolution** (wait 5 minutes, re-run query A)

### Weekly Analysis

```bash
# Generate 1-hour window for trend analysis
redis-cli HGETALL quantum:p35:decision:counts:1h
redis-cli ZREVRANGE quantum:p35:reason:top:1h 0 50 WITHSCORES
```

Look for:
- Persistent gates (same reason in top 3 for 7+ days)
- New gates (reason appeared this week, not before)
- Increasing share (gate share growing week-over-week)

---

## P3.5.1 Preview: Automated Alerts

**Coming Soon** (30-minute implementation):

### Feature: Automatic Gate Explosion Detection

**Logic:**
- Every 60s: Check top reason in 5m window
- If `top_score / TOTAL > THRESHOLD` (e.g., 40%):
  - Publish alert event to `quantum:p35:alerts` stream

**Alert Event Schema:**
```json
{
  "window": "5m",
  "reason": "p33_permit_denied",
  "share": 0.62,
  "score": 1234,
  "total": 2000,
  "timestamp": 1769929500,
  "severity": "HIGH"
}
```

**Consumers:**
- Governor UI (display notification)
- AI Engine (adjust strategy if gate = insufficient_confidence)
- Discord/Telegram bot (operator alerts)

**Implementation Plan:**
1. Add `_check_alerts()` method to P3.5 service
2. Run every 60s (same as snapshot computation)
3. Use configurable thresholds:
   - 40% = MEDIUM alert
   - 60% = HIGH alert
   - 80% = CRITICAL alert
4. Rate-limit: Max 1 alert per (reason, window) per 5 minutes

**Files to Modify:**
- `microservices/decision_intelligence/main.py` (add alert logic)
- `etc/quantum/p35-decision-intelligence.env` (add threshold configs)

---

## Summary

P3.5 provides **visibility**. This guide provides **action mapping**.

**Key Workflow:**
1. Run dashboard query (identify gate)
2. Look up action mapping (this doc)
3. Execute diagnosis commands (verify root cause)
4. Apply fix (code, config, or manual)
5. Verify resolution (re-run dashboard query)

**Most Common Gates** (expected frequency):
1. `no_position` (30-40%) - Expected for EXIT intents when no position exists
2. `p33_permit_denied` (20-30%) - P3.3 doing its job (normal)
3. `not_in_allowlist` (5-10%) - Symbol filter working
4. `leverage_ratio_exceeded` (5-10%) - Adaptive leverage working
5. `insufficient_confidence` (<5%) - Model being conservative

If distribution differs significantly from above, investigate.

---

**Last Updated:** 2026-02-01  
**Version:** 1.0  
**Maintainer:** AI Assistant (Decision Intelligence Team)
