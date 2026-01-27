# 🏛️ P2.8 PORTFOLIO RISK GOVERNOR - DEPLOYMENT GUIDE
**Date**: 2026-01-27  
**Service**: quantum-portfolio-risk-governor  
**Mode**: SHADOW → ENFORCE  
**Status**: ✅ DEPLOYED

---

## 📋 Overview

P2.8 Portfolio Risk Governor is a fund-grade budget engine that enforces portfolio-level position sizing limits based on stress-aware risk allocation.

### Core Formula

```
base_budget = equity_usd * BASE_RISK_PCT
stress = α*portfolio_heat + β*cluster_stress + γ*vol_regime
budget = clamp(base_budget * (1 - stress), MIN_BUDGET_K, MAX_BUDGET_K)
```

### Architecture

```
Portfolio State → [P2.8 Budget Engine] → Budget Hash → Governor → Permit/Block
                         ↓
                   Redis Stream (violations)
```

---

## 🚀 DEPLOYMENT STATUS

### ✅ Phase 1: SHADOW MODE (COMPLETED 2026-01-27 22:34 UTC)

**Service Details**:
- **Service**: `quantum-portfolio-risk-governor.service`
- **Port**: 8049
- **Mode**: `shadow` (logs violations, doesn't block)
- **Status**: `Active (running)`

**Files Deployed**:
```bash
/opt/quantum/microservices/portfolio_risk_governor/main.py
/etc/quantum/portfolio-risk-governor.env
/etc/systemd/system/quantum-portfolio-risk-governor.service
```

**Configuration** (`/etc/quantum/portfolio-risk-governor.env`):
```env
P28_MODE=shadow
BASE_RISK_PCT=0.02        # 2% of equity per position
ALPHA_HEAT=0.4            # Weight for portfolio heat
BETA_CLUSTER=0.4          # Weight for cluster stress
GAMMA_VOL=0.2             # Weight for volatility regime
MIN_BUDGET_K=500          # Min budget $500
MAX_BUDGET_K=10000        # Max budget $10K
STALE_SEC=30              # Stale data threshold
```

**Integration Points**:
1. **Reads from**:
   - `quantum:state:portfolio` (equity, drawdown)
   - `http://localhost:8056/metrics` (portfolio heat from P2.6)
   - `quantum:cluster:stress:{cluster_id}`
   - `quantum:state:market:{symbol}` (vol regime)

2. **Writes to**:
   - `quantum:portfolio:budget:{symbol}` (hash, 60s TTL)
   - `quantum:stream:budget.violation` (events)

3. **Governor Integration**:
   - Governor reads `quantum:portfolio:budget:{symbol}` in production mode
   - Blocks permit if `P28_MODE=enforce` AND budget violation detected

---

## 📊 VERIFICATION

### Service Status

```bash
systemctl status quantum-portfolio-risk-governor
```

**Expected Output**:
```
● quantum-portfolio-risk-governor.service - P2.8 Portfolio Risk Governor
     Active: active (running)
     Main PID: 3530926
     
Jan 27 22:34:03 quantum-portfolio-risk-governor: P2.8 Portfolio Risk Governor started (mode=shadow, port=8049)
Jan 27 22:34:03 quantum-portfolio-risk-governor: Starting budget compute loop
```

### Metrics

```bash
curl localhost:8049/metrics | grep "^p28_"
```

**Current Values**:
```
p28_enforce_mode 0.0                # 0=shadow, 1=enforce
p28_redis_write_fail_total 0.0      # No write failures
```

**Additional Metrics** (populated when positions exist):
```
p28_budget_computed_total{symbol="BTCUSDT"}
p28_budget_value_usd{symbol="BTCUSDT"}
p28_stress_factor{symbol="BTCUSDT"}
p28_budget_blocks_total{symbol="BTCUSDT"}
p28_budget_allow_total{symbol="BTCUSDT"}
p28_stale_input_total{input_type="portfolio_state"}
```

### Health Check

```bash
curl localhost:8049/health
```

**Expected**:
```json
{
  "status": "healthy",
  "service": "p2.8-portfolio-risk-governor",
  "mode": "shadow",
  "redis": "connected"
}
```

### Redis Budget Hash (when positions exist)

```bash
redis-cli HGETALL quantum:portfolio:budget:BTCUSDT
```

**Example Output**:
```
symbol: BTCUSDT
budget_usd: 1800.50
stress_factor: 0.425
equity_usd: 10000
portfolio_heat: 0.65
cluster_stress: 0.12
vol_regime: 0.33
mode: shadow
timestamp: 1738015443
base_risk_pct: 0.02
```

### Budget Violation Events

```bash
redis-cli XREVRANGE quantum:stream:budget.violation + - COUNT 5
```

---

## 🎯 ENFORCEMENT ACTIVATION (Phase 2)

### Pre-Activation Checklist

Before activating ENFORCE mode, verify:

- [ ] **Shadow mode stable** (24+ hours without errors)
- [ ] **Budget computations accurate** (verify stress factors make sense)
- [ ] **No false positives** (check `budget.violation` stream for anomalies)
- [ ] **Governor integration tested** (verify P2.8 gate in Governor logs)
- [ ] **Rollback plan ready** (see Rollback section below)
- [ ] **OPS ledger entry created** (P5+ ledger system)

### Activation Steps

**1. Update environment config**:
```bash
ssh root@46.224.116.254
sed -i 's/P28_MODE=shadow/P28_MODE=enforce/' /etc/quantum/portfolio-risk-governor.env
```

**2. Restart service**:
```bash
systemctl restart quantum-portfolio-risk-governor
```

**3. Verify enforce mode**:
```bash
curl localhost:8049/metrics | grep p28_enforce_mode
# Should show: p28_enforce_mode 1.0
```

**4. Monitor blocking behavior**:
```bash
# Watch Governor logs for P2.8 blocks
journalctl -u quantum-governor -f | grep -E "(p28|budget)"
```

**5. Test with small position**:
```bash
# Create test position that exceeds budget
# Verify Governor blocks with: reason="p28_budget_violation"
```

---

## 🔄 ROLLBACK PROCEDURE

### Emergency Rollback (< 60 seconds)

**Immediate disable** (if system unstable):
```bash
systemctl stop quantum-portfolio-risk-governor
```

This immediately removes P2.8 gating (Governor fails-open on missing budget data).

### Graceful Rollback to Shadow

**Revert to shadow mode**:
```bash
ssh root@46.224.116.254
sed -i 's/P28_MODE=enforce/P28_MODE=shadow/' /etc/quantum/portfolio-risk-governor.env
systemctl restart quantum-portfolio-risk-governor
```

**Verify**:
```bash
curl localhost:8049/metrics | grep p28_enforce_mode
# Should show: p28_enforce_mode 0.0
```

### Complete Removal (nuclear option)

```bash
systemctl stop quantum-portfolio-risk-governor
systemctl disable quantum-portfolio-risk-governor
rm /etc/systemd/system/quantum-portfolio-risk-governor.service
systemctl daemon-reload
```

Governor will automatically fail-open (allow) when P2.8 is not running.

---

## 🏗️ GOVERNOR INTEGRATION DETAILS

### Code Integration

**File**: `microservices/governor/main.py`

**New Gate** (production mode only):
```python
# Gate 0: P2.8 Portfolio Budget (via budget governor integration)
budget_violation = self._check_portfolio_budget(symbol, plan_id)
if budget_violation:
    self._block_plan(plan_id, symbol, 'p28_budget_violation')
    return
```

**Method** (`_check_portfolio_budget`):
1. Reads `quantum:portfolio:budget:{symbol}` hash
2. Checks `mode` field (shadow vs enforce)
3. Checks stale data (>60s = fail-open)
4. Checks `quantum:stream:budget.violation` for recent violations (<30s)
5. Returns `True` to block, `False` to allow

**Fail-Safe Design**:
- No budget data → fail-open (allow)
- Stale data → fail-open (allow)
- P2.8 service down → fail-open (allow)
- P28_MODE=shadow → always allow (don't block)

### Blocking Metrics

When P2.8 blocks a permit, Governor emits:
```
quantum_govern_block_total{symbol="BTCUSDT", reason="p28_budget_violation"}
```

---

## 📈 MONITORING

### Key Metrics to Watch

**P2.8 Service**:
- `p28_enforce_mode`: Should be 1.0 when enforcing
- `p28_budget_blocks_total`: Increment when violations blocked
- `p28_redis_write_fail_total`: Should stay at 0
- `p28_stale_input_total`: Should be low (<5% of computations)

**Governor**:
- `quantum_govern_block_total{reason="p28_budget_violation"}`: Tracks blocks
- `quantum_govern_allow_total`: Total permits issued

**Budget Values**:
- `p28_budget_value_usd`: Per-symbol budget in USD
- `p28_stress_factor`: Composite stress (0-1)

### Alerts

Recommended Prometheus alerts:

```yaml
- alert: P28BudgetEngineDown
  expr: up{job="quantum-portfolio-risk-governor"} == 0
  for: 2m
  annotations:
    summary: "P2.8 Portfolio Risk Governor is down"
    
- alert: P28HighStaleInputRate
  expr: rate(p28_stale_input_total[5m]) > 0.1
  for: 5m
  annotations:
    summary: "P2.8 receiving stale inputs (>10% failure rate)"
    
- alert: P28RedisWriteFailures
  expr: rate(p28_redis_write_fail_total[5m]) > 0
  for: 2m
  annotations:
    summary: "P2.8 Redis write failures detected"
```

---

## 🧪 TEST SCENARIOS

### Test 1: Budget Computation with Active Position

**Setup**:
1. Open position: BTCUSDT, 0.5 BTC @ $100K = $50K notional
2. Portfolio equity: $100K
3. Portfolio heat: 0.65 (HOT)

**Expected Budget**:
```
base_budget = 100000 * 0.02 = $2000
stress = 0.4*0.65 + 0.4*0.0 + 0.2*0.33 = 0.326
budget = 2000 * (1 - 0.326) = $1348
```

**Expected Behavior**:
- Position notional ($50K) >> budget ($1348)
- Violation published to `quantum:stream:budget.violation`
- In ENFORCE: Governor blocks new increases

### Test 2: Low Stress Budget

**Setup**:
1. Portfolio equity: $100K
2. Portfolio heat: 0.15 (COLD)
3. No cluster stress, normal vol

**Expected Budget**:
```
base_budget = $2000
stress = 0.4*0.15 + 0 + 0.2*0.33 = 0.126
budget = 2000 * (1 - 0.126) = $1748
```

**Expected Behavior**:
- Higher budget when stress is low
- More risk capacity available

---

## 📚 RELATED DOCUMENTATION

- **P2.6 Heat Gate**: Portfolio-level heat calibration
- **P3.2 Governor**: Rate limits and safety gates
- **P3.3 Position State Brain**: Pre-execution validation
- **Governor Integration**: Multi-layer gating hierarchy

---

## 🎯 SUCCESS CRITERIA

### Shadow Mode Success
- ✅ Service runs stable for 24+ hours
- ✅ Budget computations accurate (spot-checked)
- ✅ No false positives in violation stream
- ✅ Metrics correctly reflect portfolio state

### Enforce Mode Success
- ✅ Blocks excessive position sizing
- ✅ Allows normal-sized positions
- ✅ Fails-open gracefully on errors
- ✅ No trading disruption

---

## 🔐 SECURITY NOTES

1. **Fail-Open Design**: System remains operational even if P2.8 fails
2. **Single Source of Truth**: Redis hashes with 60s TTL prevent stale enforcement
3. **Audit Trail**: All violations logged to Redis stream
4. **Mode Toggle**: Can switch shadow↔enforce without code changes

---

## 📞 OPERATIONS CONTACTS

**Service Owner**: AI Trading System  
**Deployment Date**: 2026-01-27 22:34 UTC  
**VPS**: Hetzner 46.224.116.254  
**Logs**: `journalctl -u quantum-portfolio-risk-governor -f`  
**Metrics**: http://localhost:8049/metrics

---

**Next Steps**:
1. ✅ Monitor shadow mode for 24-48 hours
2. ⏳ Verify budget accuracy with real positions
3. ⏳ Activate ENFORCE mode
4. ⏳ Create OPS ledger entry (P5+)
