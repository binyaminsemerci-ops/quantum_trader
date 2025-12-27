# PORTFOLIO BALANCER AI (PBA) — Quick Reference

**Status:** ✅ DEPLOYED  
**Version:** 1.0  
**Date:** November 23, 2025

---

## 🎯 WHAT IS PBA?

**Portfolio Balancer AI** manages your ENTIRE portfolio as a unified system:
- Controls total exposure, diversification, and risk
- Enforces portfolio-level constraints
- Filters trades based on global portfolio health
- Advisory system for execution layer

**Key Principle:** PBA does NOT manage single trades — it manages the WHOLE PORTFOLIO.

---

## 🚀 QUICK START

### Import

```python
from backend.services.portfolio_balancer import (
    PortfolioBalancerAI,
    Position,
    CandidateTrade,
    PortfolioConstraints
)
```

### Initialize

```python
balancer = PortfolioBalancerAI()
```

### Run Analysis

```python
output = balancer.analyze_portfolio(
    positions=open_positions,           # List[Position]
    candidates=candidate_signals,       # List[CandidateTrade]
    total_equity=10000.0,
    used_margin=1500.0,
    free_margin=8500.0
)
```

### Use Output

```python
# Check risk mode
if output.risk_mode == "SAFE":
    logger.warning("🛡️ SAFE mode - defensive posture")

# Execute approved trades only
for trade in output.allowed_trades:
    await execute_trade(trade)

# Log dropped trades
for trade in output.dropped_trades:
    logger.info(f"❌ {trade.symbol} dropped: {trade.reason}")
```

---

## 📊 KEY CONSTRAINTS (Default)

| Constraint | Limit | Purpose |
|------------|-------|---------|
| Max Positions | 8 | Prevent over-trading |
| Max Total Risk | 15% | Protect capital |
| Max Symbol Concentration | 30% | Diversification |
| Max Sector Concentration | 40% | Sector diversification |
| Max Leverage | 30x | Risk control |
| Max Net Exposure | 80% | Balance long/short |

---

## 🎮 RISK MODES

### 🛡️ SAFE
- Critical violations OR high drawdown
- **Action:** Block most trades, reduce risk

### ⚖️ NEUTRAL
- Normal operations
- **Action:** Standard filtering

### 🚀 AGGRESSIVE
- Healthy portfolio, low risk
- **Action:** Maximum opportunity capture

---

## ⚠️ COMMON VIOLATIONS

### Max Positions Reached
**Problem:** Too many open positions (e.g., 9/8)  
**Action:** Close weak positions before opening new ones

### Total Risk Exceeded
**Problem:** Portfolio risk > 15%  
**Action:** Reduce position sizes or close losing positions

### Symbol Over-Concentration
**Problem:** Single symbol > 30% of portfolio  
**Action:** Reduce position size in that symbol

### Sector Over-Concentration
**Problem:** Single sector > 40% of portfolio  
**Action:** Diversify across more sectors

---

## 🔍 OUTPUT STRUCTURE

```python
output = {
    "risk_mode": "SAFE|NEUTRAL|AGGRESSIVE",
    "portfolio_state": {
        "total_positions": 2,
        "total_risk_pct": 1.5,
        "net_exposure": 6500.0,
        ...
    },
    "violations": [
        {
            "constraint": "max_positions",
            "severity": "CRITICAL",
            "message": "Too many positions"
        }
    ],
    "allowed_trades": [
        {
            "symbol": "BTCUSDT",
            "action": "BUY",
            "allowed": true,
            "priority_score": 95.0
        }
    ],
    "dropped_trades": [
        {
            "symbol": "DOGEUSDT",
            "action": "BUY",
            "allowed": false,
            "reason": "LOW_PRIORITY"
        }
    ],
    "recommendations": [
        "RISK MODE: SAFE",
        "High symbol concentration - diversify"
    ],
    "actions_required": [
        "BLOCK NEW TRADES until violations resolved"
    ]
}
```

---

## 🎯 TRADE REJECTION REASONS

| Reason | Meaning |
|--------|---------|
| `CRITICAL_VIOLATIONS` | Portfolio has critical violations |
| `POSITION_ALREADY_OPEN` | Symbol already has open position |
| `MAX_POSITIONS_REACHED` | At max position limit |
| `TOTAL_RISK_EXCEEDED` | Would exceed total risk limit |
| `INSUFFICIENT_MARGIN` | Not enough margin available |
| `LOW_PRIORITY` | Lower priority than other trades |

---

## 📈 TRADE PRIORITY FORMULA

```
priority_score = confidence × 100
               + category_boost (CORE: +20, EXPANSION: +10)
               + stability_score × 10
               - cost_score × 5
               + recent_performance × 10
```

**Higher score = higher priority**

---

## 🧪 TESTING

```bash
# Run tests
cd backend
pytest tests/test_portfolio_balancer.py -v

# Run standalone demo
python services/portfolio_balancer.py
```

---

## 📁 FILES

| File | Purpose |
|------|---------|
| `backend/services/portfolio_balancer.py` | Main implementation |
| `backend/tests/test_portfolio_balancer.py` | Test suite |
| `PORTFOLIO_BALANCER_AI_GUIDE.md` | Full integration guide |
| `/app/data/portfolio_balancer_output.json` | Latest output |

---

## 🔧 INTEGRATION CHECKLIST

- [ ] Import PBA in event-driven executor
- [ ] Convert positions to `Position` objects
- [ ] Convert signals to `CandidateTrade` objects
- [ ] Call `analyze_portfolio()` before execution
- [ ] Check for critical violations
- [ ] Execute only `allowed_trades`
- [ ] Log `dropped_trades` reasons
- [ ] Respect `actions_required`
- [ ] Monitor `risk_mode` transitions

---

## 💡 BEST PRACTICES

1. **Always check `actions_required`** before executing trades
2. **Never ignore SAFE mode** - it means serious issues
3. **Log all violations** for pattern analysis
4. **Review dropped trades** - might indicate overly restrictive constraints
5. **Monitor risk mode transitions** - frequent SAFE mode = problems

---

## 🚨 EMERGENCY HANDLING

### If CRITICAL violations:
1. **STOP** - Don't open new positions
2. **ANALYZE** - Which constraint violated?
3. **FIX** - Close positions or reduce sizes
4. **VERIFY** - Run PBA again
5. **RESUME** - Continue when clear

### If stuck in SAFE mode:
1. Check daily drawdown (> 3%?)
2. Check losing streak (≥ 5?)
3. Check position count (at max?)
4. Check total risk (> 12%?)
5. Address root cause

---

## 📞 QUICK HELP

**Problem:** All trades blocked  
**Check:** `output.violations` for critical issues

**Problem:** Low priority trades dropped  
**Fix:** Improve signal confidence or use CORE symbols

**Problem:** SAFE mode constantly  
**Fix:** Reduce risk, improve win rate, lower drawdown

**Problem:** High concentration warnings  
**Fix:** Diversify across more symbols/sectors

---

## ✅ SUCCESS INDICATORS

- ✅ Risk mode mostly NEUTRAL/AGGRESSIVE
- ✅ Few violations per day
- ✅ High % of allowed trades (>80%)
- ✅ Diversified portfolio (multiple symbols/sectors)
- ✅ Total risk < 12%

---

**Quick Test:**
```bash
cd backend && python services/portfolio_balancer.py
```

**Full Guide:** `PORTFOLIO_BALANCER_AI_GUIDE.md`

**Support:** Check logs in `/app/data/portfolio_balancer_output.json`
