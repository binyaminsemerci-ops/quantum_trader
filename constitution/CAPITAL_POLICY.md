# 💰 CAPITAL POLICY - Position Sizing & Scaling

**Document**: Capital Allocation Policy  
**Authority**: CONSTITUTIONAL  
**Version**: 1.0  
**Grunnlov Reference**: §1, §2, §6  

---

## 1. Fundamental Capital Rules

> **"Kapital bevares først. Profitt er et biprodukt."**

### Core Principles

1. **Never risk more than 2% per trade** (Grunnlov §1)
2. **Daily loss cap: 5% of equity** (Grunnlov §2)
3. **Drawdown recovery starts immediately**
4. **Capital is the last line of defense**

---

## 2. Position Sizing Formula

### Standard Kelly-Modified Sizing

```python
def calculate_position_size(equity, win_rate, risk_reward, max_risk_pct=0.02):
    """
    Kelly fraction modified for conservative application.
    """
    kelly_full = (win_rate * risk_reward - (1 - win_rate)) / risk_reward
    kelly_conservative = kelly_full * 0.25  # Use 25% of Kelly
    
    # Never exceed max risk
    risk_fraction = min(kelly_conservative, max_risk_pct)
    
    # Calculate position size
    risk_amount = equity * risk_fraction
    return risk_amount
```

### Position Size Limits

| Equity Range | Max Position Size |
|--------------|-------------------|
| < $10,000 | 5% of equity |
| $10,000 - $50,000 | 4% of equity |
| $50,000 - $100,000 | 3% of equity |
| > $100,000 | 2% of equity |

---

## 3. Scaling Levels (Capital Tiers)

### The 4 Levels of Capital Scaling

| Level | Stage | Capital | Max Leverage | Position Size |
|-------|-------|---------|--------------|---------------|
| **1** | PROOF_MODE | $1,000 | 2x | $50 max |
| **2** | CONFIRMED_MODE | $5,000 | 5x | $250 max |
| **3** | STABLE_MODE | $20,000 | 10x | $1,000 max |
| **4** | SCALABLE_MODE | $50,000+ | 15x | 2% of equity |

### Level Progression Requirements

#### Level 1 → Level 2 (PROOF → CONFIRMED)
- 100+ trades executed
- Win rate ≥ 45%
- Expectancy > 0.3R
- Max drawdown < 15%
- Time: minimum 30 days

#### Level 2 → Level 3 (CONFIRMED → STABLE)
- 300+ trades total
- Win rate ≥ 50%
- Expectancy > 0.5R
- Max drawdown < 12%
- Profit factor ≥ 1.5
- Time: minimum 90 days

#### Level 3 → Level 4 (STABLE → SCALABLE)
- 500+ trades total
- Win rate ≥ 52%
- Expectancy > 0.6R
- Max drawdown < 10%
- Profit factor ≥ 2.0
- Sharpe ratio ≥ 1.5
- Time: minimum 180 days

### Level Regression

Automatic downgrade if:
- Drawdown exceeds level threshold
- 10+ consecutive losses
- Win rate drops below minimum
- Any Grunnlov violation

---

## 4. Margin & Leverage Policy

### Leverage Limits by Volatility

| Volatility State | Max Leverage |
|------------------|--------------|
| Low (< 1% daily range) | Level max |
| Normal (1-3% daily range) | 75% of level max |
| High (3-5% daily range) | 50% of level max |
| Extreme (> 5% daily range) | 25% of level max or FLAT |

### Margin Utilization

| Utilization | Status | Action |
|-------------|--------|--------|
| 0-30% | Safe | Normal trading |
| 30-50% | Moderate | Monitor closely |
| 50-70% | Elevated | No new positions |
| 70-85% | Warning | Reduce existing |
| > 85% | Critical | Emergency close |

---

## 5. Capital Allocation Distribution

### Multi-Position Allocation

When running multiple positions:

```
Position 1: Max 40% of available capital
Position 2: Max 30% of available capital
Position 3: Max 20% of available capital
Reserve: Min 10% always

Total deployed: Max 90%
```

### Reserve Requirements

| Scenario | Minimum Reserve |
|----------|-----------------|
| Normal trading | 10% |
| High volatility | 20% |
| During drawdown | 30% |
| After loss series | 40% |

---

## 6. Drawdown-Based Scaling

### Automatic Size Reduction

| Drawdown | Size Multiplier |
|----------|-----------------|
| 0-3% | 1.0x (full size) |
| 3-5% | 0.85x |
| 5-8% | 0.70x |
| 8-10% | 0.50x |
| 10-12% | 0.30x |
| > 12% | 0x (no new trades) |

### Recovery Scaling (After Drawdown)

After hitting a drawdown threshold, recovery is gradual:

```python
def recovery_size_multiplier(days_since_bottom, current_dd):
    if current_dd > 12:
        return 0  # Still in danger zone
    
    base_recovery = {
        1: 0.25,
        3: 0.40,
        7: 0.60,
        14: 0.80,
        21: 0.90,
        30: 1.00
    }
    
    return min(base_recovery.get(days_since_bottom, 0.25), 1.0)
```

---

## 7. Profit Reinvestment

### Compound Growth Rules

| Profit Level | Reinvestment | Withdrawal |
|--------------|--------------|------------|
| < 10% cumulative | 100% reinvest | 0% |
| 10-25% cumulative | 90% reinvest | 10% |
| 25-50% cumulative | 80% reinvest | 20% |
| > 50% cumulative | 70% reinvest | 30% |

### Profit Lock-In

When position profit exceeds 10%, lock in 50% of gains:
- Tighten stop to break-even + 5% of unrealized profit
- This locked profit is protected from reversal

---

## 8. Capital Emergency Protocol

### Critical Capital Events

| Event | Action |
|-------|--------|
| Daily loss > 5% | Full halt, 24h pause |
| Drawdown > 15% | Emergency close, 48h pause |
| Drawdown > 20% | Kill-switch, 7-day pause |
| Margin call risk | Liquidate to 50% margin |

### Capital Recovery Procedure

After emergency:
1. **Day 1**: No trading, full system analysis
2. **Day 2-3**: Paper trading only
3. **Day 4-7**: 25% capital, 50% position size
4. **Day 8-14**: 50% capital, 75% position size
5. **Day 15+**: Gradual return to normal

---

## 9. Capital Allocation Audit

### Daily Capital Report

```
Report: Daily Capital Status
================================
Starting Equity: $XX,XXX.XX
Ending Equity: $XX,XXX.XX
Daily P&L: $X,XXX.XX (X.XX%)
Drawdown Status: X.XX%
Margin Used: XX%
Reserve Available: $X,XXX.XX (XX%)
Scaling Level: [1-4]
Size Multiplier: X.Xx

Positions:
- BTC: $X,XXX (XX% of equity)
- ETH: $X,XXX (XX% of equity)
Total Deployed: XX%
```

---

## 10. Forbidden Capital Actions

**NEVER:**
- Risk more than 2% per trade
- Exceed daily loss limit (5%)
- Go below minimum reserve (10%)
- Average down on losing positions
- Increase position size after loss
- Override drawdown scaling
- Borrow additional capital during drawdown

---

**END OF CAPITAL POLICY DOCUMENT**
