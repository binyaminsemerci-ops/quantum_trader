# 🛡️ RISK POLICY - Limits, Circuit Breakers & Fail-Closed

**Document**: Risk Management Policy  
**Authority**: CONSTITUTIONAL  
**Version**: 1.0  

---

## 1. Position Risk Limits

### Per-Trade Limits

| Parameter | Limit | Authority |
|-----------|-------|-----------|
| **Max Risk per Trade** | 2% of equity | Grunnlov §1 |
| **Max Stop Distance** | 3% from entry | Policy |
| **Max Leverage per Position** | See Scaling Levels | Policy |

### Portfolio Limits

| Parameter | Limit |
|-----------|-------|
| **Max Concurrent Positions** | 3 |
| **Max Exposure per Symbol** | 10% of equity |
| **Max Total Exposure** | 20% of equity |
| **Max Correlated Exposure** | 15% of equity |

---

## 2. Daily Risk Limits

| Parameter | Limit | Action on Breach |
|-----------|-------|------------------|
| **Max Daily Loss** | 5% of equity | FULL HALT (Grunnlov §2) |
| **Max Daily Trades** | 10 | New entries blocked |
| **Max Daily Drawdown** | 6% of equity | Reduce-only mode |

---

## 3. Drawdown Circuit Breakers

### Staged Response System

| Level | Threshold | Size Reduction | Trading Status |
|-------|-----------|----------------|----------------|
| **GREEN** | 0-5% DD | 100% | Normal |
| **YELLOW** | 5-8% DD | 75% | Warning, reduced |
| **ORANGE** | 8-12% DD | 50% | No new positions |
| **RED** | 12-15% DD | 25% | Close 50% |
| **BLACK** | >15% DD | 0% | FULL STOP |
| **CATASTROPHIC** | >20% DD | 0% | Kill-switch + 7 days |

### Automatic Actions

```
IF drawdown >= 5%:
    ALERT("WARNING: Drawdown at {dd}%")
    REDUCE_SIZE(25%)

IF drawdown >= 8%:
    BLOCK_NEW_ENTRIES()
    REDUCE_SIZE(50%)

IF drawdown >= 12%:
    CLOSE_POSITIONS(50%)
    
IF drawdown >= 15%:
    CLOSE_ALL()
    HALT_TRADING()
    
IF drawdown >= 20%:
    KILL_SWITCH()
    PAUSE(7_DAYS)
    REQUIRE_MANUAL_REVIEW()
```

---

## 4. Margin Safety

### Maintenance Requirements

| Parameter | Requirement |
|-----------|-------------|
| **Min Maintenance Margin** | 200% of exchange requirement |
| **Margin Warning Level** | 150% of exchange requirement |
| **Emergency Liquidation** | < 120% of exchange requirement |

### Margin Actions

| Level | Action |
|-------|--------|
| > 200% | Normal trading |
| 150-200% | Warning, no new positions |
| 120-150% | Reduce 50% immediately |
| < 120% | Emergency close ALL |

---

## 5. Fail-Closed Principle

> **"When in doubt, the system closes positions and halts trading."**

### Default States

| Condition | Default Action |
|-----------|----------------|
| Unknown state | HALT |
| Missing data | CLOSE POSITIONS |
| Error condition | REJECT TRADE |
| Ambiguous signal | NO ACTION |
| Network timeout | FORCE FLAT |

### There is NO "Fail-Open"

The system never:
- Assumes it's safe to continue
- Ignores missing data
- Proceeds without confirmation
- Defaults to "keep trading"

---

## 6. Loss Series Protection

### Consecutive Loss Thresholds

| Losses | System Response |
|--------|-----------------|
| 1-2 | Normal |
| 3 | Scale down 25% |
| 4-5 | Scale down 50%, reduce trade rate |
| 6 | Pause window (cooldown) |
| 7 | Strategy freeze, no new entries |
| 8-9 | Observer-only mode |
| 10+ | Shadow-only, full stop |

### Recovery Requirements

After any loss series >= 5:
1. 24-hour minimum pause
2. System health check
3. Performance review
4. Graduated restart (25% → 50% → 75% → 100%)

---

## 7. Slippage Policy

| Slippage Range | Classification | Action |
|----------------|----------------|--------|
| < 0.1% | Normal | Continue |
| 0.1% - 0.3% | Warning | Log, monitor |
| 0.3% - 0.5% | Elevated | Pause 1 hour |
| > 0.5% | Critical | Pause, investigate |

---

## 8. Volatility Adjustment

### Position Size Multiplier

| Market Volatility | Size Multiplier |
|-------------------|-----------------|
| Below average | 1.0x (full size) |
| Average | 0.75x |
| Elevated | 0.5x |
| Extreme | 0.25x or FLAT |

---

## 9. Risk Monitoring Requirements

### Real-Time Metrics

| Metric | Update Frequency | Alert Threshold |
|--------|------------------|-----------------|
| Unrealized PnL | 1 second | -3% of equity |
| Drawdown | 1 second | 5% |
| Margin Level | 1 second | 150% |
| Position Size | On change | Any limit breach |

### Daily Reports

- Total risk exposure
- Largest position
- Drawdown status
- Margin utilization
- Loss series count

---

## 10. Emergency Procedures

### Risk Emergency Protocol

1. **Detection**: Any limit breach detected
2. **Alert**: Immediate notification
3. **Assess**: Severity classification (A-E)
4. **Act**: Execute appropriate response
5. **Document**: Log all actions
6. **Review**: Post-incident analysis

### Severity Classifications

| Class | Description | Response Time |
|-------|-------------|---------------|
| A | Critical (total loss risk) | Immediate |
| B | Severe (multi-trade failure) | < 1 minute |
| C | Major (single trade issue) | < 5 minutes |
| D | Moderate (warning level) | < 1 hour |
| E | Minor (monitoring) | End of day |

---

**END OF RISK POLICY DOCUMENT**
