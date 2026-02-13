# 🚫 NO-TRADE RULES - When NOT to Trade

**Document**: No-Trade Calendar & Rules  
**Authority**: CONSTITUTIONAL  
**Version**: 1.0  
**Grunnlov Reference**: §6, §7, §8, §9  

---

## Fundamental Principle

> **"Knowing when NOT to trade is as important as knowing HOW to trade."**

The system has three levels of No-Trade severity:
1. **ABSOLUTE** - System physically blocks all trading
2. **CONDITIONAL** - Trading only with manual override + documented reason
3. **HUMAN** - Warning issued, human must acknowledge

---

## 1. ABSOLUTE No-Trade Events

**System CANNOT trade under any circumstances.**

### Scheduled Events

| Event | Start | End | Reason |
|-------|-------|-----|--------|
| **CME Bitcoin Futures Expiry** | -4h | +4h | Extreme manipulation risk |
| **CME Options Expiry** | -4h | +4h | Pin risk, volatility spike |
| **Major FOMC Announcements** | -30min | +1h | USD correlation volatility |
| **BTC/ETH Halving Events** | -24h | +24h | Unknown market behavior |
| **Exchange Maintenance** | -1h | +maintenance duration | Data integrity risk |

### Dynamic Events

| Condition | Duration | Automatic? |
|-----------|----------|------------|
| Kill-switch activated | Until manual reset | Yes |
| Daily loss limit hit (5%) | 24 hours | Yes |
| 10+ consecutive losses | Until review complete | Yes |
| Data integrity failure | Until data restored | Yes |
| Exchange API down | Until API recovered | Yes |
| Drawdown > 15% | 48 hours minimum | Yes |

### Emergency Conditions

| Condition | Action |
|-----------|--------|
| Black Swan detected (>10% 1h move) | Immediate halt, 24h minimum |
| Flash crash detected | Halt, manual review required |
| Funding rate > ±0.5% | Halt affected pairs |
| Liquidity crisis (spread > 1%) | Halt affected pairs |

---

## 2. CONDITIONAL No-Trade Events

**Trading allowed ONLY with manual override + documented justification.**

### High-Risk Periods

| Period | Risk Level | Requirement for Trading |
|--------|------------|------------------------|
| **US CPI Release** | High | Written justification |
| **NFP (Non-Farm Payrolls)** | High | Position size 50% max |
| **Major Earnings (MSTR, COIN)** | Medium | Acknowledged risk |
| **Weekend Late Night (2-6 AM UTC)** | Medium | Reduced position |
| **Holiday Periods** | Medium | Manual monitoring |

### Market Condition Triggers

| Condition | Override Requirement |
|-----------|---------------------|
| Volatility > 2x normal | Acknowledge high-vol mode |
| Volume < 50% average | Acknowledge low liquidity |
| Funding rate > ±0.3% | Acknowledge funding risk |
| Multiple conflicting signals | Manual signal review |

### Override Protocol

```
To override CONDITIONAL no-trade:
1. Log intended trade
2. Document specific reason
3. Acknowledge elevated risk
4. Reduce position size by 50%
5. Set tighter stop-loss (2% max)
6. Human confirms within 5 minutes
```

---

## 3. HUMAN No-Trade Events

**System warns, human acknowledges, trading proceeds at reduced size.**

### Warning Events

| Event | Warning | Size Reduction |
|-------|---------|----------------|
| Minor economic data | "Economic event in 30min" | 25% |
| Unusual spread | "Spread elevated: 0.15%" | 25% |
| Time of day (off-hours) | "Low liquidity period" | 25% |
| Low confidence signal | "Signal confidence: 45%" | 50% |
| Counter-trend trade | "Trading against trend" | 50% |

### Human Acknowledgment

```
SYSTEM: Warning - [EVENT TYPE]
        Current risk: [LEVEL]
        Recommended: Wait or reduce size
        
HUMAN: [PROCEED / WAIT]

If PROCEED:
  - Log acknowledgment
  - Apply size reduction
  - Continue with trade
```

---

## 4. No-Trade Calendar 2026

### Fixed Annual Events

| Date | Event | Severity | Window |
|------|-------|----------|--------|
| Jan 27-29 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Mar 17-19 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Apr 15 | US Tax Deadline | HUMAN | Full day |
| May 5-7 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Jun 16-18 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Jul 28-30 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Sep 15-17 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Nov 3-5 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Dec 15-17 | FOMC Meeting | CONDITIONAL | ±4h of announcement |
| Dec 24-26 | Christmas | HUMAN | Low liquidity |
| Dec 31-Jan 2 | New Year | HUMAN | Low liquidity |

### Monthly Events (CME)

| Event | Day | Severity |
|-------|-----|----------|
| CME BTC Futures Expiry | Last Friday | ABSOLUTE (±4h) |
| CME Options Expiry | Last Friday | ABSOLUTE (±4h) |
| BTC Monthly Candle Close | Last day, 23:00-01:00 UTC | HUMAN |

### Weekly Patterns

| Day | Period | Severity | Reason |
|-----|--------|----------|--------|
| Sunday | 20:00-24:00 UTC | HUMAN | CME gap risk |
| Monday | 00:00-04:00 UTC | HUMAN | CME open volatility |
| Friday | 20:00-24:00 UTC | HUMAN | Weekend positioning |

---

## 5. Dynamic No-Trade Detection

### Market Condition Checks

```python
def should_trade(market_data):
    """
    Returns: (can_trade: bool, severity: str, reason: str)
    """
    
    # ABSOLUTE checks
    if kill_switch_active():
        return (False, "ABSOLUTE", "Kill-switch active")
    
    if daily_loss_exceeded():
        return (False, "ABSOLUTE", "Daily loss limit hit")
    
    if data_integrity_failed():
        return (False, "ABSOLUTE", "Data integrity failure")
    
    # CONDITIONAL checks
    if is_fomc_window():
        return (False, "CONDITIONAL", "FOMC announcement window")
    
    if funding_rate > 0.003:
        return (False, "CONDITIONAL", "High funding rate")
    
    # HUMAN checks
    if volatility > normal_volatility * 2:
        return (True, "HUMAN", "Elevated volatility - acknowledge")
    
    if signal_confidence < 0.5:
        return (True, "HUMAN", "Low confidence signal")
    
    return (True, "CLEAR", "Normal trading conditions")
```

---

## 6. No-Trade Override Authority

### Who Can Override What?

| Severity | Override Authority |
|----------|--------------------|
| ABSOLUTE | **NOBODY** during active conditions |
| CONDITIONAL | Human with documented justification |
| HUMAN | Human acknowledgment required |

### Override Logging

Every override is logged:

```json
{
    "override_id": "uuid",
    "timestamp": "ISO8601",
    "severity_overridden": "CONDITIONAL",
    "original_reason": "FOMC window",
    "justification": "...",
    "human_id": "founder_1",
    "size_reduction_applied": 0.5,
    "trade_result": "pending"
}
```

---

## 7. Grunnlov Integration

### No-Trade Rules by Grunnlov

| Grunnlov | No-Trade Trigger |
|----------|------------------|
| §2 (Daily Loss Limit) | If 5% loss → ABSOLUTE halt |
| §6 (Min Data Quality) | If data < 95% quality → ABSOLUTE |
| §7 (Fail-Closed) | On ANY error → default no-trade |
| §8 (No Holiday Trading) | Major holidays → HUMAN |
| §9 (Pre-Flight Required) | If pre-flight fails → ABSOLUTE |

---

## 8. No-Trade Checklist

Before every trade, verify:

```
□ Not during CME expiry window
□ Not during FOMC announcement
□ Not during data integrity issue
□ Not during kill-switch activation
□ Not during daily loss limit breach
□ Not during excessive drawdown
□ Volatility within acceptable range
□ Funding rate within limits
□ Liquidity acceptable
□ Signal confidence above threshold
```

If ANY box unchecked → EVALUATE no-trade severity.

---

**END OF NO-TRADE RULES DOCUMENT**
