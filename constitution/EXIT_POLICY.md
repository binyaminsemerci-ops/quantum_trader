# 🚪 EXIT POLICY - The 5 Exit Formulas

**Document**: Exit & Stop-Loss Policy  
**Authority**: CONSTITUTIONAL  
**Version**: 1.0  
**Grunnlov Reference**: §3, §4, §5  

---

## Fundamental Principle

> **"AI determines entries. ExitBrain determines exits. No exceptions."**

AI signals are ADVISORY ONLY for exits. The ExitBrain service has autonomous authority over all position closures.

---

## 1. The 5 Exit Formulas

### Formula 1: Stop-Loss (MANDATORY)

Every position MUST have a stop-loss.

```
STOP_LOSS = CATASTROPHIC_PROTECTION

Parameters:
- Max Stop Distance: 3% from entry
- Trailing Mode: Optional, min 1% from peak
- Never removed, only tightened
```

**Implementation**:
```python
if entry_price and not stop_loss:
    REJECT_TRADE("Grunnlov §5 violation: No stop-loss")
```

---

### Formula 2: Take-Profit (SMART SCALING)

Tiered profit-taking based on R-multiples.

| Target | R-Multiple | Action |
|--------|------------|--------|
| TP1 | 1.0R | Close 40% |
| TP2 | 1.5R | Close 30% |
| TP3 | 2.0R | Close remaining |

**R-Multiple Definition**:
```
R = Initial Risk Amount
1R = distance from entry to stop-loss
```

---

### Formula 3: Time-Based Exit (MAX HOLD)

No position lives forever.

| Position Age | Action |
|--------------|--------|
| < 24 hours | Normal |
| 24-48 hours | Review required |
| 48-72 hours | Escalated monitoring |
| > 72 hours | FORCE CLOSE |

**Exception**: Positions at 1.5R+ profit may extend to 96 hours max.

---

### Formula 4: Regime-Change Exit

Exit when market conditions no longer support the trade thesis.

**Triggers**:
- Volatility regime shift (e.g., calm → chaotic)
- Trend invalidation (ADX < 20)
- Correlation break (correlated assets diverge)
- Liquidity deterioration (spread > 2x normal)

**Action**:
```
IF regime != entry_regime:
    INITIATE_ORDERLY_EXIT()
    # Close 50% immediately, 50% over next hour
```

---

### Formula 5: Circuit-Breaker Exit

Emergency exit when risk limits are breached.

| Trigger | Exit Speed |
|---------|------------|
| Daily loss limit | Market order, ALL positions |
| Portfolio drawdown 12%+ | Close 50% immediately |
| Kill-switch active | Liquidate ALL |
| Data integrity fail | Close ALL, go flat |

---

## 2. Stop-Loss Rules

### Immutable Stop-Loss Principles

1. **Every trade has a stop** - No exceptions
2. **Stops can only tighten** - Never widened
3. **Stop is set at entry** - Before position opens
4. **Stop survives system restart** - Stored redundantly
5. **Stop is exchange-side** - Not software-only

### Stop-Loss Types

| Type | Use Case | Max Distance |
|------|----------|--------------|
| **Fixed** | Standard trades | 3% |
| **Trailing** | Trending markets | 1.5% from peak |
| **Volatility-Adjusted** | High-vol regimes | 2x ATR, max 4% |

### Stop Recovery After Fill

If stop-loss hit:
1. Log exit reason
2. Calculate realized loss
3. Update loss series counter
4. Apply cooldown if loss series >= 3
5. No re-entry same direction for 4 hours

---

## 3. Trailing Stop Logic

### Activation
Trailing stop activates when position reaches 1R profit.

### Trailing Algorithm
```python
def update_trailing_stop(position, current_price):
    if position.is_long:
        high_water = max(position.high_water_mark, current_price)
        new_stop = high_water * (1 - TRAIL_PERCENT)
        if new_stop > position.stop_loss:
            position.stop_loss = new_stop
    else:
        low_water = min(position.low_water_mark, current_price)
        new_stop = low_water * (1 + TRAIL_PERCENT)
        if new_stop < position.stop_loss:
            position.stop_loss = new_stop
```

### Trailing Parameters

| Profit Range | Trail Distance |
|--------------|----------------|
| 1R - 1.5R | 2% |
| 1.5R - 2R | 1.5% |
| > 2R | 1% |

---

## 4. Partial Exit Rules

### Scaling Out

| Event | Action | Remaining Position |
|-------|--------|-------------------|
| TP1 hit (1R) | Exit 40% | 60% |
| TP2 hit (1.5R) | Exit 50% of remaining | 30% |
| TP3 hit (2R) | Exit remaining | 0% |

### Forced Partial Exit

```
IF unrealized_loss > 2% AND position_age > 12h:
    CLOSE(50%)
    TIGHTEN_STOP()
```

---

## 5. Exit Priority Chain

When multiple exit conditions are present:

```
1. Kill-Switch         → Immediate liquidation
2. Circuit-Breaker     → Market close ALL
3. Stop-Loss           → Execute stop
4. Time-Based          → Close at max hold
5. Regime-Change       → Orderly exit
6. Take-Profit         → Scaled exit
```

Higher priority always overrides.

---

## 6. Exit Execution

### Order Types

| Exit Reason | Order Type | Slippage Tolerance |
|-------------|------------|-------------------|
| Stop-Loss | Limit with fallback to market | 0.5% |
| Take-Profit | Limit | 0.2% |
| Time-Based | Limit, then market after 5 min | 0.3% |
| Emergency | Market | Unlimited |

### Execution Monitoring

Every exit is monitored:
- Expected fill price
- Actual fill price
- Slippage amount
- Execution time
- Anomaly detection

---

## 7. No AI Override

AI cannot:
- Cancel an exit order
- Move a stop-loss
- Extend hold time
- Override circuit-breaker

AI can only:
- Suggest exit timing (advisory)
- Recommend partial exit levels
- Request manual review

---

## 8. Exit Logging

Every exit logs:

```json
{
    "exit_id": "uuid",
    "position_id": "ref",
    "exit_type": "stop_loss|take_profit|time|regime|circuit",
    "exit_formula": 1-5,
    "entry_price": 50000.0,
    "exit_price": 49000.0,
    "pnl_usd": -200.0,
    "pnl_percent": -2.0,
    "hold_time_hours": 12.5,
    "exit_reason_detail": "...",
    "timestamp": "ISO8601"
}
```

---

**END OF EXIT POLICY DOCUMENT**
