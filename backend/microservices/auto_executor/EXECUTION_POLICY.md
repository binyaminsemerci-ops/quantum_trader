# Execution Policy & Capital Deployment

## Overview
P1-B central policy controlling **WHEN** positions open and **HOW MUCH** capital deployed. All entry decisions route through `ExecutionPolicy.allow_new_entry()`.

## Configuration (docker-compose.yml)
```bash
POLICY_MAX_POSITIONS_TOTAL=10              # Max open positions across all symbols
POLICY_MAX_PER_SYMBOL=3                    # Max positions per symbol
POLICY_MAX_PER_REGIME=5                    # Max positions per market regime
POLICY_MAX_EXPOSURE_TOTAL=5000.0           # Total capital at risk (USDT)
POLICY_MAX_EXPOSURE_PER_SYMBOL=1000.0      # Capital per symbol (USDT)
POLICY_ALLOW_SCALE_IN=true                 # Allow adding to existing positions
POLICY_SCALE_IN_MAX=2                      # Max scale-in count per symbol
POLICY_SCALE_IN_CONF_DELTA=0.05            # Confidence improvement required for scale-in
POLICY_COOLDOWN_SYMBOL=300                 # Seconds between trades (same symbol)
POLICY_COOLDOWN_GLOBAL=60                  # Seconds between trades (any symbol)
POLICY_MIN_CONFIDENCE=0.7                  # Minimum signal confidence
```

## Decision Flow (9-Step Priority)
```
1. Confidence Check        → BLOCK_LOW_CONFIDENCE if < 0.7
2. Global Cooldown         → BLOCK_COOLDOWN if < 60s since last trade
3. Symbol Cooldown         → BLOCK_COOLDOWN if < 300s since last trade (symbol)
4. Max Total Positions     → BLOCK_MAX_POSITIONS if >= 10
5. Max Total Exposure      → BLOCK_MAX_EXPOSURE if total >= $5000
6. Existing Position?
   ├─ YES + Scale-in ON    → ALLOW_SCALE_IN if:
   │                          • Same direction
   │                          • Confidence > existing + 0.05
   │                          • Count < 2
   │                          • Symbol exposure < $1000
   └─ NO                   → Continue
7. Max Exposure Per Symbol → BLOCK_MAX_EXPOSURE if symbol >= $1000
8. Max Positions Per Symbol→ BLOCK_MAX_POSITIONS if symbol count >= 3
9. Max Per Regime          → BLOCK_REGIME_RULE if regime count >= 5

✅ ALLOW_NEW_ENTRY if all checks pass
```

## Capital Allocation Formula
```python
base = available_capital * 0.20  # 20% of remaining capital
conf_mult = 0.70 + (confidence * 0.30)  # 0.7→0.85x, 1.0→1.15x
allocation = base * conf_mult * risk_score  # Adjusted by Risk Brain

# Hard caps
allocation = min(allocation, max_exposure_per_symbol - current_symbol_exposure)
allocation = min(allocation, max_total_exposure - current_total_exposure)
allocation = max(allocation, 10.0)  # Minimum $10 trade

# Convert to quantity
quantity = (allocation * leverage) / price
```

## Before P1-B (OLD)
```python
if executor.has_open_position(symbol):
    logger.warning(f"Already have position for {symbol}")
    return False  # Silent block

if confidence < 0.7:
    logger.warning(f"Confidence {confidence} below threshold")
    return False  # Silent block

qty = executor.calculate_position_size(symbol, price)  # Simple 1% account balance
```

## After P1-B (NEW)
```python
portfolio = executor.build_portfolio_state()
decision, reason = policy.allow_new_entry(intent, portfolio)

if decision not in [ALLOW_NEW_ENTRY, ALLOW_SCALE_IN]:
    logger.info(f"🚫 Entry blocked: {decision} | {reason}")
    return False  # Explicit logged block

qty = policy.compute_order_size(intent, portfolio, risk_score)
logger.info(f"✅ Policy approved: {decision} | qty={qty} | {reason}")
```

## Log Examples
```
✅ ENTRY_POLICY_DECISION | BTCUSDT | decision=ALLOW_NEW_ENTRY | confidence=0.85, exposure_after=$500, positions_after=1
🚫 ENTRY_POLICY_DECISION | ETHUSDT | decision=BLOCK_MAX_POSITIONS | total_positions=10 >= max=10
✅ ENTRY_POLICY_DECISION | BTCUSDT | decision=ALLOW_SCALE_IN | confidence=0.85, exposure_after=$850, positions_after=2
💰 CAPITAL_ALLOCATION | BTCUSDT | allocated=$500 (cap=$1000, avail=$4500) | qty=0.100 @ $50000 | lev=10x
```

## Testing
```bash
cd backend/microservices/auto_executor
python test_execution_policy.py
# Expects 7/7 tests PASSED
```

## Key Changes
- **Centralized Control**: All entry decisions in ONE class (`ExecutionPolicy`)
- **Explicit Logging**: Every block reason logged with `ENTRY_POLICY_DECISION`
- **Hard Caps**: Capital allocation capped by exposure limits ($5000 total, $1000 per symbol)
- **Scale-in Logic**: Requires confidence improvement (0.05 delta) + count limit (max 2)
- **Cooldown Controls**: 60s global, 300s per-symbol rate limiting
- **No AI Changes**: Only executor modified - models/Exit Brain/observability untouched
