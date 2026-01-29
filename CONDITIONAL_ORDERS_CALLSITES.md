# CONDITIONAL ORDERS CALLSITES - FORENSIC AUDIT

**Date**: 2026-01-29  
**Status**: POLICY VIOLATION CONFIRMED  
**Action**: BLOCK + DISABLE

---

## Confirmed Callsites Creating Conditional Orders

### 1. `backend/services/execution/execution.py`

**Line 2657-2672**: `place_tpsl_orders()` - TAKE_PROFIT_MARKET
```python
tp_params = {
    'symbol': intent.symbol,
    'side': tp_side,
    'type': 'TAKE_PROFIT_MARKET',  # ❌ CONDITIONAL
    'stopPrice': tp_price,
    'closePosition': True,
    'workingType': 'MARK_PRICE'
}
```
- **Module**: `"execution_tpsl_shield"`
- **Trigger**: After entry execution in trading_bot
- **Risk**: HIGH - runs on every trade

**Line 2677-2692**: `place_tpsl_orders()` - STOP_MARKET
```python
sl_params = {
    'symbol': intent.symbol,
    'side': sl_side,
    'type': 'STOP_MARKET',  # ❌ CONDITIONAL
    'stopPrice': sl_price,
    'closePosition': True,
    'workingType': 'MARK_PRICE'
}
```
- **Module**: `"execution_tpsl_shield"`
- **Trigger**: After entry execution in trading_bot
- **Risk**: HIGH - runs on every trade

**Line 945-950**: `create_stop_loss_order()` - STOP_MARKET (fallback)
```python
sl_market_params = {
    "symbol": symbol,
    "side": sl_side.upper(),
    "type": "STOP_MARKET",  # ❌ CONDITIONAL
    "stopPrice": str(levels.sl_init),
    "quantity": sl_qty,
    "positionSide": position_side,
    "closePosition": "false"
}
```
- **Module**: Various (position management)
- **Trigger**: Fallback if LIMIT SL unfilled after 5s
- **Risk**: MEDIUM - fallback path

**Line 967-973**: `create_stop_loss_order()` - STOP_MARKET (immediate)
```python
sl_direct_params = {
    "symbol": symbol,
    "side": sl_side.upper(),
    "type": "STOP_MARKET",  # ❌ CONDITIONAL
    "stopPrice": str(levels.sl_init),
    "quantity": sl_qty,
    "positionSide": position_side,
    "closePosition": "false"
}
```
- **Module**: Various (position management)
- **Trigger**: Direct path (no LIMIT attempt)
- **Risk**: MEDIUM

---

### 2. `backend/services/execution/event_driven_executor.py`

**Line 3429-3436**: `_place_emergency_sl()` - STOP_MARKET
```python
sl_order_params = {
    'symbol': symbol,
    'side': sl_side,
    'type': 'STOP_MARKET',  # ❌ CONDITIONAL
    'stopPrice': sl_price,
    'closePosition': True,
    'workingType': 'MARK_PRICE'
}
```
- **Module**: `"event_driven_executor"`
- **Trigger**: Emergency SL shield
- **Risk**: HIGH - emergency path

**Line 3478-3485**: `_place_emergency_sl()` - STOP_MARKET (retry)
```python
sl_order_params_retry = {
    'symbol': symbol,
    'side': sl_side,
    'type': 'STOP_MARKET',  # ❌ CONDITIONAL
    'stopPrice': sl_price_retry,
    'closePosition': True,
    'workingType': 'MARK_PRICE'
}
```
- **Module**: `"event_driven_executor"`
- **Trigger**: Retry after failed initial attempt
- **Risk**: HIGH - emergency path

---

### 3. `backend/services/execution/hybrid_tpsl.py`

**Multiple lines**: Legacy TP/SL management
- Line 299: STOP_MARKET
- Line 308: TAKE_PROFIT_MARKET
- Line 322: STOP_MARKET
- Line 347: TAKE_PROFIT_MARKET
- Line 363: TAKE_PROFIT_MARKET
- Line 384: TRAILING_STOP_MARKET

**Status**: Legacy module (may not be active)
**Risk**: LOW - if not used

---

### 4. `backend/services/execution/trailing_stop_manager.py`

**Line 192**: STOP_MARKET
- **Module**: Trailing stop manager
- **Status**: Legacy/unused?
- **Risk**: LOW

---

## Responsible Services

| Service | Uses Component | Conditional Orders |
|---------|---------------|-------------------|
| `quantum-trading_bot.service` | `execution.py` TPSL shield | ✅ YES (HIGH) |
| `quantum-apply-layer.service` | Unknown | ❓ UNKNOWN |
| `quantum-intent-executor.service` | `event_driven_executor.py` | ✅ YES (MEDIUM) |
| `quantum-exitbrain-v35.service` | Exit Brain v3.5 | ❌ NO (MARKET only) |

---

## Policy Violation Summary

**Total Callsites**: 8+ locations creating conditional orders  
**Primary Violator**: `execution.py` TPSL shield (lines 2650-2710)  
**Secondary**: `event_driven_executor.py` emergency SL (lines 3429-3485)  
**Tertiary**: Legacy modules (hybrid_tpsl, trailing_stop_manager)

**Policy**: NO conditional orders. All exits via internal intents → MARKET execution only.

**Current Behavior**: Trading bot places conditional TP/SL on Binance after every entry.

---

## Remediation Plan

1. ✅ **Gateway Guard**: Block all conditional types at `exit_order_gateway.py`
2. ✅ **Disable Shield**: Gate TPSL shield behind `EXECUTION_TPSL_SHIELD_ENABLED=false`
3. ✅ **Proof Script**: Verify 0 conditional orders can be placed
4. ✅ **Deploy**: Update VPS configuration

---

**Next**: Implement gateway guard + disable shield
