# 🚨 CRITICAL BUG: Entry Price Mismatch

## Problem
**Systemet bruker signal price i stedet for actual fill price fra exchange!**

### Evidence
```
TRXUSDT:
- Logged Entry: $0.2823
- Actual Fill:  $0.28463
- Slippage:     +0.68% (!!)

SOLUSDT:
- Logged Entry: $133.4100
- Actual Fill:  $133.8500
- Slippage:     +0.33% (!!)
```

### Root Cause Analysis

#### 1. `BinanceFuturesAdapter.submit_order()` (execution.py:660-724)
```python
data = await self._signed_request("POST", "/fapi/v1/order", params)
order_id = str(data.get("orderId") or data.get("clientOrderId") or "")
# ❌ BUG: Ignores data['avgPrice'] - ACTUAL FILL PRICE!
return order_id  # Only returns ID
```

**Binance response contains:**
- `orderId`: Order ID
- `executedQty`: Filled quantity  
- **`avgPrice`: ACTUAL AVERAGE FILL PRICE** ← WE IGNORE THIS!

#### 2. `EventDrivenExecutor._execute_signals()` (event_driven_executor.py:2665)
```python
order_id = await self._adapter.submit_order(
    symbol=symbol,
    side=side,
    quantity=quantity,
    price=price,  # Signal price
    leverage=leverage
)
# ❌ BUG: Uses signal 'price' everywhere, not actual fill price

# Line 2686: TradeStore gets WRONG price
trade_obj = Trade(
    entry_price=price,  # ❌ Signal price, not fill price
    ...
)

# Line 2741: Slippage check uses WRONG price  
actual_fill_price = price  # TODO: Get from order response ← THIS TODO!

# Line 2757: TradeLifecycleManager gets WRONG price
trade = self._trade_manager.open_trade(
    actual_entry_price=price  # ❌ Signal price, not fill price
)
```

## Impact

### Financial Impact
- **All TP/SL calculations WRONG** - based on signal price, not entry
- **Risk calculations WRONG** - using incorrect cost basis
- **PnL tracking WRONG** - shows worse performance than actual
- **Trailing stops trigger at WRONG levels**

### Examples
```
TRXUSDT Trade:
- Signal: Entry $0.2823, SL $0.2894 (2.5%)
- Reality: Entry $0.28463, SL should be $0.29175 (2.5%)
- Current SL: $0.28894 = Only 1.51% protection!
- RISK EXPOSURE: 40% HIGHER THAN INTENDED
```

## Fix Required

### 1. Change `submit_order()` return type
```python
# BEFORE
async def submit_order(...) -> str:  # Returns only order_id
    data = await self._signed_request(...)
    return str(data.get("orderId"))

# AFTER  
async def submit_order(...) -> Dict[str, Any]:  # Returns full order data
    data = await self._signed_request(...)
    return {
        'order_id': str(data.get("orderId")),
        'filled_qty': float(data.get("executedQty", 0)),
        'avg_price': float(data.get("avgPrice", 0)),  # ← CRITICAL!
        'status': data.get("status"),
        'raw_response': data
    }
```

### 2. Update all `submit_order()` call sites
```python
# BEFORE
order_id = await self._adapter.submit_order(...)

# AFTER
order_result = await self._adapter.submit_order(...)
order_id = order_result['order_id']
actual_entry_price = order_result['avg_price']  # ← USE THIS!
```

### 3. Propagate actual_entry_price throughout
- Line 2686: `trade_obj = Trade(entry_price=actual_entry_price, ...)`
- Line 2741: `actual_fill_price = actual_entry_price` (remove TODO)
- Line 2757: `open_trade(actual_entry_price=actual_entry_price)`
- Line 2588: `self._symbol_tpsl[symbol]["entry_price"] = actual_entry_price`

### 4. Recalculate TP/SL based on ACTUAL entry
```python
# After getting actual_entry_price, recalculate:
if rl_decision:
    actual_tp = actual_entry_price * (1 + rl_decision.tp_pct / 100)
    actual_sl = actual_entry_price * (1 - rl_decision.sl_pct / 100)
    
    # Update decision with actual prices
    decision.take_profit = actual_tp
    decision.stop_loss = actual_sl
```

## Files to Modify

### Priority 1 (CRITICAL)
1. `backend/services/execution/execution.py`
   - Change `submit_order()` return type (line 660)
   - Return full order data with `avgPrice`

2. `backend/services/execution/event_driven_executor.py`
   - Update `submit_order()` call (line 2664)
   - Capture `order_result` instead of just `order_id`
   - Extract `actual_entry_price = order_result['avg_price']`
   - Use `actual_entry_price` everywhere (lines 2686, 2741, 2757, 2588)
   - Recalculate TP/SL based on actual entry

### Priority 2 (Related fixes)
3. `backend/services/risk_management/trade_lifecycle_manager.py`
   - Verify `actual_entry_price` parameter is used correctly
   - Update any downstream calculations

4. `backend/services/monitoring/position_monitor.py`
   - Verify it reads correct entry price from positions
   - Cross-check with stored trade data

## Testing Required

### 1. Unit Tests
- Test `submit_order()` returns correct structure
- Verify `avgPrice` extraction from Binance response
- Test slippage calculation with real fill prices

### 2. Integration Tests  
- Place real order (testnet)
- Verify logged entry price matches exchange
- Confirm TP/SL distances are correct percentages from ACTUAL entry

### 3. Regression Tests
- Check all existing trades still load correctly
- Verify position monitor shows correct data
- Confirm PnL calculations accurate

## Rollout Plan

### Phase 1: Fix Submit Order (30 min)
1. Modify `submit_order()` return structure
2. Extract `avgPrice` from Binance response
3. Return Dict with all order data

### Phase 2: Fix Event Executor (45 min)
1. Update `submit_order()` call site
2. Capture full order result
3. Extract and use `actual_entry_price`
4. Recalculate TP/SL based on actual entry
5. Update all downstream uses

### Phase 3: Testing (30 min)
1. Restart backend
2. Place test trade (TESTNET!)
3. Verify:
   - Logged entry == exchange entry
   - TP/SL distances correct
   - Position monitor shows right price
   - PnL calculation accurate

### Phase 4: Deployment (15 min)
1. Commit changes
2. Restart backend
3. Monitor first live trade closely
4. Verify all prices match

## Priority
**🔴 CRITICAL - Fix IMMEDIATELY**

This bug affects:
- ✅ Risk management (SL too tight/wide)
- ✅ PnL accuracy (wrong cost basis)
- ✅ Trade analytics (wrong entry prices)
- ✅ Trailing stops (wrong trigger levels)
- ✅ Breakeven moves (wrong calculation base)

**Every trade is affected!**
