# STEP 3 – ORDER WIRING VERIFICATION

## Status: ✅ COMPLETE (No Changes Required)

### Discovery

The order recording infrastructure is **already in place**:

#### 1. Order Creation (NEW Status)
**File**: `backend/routes/trades.py` (Lines 283-287)
```python
t = TradeLog(
    symbol=symbol_upper,
    side=side_upper,
    qty=payload.qty,
    price=payload.price,
    status="NEW",  # ← Orders start as NEW
)
db.add(t)
db.commit()
```

**Trigger**: `POST /trades` endpoint
**Result**: TradeLog entry with status="NEW" is created when order is placed

#### 2. Order Closure (CLOSED Status)
**File**: `backend/services/risk_management/trade_lifecycle_manager.py` (Lines 652-666)
```python
trade_log = TradeLog(
    symbol=trade.symbol,
    side=trade.action.upper(),
    qty=trade.position_size,
    price=trade.exit_price or trade.entry_price,
    status="CLOSED",  # ← Orders closed when position exits
    reason=trade.exit_reason or "UNKNOWN",
    timestamp=trade.exit_time,
    realized_pnl=trade.realized_pnl_usd,
    realized_pnl_pct=trade.realized_pnl_pct,
    entry_price=trade.entry_price,
    exit_price=trade.exit_price or trade.entry_price,
    strategy_id=trade.strategy_id or "default"
)
db.add(trade_log)
db.commit()
```

**Trigger**: TradeLifecycleManager.exit_trade() when position closes
**Result**: TradeLog entry with status="CLOSED" is created when position exits

#### 3. Order Retrieval (OrderService)
**File**: `backend/domains/orders/service.py` (Lines 58-95)
```python
def get_recent_orders(self, limit: int = 50) -> List[OrderRecord]:
    try:
        trades = self.db_session.query(TradeLog).order_by(
            TradeLog.timestamp.desc()
        ).limit(limit).all()
        
        orders = []
        for trade in trades:
            # Maps TradeLog → OrderRecord
            status = self._map_status(trade.status)
            order = OrderRecord(...)
            orders.append(order)
        
        return orders
    except Exception as e:
        logger.error(f"[OrderService] Error: {e}")
        return []
```

**Result**: OrderService reads TradeLog and returns OrderRecord list

### Data Flow

```
User/AI → POST /trades
    ↓
TradeLog (status="NEW") ← OrderService.get_recent_orders()
    ↓                         ↓
Position Opens         Dashboard BFF (STEP 6)
    ↓                         ↓
Position Closes        Recent Orders Panel
    ↓
TradeLog (status="CLOSED")
    ↓
OrderService.get_recent_orders()
```

### Conclusion

**No additional wiring is needed for STEP 3** because:

1. ✅ Order creation is already hooked into TradeLog via `/trades` endpoint
2. ✅ Order closure is already hooked into TradeLog via TradeLifecycleManager
3. ✅ OrderService already reads from TradeLog
4. ✅ Status tracking is working (NEW → CLOSED)

### Next Action

Proceed to **STEP 4** - Wire AI Engine → SignalStore
