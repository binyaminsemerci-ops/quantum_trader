# TradeStore Quick Reference

## 🚀 Quick Start

```python
from backend.core.trading import get_trade_store, Trade, TradeSide, TradeStatus

# Initialize (auto-selects Redis or SQLite)
store = await get_trade_store(redis_client=redis_client)

# Create trade
trade = Trade(
    trade_id="BTC_001",
    symbol="BTCUSDT",
    side=TradeSide.LONG,
    status=TradeStatus.OPEN,
    quantity=1.0,
    leverage=10.0,
    margin_usd=5000.0,
    entry_price=50000.0,
    entry_time=datetime.utcnow(),
    sl_price=48000.0,
    tp_price=55000.0
)

# Save
await store.save_new_trade(trade)

# Retrieve
trade = await store.get_trade_by_id("BTC_001")

# Update
await store.update_trade("BTC_001", sl_price=49000.0)

# Close
await store.mark_trade_closed(
    "BTC_001",
    exit_price=54000.0,
    exit_time=datetime.utcnow(),
    close_reason="TP"
)
```

## 📋 Trade Model Fields

| Category | Fields |
|----------|--------|
| **Identity** | `trade_id`, `symbol`, `side`, `status` |
| **Position** | `quantity`, `leverage`, `margin_usd` |
| **Entry** | `entry_price`, `entry_time` |
| **Exit Mgmt** | `sl_price`, `tp_price`, `trail_percent` |
| **Exit Details** | `exit_price`, `exit_time`, `close_reason` |
| **Performance** | `pnl_usd`, `pnl_pct`, `r_multiple` |
| **Fees** | `entry_fee_usd`, `exit_fee_usd`, `funding_fees_usd` |
| **AI Context** | `model`, `confidence`, `meta_strategy_id`, `regime` |
| **RL Sizing** | `rl_state_key`, `rl_action_key`, `rl_leverage_original` |
| **Exchange** | `exchange_order_id`, `sl_order_id`, `tp_order_id` |
| **Metadata** | `metadata`, `created_at`, `updated_at` |

## 🔄 Trade Status Enum

```python
TradeStatus.PENDING      # Order submitted, awaiting fill
TradeStatus.OPEN         # Position active
TradeStatus.PARTIAL_TP   # Partial profit taken
TradeStatus.CLOSED       # Position fully closed
TradeStatus.CANCELLED    # Order cancelled before fill
TradeStatus.FAILED       # Order/execution failed
```

## 📊 Common Operations

### Get All Open Trades
```python
open_trades = await store.get_open_trades()
```

### Get Open Trades for Symbol
```python
btc_trades = await store.get_open_trades(symbol="BTCUSDT")
```

### Update Multiple Fields
```python
await store.update_trade(
    trade_id,
    sl_price=new_sl,
    status=TradeStatus.BREAKEVEN.value,
    trail_percent=0.025
)
```

### Get Storage Stats
```python
stats = await store.get_stats()
# Returns: backend, total_trades, open_trades, closed_trades, total_pnl_usd
```

## 🎛️ Backend Selection

```python
# Auto-select (Redis if available, else SQLite)
store = await get_trade_store(redis_client=redis)

# Force SQLite (e.g., for tests)
store = await get_trade_store(force_sqlite=True)

# Reset singleton (for tests)
from backend.core.trading import reset_trade_store
reset_trade_store()
```

## 📁 Storage Locations

- **SQLite**: `runtime/trades.db`
- **Redis**: Keys `trade:{id}` (hashes)

## ✅ Test Status

**14/14 tests passing** (100%)
- SQLite backend: ✅
- Factory/selector: ✅
- Integration: ✅
- Edge cases: ✅
- Redis backend: Skipped (not running locally)

## 📚 Documentation

See `SPRINT1_D5_TRADESTORE_MIGRATION.md` for:
- Complete architecture
- Implementation details
- Usage examples
- Migration guide
- Impact analysis
