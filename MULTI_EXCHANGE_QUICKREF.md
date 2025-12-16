# Multi-Exchange Quick Reference (EPIC-EXCH-001)

**Status:** ✅ Phase 1 Complete (Foundation + Binance)  
**Phase 2:** System Integration (DEL 6) – PENDING

---

## 📦 Quick Import

```python
from backend.integrations.exchanges import (
    # Protocol & Exceptions
    IExchangeClient,
    ExchangeAPIError,
    
    # Enums
    OrderSide,           # BUY, SELL
    OrderType,           # MARKET, LIMIT, STOP_MARKET, etc.
    OrderStatus,         # NEW, FILLED, CANCELED, etc.
    TimeInForce,         # GTC, IOC, FOK, GTX
    PositionSide,        # BOTH, LONG, SHORT
    
    # Models
    OrderRequest,        # Request to place order
    OrderResult,         # Order placement result
    CancelResult,        # Order cancellation result
    Position,            # Futures position
    Balance,             # Account balance
    Kline,               # OHLCV candlestick
    
    # Factory
    ExchangeType,        # BINANCE, BYBIT, OKX
    ExchangeConfig,      # Exchange connection config
    get_exchange_client, # Factory function
    resolve_exchange_for_symbol,  # Symbol routing
    
    # Adapters (direct import optional)
    BinanceAdapter,
    BybitAdapter,
    OKXAdapter,
)
```

---

## 🚀 Usage Patterns

### Pattern 1: Direct Adapter Usage (Binance)

```python
from binance.client import Client
from backend.integrations.exchanges import BinanceAdapter, OrderRequest, OrderSide, OrderType
from decimal import Decimal

# Setup
client = Client(api_key, api_secret)
adapter = BinanceAdapter(client=client, testnet=False)

# Place market order
request = OrderRequest(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.01")
)
result = await adapter.place_order(request)
print(f"Order {result.order_id}: {result.status.value}")

# Place limit order with TP/SL
request = OrderRequest(
    symbol="ETHUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.LIMIT,
    quantity=Decimal("1.0"),
    price=Decimal("3000.00"),
    leverage=10,
    take_profit_price=Decimal("3500.00"),
    stop_loss_price=Decimal("2800.00")
)
result = await adapter.place_order(request)

# Get positions
positions = await adapter.get_open_positions(symbol="BTCUSDT")
for pos in positions:
    print(f"{pos.symbol}: {pos.quantity} @ {pos.entry_price} (PnL: {pos.unrealized_pnl})")

# Cancel order
cancel_result = await adapter.cancel_order("BTCUSDT", "12345")
print(f"Canceled: {cancel_result.success}")

# Close position
close_result = await adapter.close_position("BTCUSDT")
print(f"Position closed: {close_result.order_id}")
```

---

### Pattern 2: Factory + Routing

```python
from backend.integrations.exchanges import (
    ExchangeType,
    ExchangeConfig,
    get_exchange_client,
    resolve_exchange_for_symbol,
    OrderRequest,
    OrderSide,
    OrderType
)

# Setup configs
configs = {
    ExchangeType.BINANCE: ExchangeConfig(
        exchange=ExchangeType.BINANCE,
        api_key=os.getenv("BINANCE_API_KEY"),
        api_secret=os.getenv("BINANCE_API_SECRET"),
        client=binance_client,
        wrapper=rate_limiter,
        testnet=False
    ),
    ExchangeType.BYBIT: ExchangeConfig(
        exchange=ExchangeType.BYBIT,
        api_key=os.getenv("BYBIT_API_KEY"),
        api_secret=os.getenv("BYBIT_API_SECRET"),
        testnet=True
    ),
}

# Smart order placement (auto-route to correct exchange)
async def place_smart_order(symbol: str, side: OrderSide, quantity: Decimal):
    # Resolve exchange for symbol
    exchange_type = resolve_exchange_for_symbol(symbol)
    
    # Get adapter
    adapter = get_exchange_client(configs[exchange_type])
    
    # Place order
    request = OrderRequest(
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        quantity=quantity
    )
    
    result = await adapter.place_order(request)
    print(f"Order placed on {result.exchange}: {result.order_id}")
    return result

# Usage
await place_smart_order("BTCUSDT", OrderSide.BUY, Decimal("0.1"))
```

---

### Pattern 3: Multi-Exchange Position Aggregation

```python
from backend.integrations.exchanges import ExchangeType, get_exchange_client

async def get_all_positions() -> Dict[str, List[Position]]:
    """Fetch positions from all configured exchanges."""
    all_positions = {}
    
    for exchange_type in [ExchangeType.BINANCE, ExchangeType.BYBIT, ExchangeType.OKX]:
        try:
            adapter = get_exchange_client(configs[exchange_type])
            positions = await adapter.get_open_positions()
            all_positions[exchange_type.value] = positions
            
            print(f"{exchange_type.value}: {len(positions)} positions")
            
        except NotImplementedError:
            # Skip skeleton adapters (Bybit/OKX)
            print(f"{exchange_type.value}: Not implemented yet")
            continue
            
        except ExchangeAPIError as e:
            # Log error but continue
            print(f"{exchange_type.value}: Error - {e.message}")
            continue
    
    return all_positions

# Usage
positions_by_exchange = await get_all_positions()
```

---

### Pattern 4: Symbol Routing Configuration

```python
from backend.integrations.exchanges import (
    ExchangeType,
    set_symbol_exchange_mapping,
    load_symbol_mapping_from_policy
)

# Option 1: Programmatic mapping
set_symbol_exchange_mapping({
    "BTCUSDT": ExchangeType.BINANCE,
    "ETHUSDT": ExchangeType.BINANCE,
    "SOLUSDT": ExchangeType.BYBIT,
    "ADAUSDT": ExchangeType.OKX,
})

# Option 2: Load from PolicyStore
load_symbol_mapping_from_policy(policy_store)

# Check current mapping
from backend.integrations.exchanges import get_current_symbol_mapping
mapping = get_current_symbol_mapping()
print(mapping)
# {"BTCUSDT": ExchangeType.BINANCE, "SOLUSDT": ExchangeType.BYBIT, ...}
```

---

### Pattern 5: Error Handling

```python
from backend.integrations.exchanges import ExchangeAPIError

try:
    result = await adapter.place_order(request)
    
except ExchangeAPIError as e:
    # Unified error handling for all exchanges
    print(f"Exchange: {e.exchange}")
    print(f"Error: {e.message}")
    print(f"Code: {e.code}")
    print(f"Original: {e.original_error}")
    
    # Handle specific errors
    if e.code == -2010:  # Binance: Insufficient balance
        await handle_insufficient_balance()
    elif "rate limit" in e.message.lower():
        await handle_rate_limit()
    else:
        await handle_generic_error(e)
```

---

## 🔧 Model Reference

### OrderRequest
```python
OrderRequest(
    symbol: str,                      # Required: "BTCUSDT"
    side: OrderSide,                  # Required: BUY / SELL
    order_type: OrderType,            # Required: MARKET / LIMIT / etc.
    quantity: Decimal,                # Required: 0.5
    price: Decimal = None,            # Limit orders only
    stop_price: Decimal = None,       # Stop orders only
    time_in_force: TimeInForce = None,# GTC / IOC / FOK / GTX
    reduce_only: bool = False,        # True = close only
    position_side: PositionSide = None,# BOTH / LONG / SHORT
    client_order_id: str = None,      # Custom ID
    leverage: int = None,             # 1-125x
    take_profit_price: Decimal = None,# TP level
    stop_loss_price: Decimal = None,  # SL level
    trailing_stop_pct: Decimal = None,# Trailing %
)
```

### OrderResult
```python
OrderResult(
    order_id: str,                    # "12345"
    client_order_id: str,             # "my_order_1"
    symbol: str,                      # "BTCUSDT"
    side: OrderSide,                  # BUY / SELL
    order_type: OrderType,            # MARKET / LIMIT / etc.
    quantity: Decimal,                # 0.5
    filled_quantity: Decimal,         # 0.5 (partial/full)
    price: Decimal,                   # Order price
    average_price: Decimal,           # Execution price
    status: OrderStatus,              # NEW / FILLED / etc.
    timestamp: datetime,              # Execution time
    exchange: str,                    # "binance"
    raw_response: Dict = None,        # Original API response
)
```

### Position
```python
Position(
    symbol: str,                      # "BTCUSDT"
    side: OrderSide,                  # BUY (long) / SELL (short)
    quantity: Decimal,                # 1.5
    entry_price: Decimal,             # 50000.00
    mark_price: Decimal,              # 51000.00
    liquidation_price: Decimal,       # 45000.00
    unrealized_pnl: Decimal,          # 1500.00
    realized_pnl: Decimal,            # 0.00
    leverage: int,                    # 10
    margin: Decimal,                  # 5100.00
    take_profit: Decimal = None,      # TP level
    stop_loss: Decimal = None,        # SL level
    exchange: str,                    # "binance"
    position_side: PositionSide = None,# BOTH / LONG / SHORT
)
```

---

## 🎯 IExchangeClient Protocol

All adapters implement these 9 methods:

```python
class IExchangeClient(Protocol):
    async def place_order(request: OrderRequest) -> OrderResult
    async def cancel_order(symbol: str, order_id: str) -> CancelResult
    async def get_open_positions(symbol: Optional[str] = None) -> List[Position]
    async def get_balances(asset: Optional[str] = None) -> List[Balance]
    async def get_klines(
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Kline]
    async def get_order_status(symbol: str, order_id: str) -> OrderResult
    async def set_leverage(symbol: str, leverage: int) -> bool
    async def close_position(symbol: str) -> OrderResult
    def get_exchange_name() -> str
```

---

## ⚙️ Configuration

### Environment Variables
```bash
# Binance
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Bybit
BYBIT_API_KEY=your_key
BYBIT_API_SECRET=your_secret

# OKX
OKX_API_KEY=your_key
OKX_API_SECRET=your_secret
OKX_PASSPHRASE=your_passphrase
```

### Symbol Routing (policy/exchanges.yml)
```yaml
symbol_exchange_mapping:
  BTCUSDT: binance
  ETHUSDT: binance
  SOLUSDT: bybit
  ADAUSDT: okx
```

### Load at Startup (backend/main.py)
```python
from backend.integrations.exchanges import load_symbol_mapping_from_policy

@app.on_event("startup")
async def startup():
    load_symbol_mapping_from_policy(policy_store)
    logger.info("Multi-exchange routing loaded")
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/unit/test_multi_exchange_epic_exch_001.py -v
```

### Test BinanceAdapter (Manual)
```python
from binance.client import Client
from backend.integrations.exchanges import BinanceAdapter, OrderRequest, OrderSide, OrderType
from decimal import Decimal

# Testnet
client = Client(api_key, api_secret, testnet=True)
adapter = BinanceAdapter(client=client, testnet=True)

# Test order
request = OrderRequest(
    symbol="BTCUSDT",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=Decimal("0.001")
)

result = await adapter.place_order(request)
print(f"Order placed: {result.order_id}")

# Test positions
positions = await adapter.get_open_positions()
print(f"Open positions: {len(positions)}")
```

---

## 🚨 Current Limitations

### Bybit/OKX (Skeletons)
- ❌ All methods raise `NotImplementedError`
- ✅ Can create adapter instances
- ✅ Factory routing works
- ✅ Ready for future implementation

### System Integration
- ⏳ **Execution Service NOT updated** (DEL 6 pending)
- ⏳ **Portfolio Intelligence NOT updated**
- ⏳ Direct Binance calls still in use

### Advanced Features
- ⏳ WebSocket support (coming)
- ⏳ Fee model standardization (coming)
- ⏳ Multi-exchange dashboard (coming)

---

## 📚 See Also

- **Full Report:** `EPIC_EXCH_001_COMPLETION.md`
- **Protocol Definition:** `backend/integrations/exchanges/base.py`
- **Models:** `backend/integrations/exchanges/models.py`
- **BinanceAdapter:** `backend/integrations/exchanges/binance_adapter.py`
- **Factory:** `backend/integrations/exchanges/factory.py`
- **Tests:** `tests/unit/test_multi_exchange_epic_exch_001.py`

---

**Ready to use BinanceAdapter?** ✅  
**Ready to integrate into Execution Service?** ⏳ DEL 6 pending  
**Ready for Bybit/OKX?** ⏳ Coming soon
