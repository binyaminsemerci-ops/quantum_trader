# Execution Service

**Service #2 of 7 - Sprint 2 Microservices Architecture**

## Purpose

Orchestrates order execution, position monitoring, and trade lifecycle management with multi-layer safety guards.

## Responsibilities

- **Order Execution:** Entry + exit order placement via Binance Futures
- **Position Monitoring:** 10-second loop to track open positions and manage TP/SL
- **Trade Lifecycle:** Full lifecycle from signal → order → fill → close → TradeStore
- **Safety Integration:** ESS checks, ExecutionSafetyGuard validation, SafeOrderExecutor retry logic

## Architecture

### Port
- **8002** (HTTP REST API)

### Events IN (Subscriptions)
- `ai.decision.made` - Main execution trigger from ai-engine-service
- `signal.execute` - Manual signal execution
- `ess.tripped` - ESS state change (blocks orders if CRITICAL)
- `policy.updated` - Policy change notification
- `model.promoted` - Model update notification

### Events OUT (Publications)
- `order.placed` - Order successfully placed on Binance
- `order.filled` - Order fill confirmation
- `order.failed` - Order placement/fill failure
- `trade.opened` - New trade opened with entry price
- `trade.closed` - Trade closed with exit price and PnL
- `position.updated` - Position status update (PnL, margin, etc.)

### REST API Endpoints

#### Health
- `GET /health` - Service health with component status

#### Order Management
- `POST /api/execution/order` - Place manual order

#### Position Queries
- `GET /api/execution/positions` - List all current positions

#### Trade Queries
- `GET /api/execution/trades` - Trade history (paginated)
- `GET /api/execution/trades/{trade_id}` - Specific trade details

#### Metrics
- `GET /api/execution/metrics` - Execution performance metrics

## Dependencies

### Internal (Sprint 1 Modules)
- **D5: TradeStore** - Trade persistence (Redis + SQLite)
- **D6: GlobalRateLimiter** - Rate limiting (1200 RPM)
- **D6: BinanceClientWrapper** - Rate-limited Binance API client
- **D7: ExecutionSafetyGuard** - Slippage validation
- **D7: SafeOrderExecutor** - Order placement with retry logic

### External Services
- **risk-safety-service** (:8003) - ESS status checks, PolicyStore queries
- **Redis** - EventBus + TradeStore backend
- **Binance Futures API** - Order execution

## Configuration

See `config.py` for all settings. Key configs:

```python
# Binance
USE_BINANCE_TESTNET = True  # Auto-switch testnet/mainnet
BINANCE_API_KEY = "..."
BINANCE_API_SECRET = "..."

# Redis
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Risk-safety-service
RISK_SAFETY_SERVICE_URL = "http://localhost:8003"

# Execution
POSITION_MONITOR_INTERVAL_SEC = 10
BINANCE_RATE_LIMIT_RPM = 1200
```

## Trade Execution Flow

```
1. AI Decision Event → execution-service
   └─ ai.decision.made with symbol, side, confidence, TP/SL

2. ESS Check
   └─ HTTP GET to risk-safety-service: /api/risk/ess/status
   └─ Block order if ESS state = CRITICAL

3. Safety Validation (D7)
   └─ ExecutionSafetyGuard.validate_and_adjust_order()
   └─ Check slippage limits
   └─ Adjust TP/SL if needed

4. Order Placement (D7)
   └─ SafeOrderExecutor.place_order()
   └─ Binance API call with retry logic (3 attempts)
   └─ Rate limited via D6 GlobalRateLimiter

5. Trade Persistence (D5)
   └─ TradeStore.save_new_trade()
   └─ Redis + SQLite dual backend

6. Event Publication
   └─ order.placed → EventBus
   └─ trade.opened → EventBus

7. Position Monitoring Loop
   └─ Every 10 seconds: Check open positions
   └─ Place TP/SL orders if missing
   └─ Detect fills and close trades
   └─ Update TradeStore with exit prices/PnL
```

## Position Monitoring

Background task running every 10 seconds:

1. Fetch all open positions from Binance
2. Cross-reference with TradeStore open trades
3. For each position:
   - Check if TP/SL orders exist
   - Place missing TP/SL orders
   - Detect exits (position closed)
   - Update TradeStore with exit data
   - Publish `trade.closed` event

## Safety Layers

### Layer 1: ESS Check (D3)
- Query risk-safety-service before every order
- Block trading if ESS state = CRITICAL
- Fail-open if ESS check times out

### Layer 2: ExecutionSafetyGuard (D7)
- Slippage validation (configurable max %)
- TP/SL sanity checks (min distance from entry)
- Automatic adjustment if needed

### Layer 3: SafeOrderExecutor (D7)
- Retry logic for transient errors (max 3 attempts)
- Binance error code handling (-1001, -2011, etc.)
- Order status verification after placement

### Layer 4: GlobalRateLimiter (D6)
- Token bucket rate limiting (1200 RPM)
- Prevents API ban from excessive requests

## Running the Service

### Local Development
```bash
cd microservices/execution
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8002
```

### Docker
```bash
docker build -t execution-service .
docker run -p 8002:8002 \
  -e BINANCE_API_KEY="your_key" \
  -e BINANCE_API_SECRET="your_secret" \
  -e REDIS_HOST="redis" \
  execution-service
```

### Docker Compose (Sprint 2 Full Stack)
```bash
cd microservices
docker-compose up execution
```

## Testing

```bash
# Health check
curl http://localhost:8002/health

# Place manual order
curl -X POST http://localhost:8002/api/execution/order \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "long",
    "quantity": 0.001,
    "price": 50000,
    "leverage": 10,
    "stop_loss": 49000,
    "take_profit": 52000
  }'

# Get positions
curl http://localhost:8002/api/execution/positions

# Get trades
curl http://localhost:8002/api/execution/trades?limit=50&status=closed
```

## Integration with Other Services

### risk-safety-service (:8003)
- **ESS status:** `GET /api/risk/ess/status` before every order
- **PolicyStore:** `GET /api/policy/{key}` for config values

### ai-engine-service (:8001)
- **Consumes events:** `ai.decision.made` (main execution trigger)

### portfolio-intelligence-service (:8004)
- **Publishes events:** `trade.opened`, `trade.closed`, `position.updated`
- Portfolio service consumes these for PnL tracking

### monitoring-health-service (:8005)
- **Publishes metrics:** `order.placed`, `order.failed` counts
- Health service aggregates for dashboard

## Known Limitations (MVP)

1. **PolicyStore snapshot:** Currently not cached locally, ESS check is the only external dependency enforced
2. **Position monitoring:** Basic 10-second polling (future: WebSocket for real-time updates)
3. **Paper adapter:** PaperExchangeAdapter not fully integrated (use testnet for testing)
4. **Metrics:** Execution metrics API endpoint returns placeholder data

## Future Enhancements (Post-Sprint 2)

- [ ] Local PolicyStore snapshot caching
- [ ] WebSocket for real-time position updates
- [ ] Advanced order types (trailing stops, iceberg orders)
- [ ] Paper trading adapter integration
- [ ] Prometheus metrics exporter
- [ ] Circuit breaker for Binance API failures

## Sprint 2 Status

✅ **COMPLETE** (Service #2 of 7)
- [x] Boilerplate (main.py, config.py, models.py)
- [x] Core service logic (service.py)
- [x] REST API endpoints (api.py)
- [x] Event handlers (integrated in service.py)
- [x] Dockerfile
- [x] Documentation

**Next:** Service #3 (ai-engine-service)
