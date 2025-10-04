# 📚 Quantum Trader API Documentation

Complete reference for the Quantum Trader REST API with examples, authentication, and best practices.

## 🔗 Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`
- **Interactive Docs**: `http://localhost:8000/api/docs`
- **ReDoc**: `http://localhost:8000/api/redoc`

## 🔐 Authentication

Most endpoints require API key authentication. Set your credentials via the settings endpoint.

### Setting API Credentials

```http
POST /api/settings
Content-Type: application/json

{
  "api_key": "your_binance_api_key",
  "api_secret": "your_binance_api_secret"
}
```

### Headers

```http
X-API-Key: your_api_key
X-API-Secret: your_api_secret
```

## 📊 Core Endpoints

### Health Check

Monitor application status and database connectivity.

```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-04T13:54:00Z",
  "version": "1.0.0",
  "database": "connected"
}
```

### Trading Operations

#### Get Trading History

Retrieve recent trades with filtering options.

```http
GET /api/trades?limit=50&symbol=BTCUSDT&status=FILLED
```

**Parameters:**
- `limit` (optional): Number of trades to return (default: 100, max: 1000)
- `symbol` (optional): Filter by trading pair (e.g., "BTCUSDT")
- `status` (optional): Filter by status ("FILLED", "CANCELLED", "PARTIALLY_FILLED")
- `from_date` (optional): Start date (ISO 8601 format)
- `to_date` (optional): End date (ISO 8601 format)

**Response:**
```json
{
  "trades": [
    {
      "id": 123,
      "symbol": "BTCUSDT",
      "side": "BUY",
      "qty": 0.01,
      "price": 43500.00,
      "status": "FILLED",
      "reason": "Strong bullish signal detected",
      "timestamp": "2025-10-04T11:30:00Z"
    }
  ],
  "total": 1,
  "page": 1,
  "pages": 1
}
```

#### Execute Trade

Place a new trading order.

```http
POST /api/trades
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "side": "BUY",
  "qty": 0.01,
  "price": 43500.00,
  "order_type": "LIMIT"
}
```

**Request Body:**
- `symbol` (required): Trading pair symbol
- `side` (required): "BUY" or "SELL"
- `qty` (required): Quantity to trade
- `price` (optional): Limit price (required for LIMIT orders)
- `order_type` (optional): "MARKET" or "LIMIT" (default: "MARKET")

**Response:**
```json
{
  "order_id": "12345",
  "status": "FILLED",
  "symbol": "BTCUSDT",
  "side": "BUY", 
  "qty": 0.01,
  "filled_qty": 0.01,
  "price": 43500.00,
  "timestamp": "2025-10-04T13:54:00Z"
}
```

### Market Data

#### Get Current Prices

Retrieve real-time price data for trading pairs.

```http
GET /api/prices?symbols=BTCUSDT,ETHUSDT
```

**Response:**
```json
{
  "BTCUSDT": {
    "price": 43500.00,
    "change_24h": 2.5,
    "volume_24h": 123456.78,
    "timestamp": "2025-10-04T13:54:00Z"
  },
  "ETHUSDT": {
    "price": 2650.00,
    "change_24h": -1.2,
    "volume_24h": 67890.12,
    "timestamp": "2025-10-04T13:54:00Z"
  }
}
```

#### Get Candle Data

Historical OHLCV data for charting and analysis.

```http
GET /api/candles?symbol=BTCUSDT&interval=1h&limit=100
```

**Parameters:**
- `symbol` (required): Trading pair symbol
- `interval` (optional): Timeframe ("1m", "5m", "15m", "1h", "4h", "1d") - default: "1h"
- `limit` (optional): Number of candles (default: 100, max: 1000)
- `start_time` (optional): Start timestamp (Unix milliseconds)
- `end_time` (optional): End timestamp (Unix milliseconds)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1h",
  "candles": [
    {
      "timestamp": 1696435200000,
      "open": 43400.00,
      "high": 43600.00,
      "low": 43200.00,
      "close": 43500.00,
      "volume": 123.45
    }
  ]
}
```

### AI Signals

#### Get Trading Signals

Retrieve AI-generated trading recommendations.

```http
GET /api/signals?symbol=BTCUSDT&timeframe=1h
```

**Response:**
```json
{
  "signals": [
    {
      "symbol": "BTCUSDT",
      "signal": "BUY",
      "confidence": 0.85,
      "entry_price": 43500.00,
      "target_price": 45000.00,
      "stop_loss": 42000.00,
      "reason": "Strong bullish momentum with RSI oversold",
      "timestamp": "2025-10-04T13:54:00Z"
    }
  ]
}
```

#### Generate New Signal

Request AI analysis for a specific trading pair.

```http
POST /api/signals/analyze
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "timeframe": "1h"
}
```

**Response:**
```json
{
  "analysis": {
    "symbol": "BTCUSDT",
    "signal": "BUY",
    "confidence": 0.78,
    "technical_indicators": {
      "rsi": 35.2,
      "macd": "bullish_crossover",
      "sma_20": 43200.00,
      "sma_50": 42800.00
    },
    "sentiment_score": 0.65,
    "news_sentiment": "positive"
  }
}
```

## 📈 Performance Monitoring

Access comprehensive performance metrics for monitoring and optimization.

### Request Metrics

```http
GET /api/metrics/requests
```

**Response:**
```json
{
  "total_requests": 1250,
  "avg_duration_ms": 45.2,
  "max_duration_ms": 250.8,
  "slow_requests": 3,
  "avg_db_queries": 1.2,
  "endpoint_performance": {
    "GET /api/trades": {
      "count": 450,
      "avg_ms": 25.5,
      "max_ms": 120.0
    }
  }
}
```

### Database Metrics

```http
GET /api/metrics/database
```

### System Metrics

```http
GET /api/metrics/system
```

## 🚨 Error Handling

The API uses standard HTTP status codes and returns consistent error responses.

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Trading pair INVALID not supported",
    "details": {
      "symbol": "INVALID",
      "supported_symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
    },
    "timestamp": "2025-10-04T13:54:00Z"
  }
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_SYMBOL` | Unsupported trading pair | 400 |
| `INSUFFICIENT_BALANCE` | Not enough balance for trade | 400 |
| `INVALID_QUANTITY` | Quantity outside allowed range | 400 |
| `API_KEY_REQUIRED` | Authentication required | 401 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `EXCHANGE_ERROR` | External exchange error | 502 |
| `INTERNAL_ERROR` | Server error | 500 |

## 📊 Rate Limits

API requests are rate-limited to ensure fair usage and system stability.

### Current Limits

- **General endpoints**: 100 requests per minute
- **Trading endpoints**: 50 requests per minute  
- **Market data**: 200 requests per minute
- **WebSocket connections**: 10 per IP address

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1696435260
```

## 🔌 WebSocket API

Real-time data streaming via WebSocket connections.

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

### Subscribe to Price Updates

```json
{
  "action": "subscribe",
  "channel": "prices",
  "symbols": ["BTCUSDT", "ETHUSDT"]
}
```

### Price Update Message

```json
{
  "channel": "prices",
  "data": {
    "symbol": "BTCUSDT",
    "price": 43500.00,
    "change": 2.5,
    "timestamp": "2025-10-04T13:54:00Z"
  }
}
```

## 🧪 Testing & Examples

### cURL Examples

**Get current prices:**
```bash
curl -X GET "http://localhost:8000/api/prices?symbols=BTCUSDT" \
  -H "Content-Type: application/json"
```

**Place a trade:**
```bash
curl -X POST "http://localhost:8000/api/trades" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "BUY",
    "qty": 0.01,
    "order_type": "MARKET"
  }'
```

### Python SDK Example

```python
import requests

class QuantumTraderAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_prices(self, symbols):
        response = self.session.get(
            f"{self.base_url}/api/prices",
            params={"symbols": ",".join(symbols)}
        )
        return response.json()
    
    def place_trade(self, symbol, side, qty, order_type="MARKET"):
        response = self.session.post(
            f"{self.base_url}/api/trades",
            json={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type
            }
        )
        return response.json()

# Usage
api = QuantumTraderAPI()
prices = api.get_prices(["BTCUSDT", "ETHUSDT"])
print(prices)
```

### JavaScript/TypeScript Example

```typescript
class QuantumTraderClient {
  constructor(private baseUrl = 'http://localhost:8000') {}

  async getPrices(symbols: string[]) {
    const response = await fetch(
      `${this.baseUrl}/api/prices?symbols=${symbols.join(',')}`,
      { headers: { 'Content-Type': 'application/json' } }
    );
    return response.json();
  }

  async placeTrade(trade: {
    symbol: string;
    side: 'BUY' | 'SELL';
    qty: number;
    order_type?: 'MARKET' | 'LIMIT';
  }) {
    const response = await fetch(`${this.baseUrl}/api/trades`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(trade)
    });
    return response.json();
  }
}
```

## 🔧 Configuration & Environment

### Environment Variables

```bash
# Required for live trading
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key

# Database configuration
QUANTUM_TRADER_DATABASE_URL=postgresql://user:pass@localhost/quantum_trader

# Optional features
ENABLE_METRICS=true
LOG_LEVEL=INFO
RATE_LIMIT_REQUESTS=100
```

### Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Database | SQLite | PostgreSQL |
| CORS | Permissive | Restricted |
| Logging | Console | File + Structured |
| Rate Limiting | Disabled | Enabled |
| API Keys | Mock/Demo | Real |

## 📚 Additional Resources

- **Interactive API Docs**: [http://localhost:8000/api/docs](http://localhost:8000/api/docs)
- **ReDoc Documentation**: [http://localhost:8000/api/redoc](http://localhost:8000/api/redoc)
- **OpenAPI Specification**: [http://localhost:8000/api/openapi.json](http://localhost:8000/api/openapi.json)
- **GitHub Repository**: [https://github.com/binyaminsemerci-ops/quantum_trader](https://github.com/binyaminsemerci-ops/quantum_trader)
- **Issue Tracker**: [GitHub Issues](https://github.com/binyaminsemerci-ops/quantum_trader/issues)

---

*For support or questions, please open an issue on GitHub or contact [support@quantumtrader.dev](mailto:support@quantumtrader.dev)*