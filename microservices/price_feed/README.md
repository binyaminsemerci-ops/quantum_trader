# Quantum Price Feed Service

WebSocket-basert price feed som publiserer real-time priser til Redis.

## Funksjon

Subscribes til Binance testnet mark price websocket og publiserer:
- `quantum:ticker:{symbol}` - Harvest Brain primary source
- `quantum:market:{symbol}` - Harvest Brain fallback

## Features

- ✅ Real-time mark prices (1 second updates)
- ✅ Auto-reconnect on disconnect
- ✅ Dynamic symbol loading from universe
- ✅ Low latency (< 100ms lag)
- ✅ Eliminates API rate limits

## Performance

| Metric | Value |
|--------|-------|
| Update frequency | 1 second |
| Symbols tracked | Dynamic (universe-based) |
| Redis TTL | 10 seconds |
| Memory usage | ~30 MB |
| CPU usage | ~2% |

## Usage

### Start Service
```bash
sudo systemctl start quantum-price-feed
```

### Check Status
```bash
sudo systemctl status quantum-price-feed
tail -f /var/log/quantum/price_feed.log
```

### Verify Prices
```bash
redis-cli hgetall quantum:ticker:BTCUSDT
```

## Configuration

Environment variables (optional):
- `REDIS_HOST` - Redis host (default: localhost)
- `REDIS_PORT` - Redis port (default: 6379)

## Integration

Harvest Brain automatically uses these price feeds:
1. Checks `quantum:ticker:{symbol}` (WebSocket feed)
2. Falls back to `quantum:market:{symbol}`
3. Falls back to direct API call (if both missing)

With this service, Harvest Brain will **never hit API rate limits** for price data.

## Monitoring

Statistics printed every 60 seconds:
```
📊 STATS: 52847 updates, 880.8 updates/sec, 0 errors, 19 symbols
```

## Dependencies

- Python packages: `redis`, `websockets`
- Already included in `/opt/quantum/venvs/ai-client-base`
