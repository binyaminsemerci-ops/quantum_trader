# MetricPack Builder v1 (P3.8.1)

**READ-ONLY Exit/Harvest Metrics Aggregator**

## Purpose

Consumes `quantum:stream:apply.result` to reconstruct trades and export Prometheus metrics:
- Realized PnL per symbol/regime/action
- Win rate for partial exits (PARTIAL_25/50/75) and FULL_CLOSE
- Expectancy and profit factor
- MFE/MAE in ATR units
- Time in trade distribution

## Architecture

```
apply.result stream → MetricPack Builder → Prometheus :8051
                           ↓
                    Trade Reconstruction
                           ↓
                    Regime Detection (ADX)
                           ↓
                    Metrics Export
```

## Audit-Safe Guarantees

✅ **Read-only** - No XADD to trading streams  
✅ **Idempotent** - Checkpoint + dedupe (plan_id, order_id)  
✅ **Deterministic** - Fixed PnL calculations  
✅ **Bounded memory** - LRU dedupe cache (10k entries)

## Metrics

### Counters
- `quantum_exit_realized_pnl_total{symbol, regime, action}`
- `quantum_exit_trades_total{symbol, regime, action, outcome}`
- `quantum_exit_events_processed_total`

### Gauges
- `quantum_exit_winrate{symbol, regime, action}`
- `quantum_exit_expectancy{symbol, regime}`
- `quantum_exit_profit_factor{symbol, regime}`
- `quantum_exit_builder_lag_seconds`

### Histograms
- `quantum_exit_time_in_trade_seconds{symbol, regime}`
- `quantum_exit_mfe_atr{symbol, regime}`
- `quantum_exit_mae_atr{symbol, regime}`

## Trade Reconstruction

1. **Entry**: First `executed=true` with `reduceOnly=false` → create TradeState
2. **Exits**: All `reduceOnly=true` events → add to exits[], calculate PnL
3. **Closed**: When `remaining_qty <= 0.001` → finalize and update metrics

## Regime Detection

- **trend**: ADX(14) > 25
- **chop**: ADX(14) < 20
- **unknown**: otherwise or insufficient data

Uses Binance testnet/mainnet futures klines API (read-only).

## Configuration

Environment variables in `/etc/quantum/metricpack-builder.env`:

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
METRICPACK_PORT=8051
METRICPACK_SYMBOLS=BTCUSDT,ETHUSDT,TRXUSDT
METRICPACK_MODE=testnet
APPLY_RESULT_STREAM=quantum:stream:apply.result
METRICPACK_CONSUMER_GROUP=metricpack_builder
```

## Deployment

See deployment commands in parent README or deployment scripts.

## Endpoints

- `GET /health` - Service status
- `GET /metrics` - Prometheus metrics (text format)

## Checkpoint

Last processed message ID stored in Redis key: `quantum:metricpack:last_id`

Restart resumes from checkpoint (no double-counting).

## Performance

- CPU: < 5% single core
- Memory: ~50-100 MB
- Latency: < 100ms per event
- Throughput: 100+ events/second
