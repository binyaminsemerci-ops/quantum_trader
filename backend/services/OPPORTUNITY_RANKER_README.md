# Opportunity Ranker (OppRank) Module

## Overview

The **Opportunity Ranker (OppRank)** is a sophisticated market-quality evaluation engine that continuously scans all available trading symbols and ranks them by opportunity score (0.0–1.0). It combines multiple quantitative metrics to identify high-edge trading opportunities while filtering out low-quality symbols.

## Purpose

OppRank serves as a **critical decision input** for multiple Quantum Trader components:

- **Strategy Runtime Engine**: Receives top-ranked symbols for strategy execution
- **Orchestrator Policy**: Uses rankings to allow/block trades per symbol
- **MSC AI**: Leverages rankings for global risk mode and strategy selection
- **Strategy Generator AI**: Uses rankings to focus strategy generation on high-quality symbols

By focusing trading activity on **objectively superior opportunities**, OppRank directly increases profitability and reduces noise trading.

---

## Architecture

### Core Components

```
OpportunityRanker (Main Class)
├─ Metric Calculators
│  ├─ Trend Strength (EMA, HH/HL, slope)
│  ├─ Volatility Quality (ATR, stability)
│  ├─ Liquidity Score (volume, depth)
│  ├─ Spread Score (bid-ask spread)
│  ├─ Symbol Winrate (historical performance)
│  ├─ Regime Compatibility (regime alignment)
│  └─ Noise Score (wick ratio, variance)
│
├─ Score Aggregation (weighted synthesis)
├─ Ranking Engine (sorting, filtering)
└─ Storage Interface (OpportunityStore)
```

### Dependencies

OppRank requires four protocol interfaces:

1. **MarketDataClient**: Provides OHLCV candles, spread, liquidity
2. **TradeLogRepository**: Historical trade performance per symbol
3. **RegimeDetector**: Current global market regime
4. **OpportunityStore**: Persistent storage for rankings

---

## Metrics Explained

### 1. Trend Strength (0–1)

**What it measures**: Strength and consistency of directional movement

**Components**:
- EMA alignment (price vs EMA 50 vs EMA 200)
- EMA slope (rate of change)
- Higher highs / higher lows consistency
- Trend quality over last 20 periods

**Score interpretation**:
- `0.8–1.0`: Strong, clean trend (ideal for trend-following)
- `0.5–0.7`: Moderate trend
- `0.0–0.4`: Weak or no trend (choppy)

### 2. Volatility Quality (0–1)

**What it measures**: Optimal volatility for trading (not too low, not too high)

**Optimal range**: 1.5%–8% daily ATR as percentage of price

**Components**:
- ATR percentage (normalized to price)
- Volatility stability (low variance = predictable)

**Score interpretation**:
- `0.8–1.0`: Optimal volatility (enough movement, not chaotic)
- `0.5–0.7`: Acceptable volatility
- `0.0–0.4`: Too low (no profit potential) or too high (too risky)

### 3. Liquidity Score (0–1)

**What it measures**: Ease of entry/exit without slippage

**Components**:
- 24h volume in USD
- Bid/ask depth

**Thresholds**:
- Minimum: $1M (poor liquidity)
- Optimal: $100M+ (excellent liquidity)

**Score interpretation**:
- `0.8–1.0`: Excellent liquidity (institutional-grade)
- `0.5–0.7`: Good liquidity
- `0.0–0.4`: Poor liquidity (risk of slippage)

### 4. Spread Score (0–1)

**What it measures**: Cost efficiency (bid-ask spread)

**Thresholds**:
- Optimal: ≤0.05% (5 basis points)
- Poor: ≥0.2% (20 basis points)

**Score interpretation**:
- `0.9–1.0`: Minimal spread (negligible cost)
- `0.6–0.8`: Acceptable spread
- `0.0–0.5`: High spread (eats into profits)

### 5. Symbol Winrate Score (0–1)

**What it measures**: Historical success rate for this symbol in your system

**Components**:
- Winrate from last 200 trades (or configurable N)
- Direct passthrough (0.65 winrate = 0.65 score)

**Score interpretation**:
- `0.7–1.0`: Historically profitable symbol
- `0.5–0.6`: Break-even to slight edge
- `0.0–0.4`: Historically unprofitable

### 6. Regime Compatibility Score (0–1)

**What it measures**: Alignment between symbol trend and global market regime

**Logic**:
```
If global regime = BULL:
  - Bullish symbol → 1.0 (perfect alignment)
  - Neutral symbol → 0.6
  - Bearish symbol → 0.3 (misalignment)

If global regime = BEAR:
  - Bearish symbol → 1.0
  - Neutral symbol → 0.6
  - Bullish symbol → 0.3

If global regime = CHOPPY:
  - Neutral symbol → 0.8
  - Trending symbol → 0.5
```

**Score interpretation**:
- `0.8–1.0`: Symbol trend matches global regime
- `0.5–0.7`: Partial alignment
- `0.0–0.4`: Opposing trend (counter-regime)

### 7. Noise Score (0–1)

**What it measures**: Inverse of market noise (cleanliness of price action)

**Components**:
- Wick-to-body ratio (large wicks = noise)
- Close-to-close variance (erratic = noise)
- High-low stability (consistency)

**Score interpretation**:
- `0.8–1.0`: Clean, predictable movement
- `0.5–0.7`: Moderate noise
- `0.0–0.4`: Very noisy (unpredictable)

---

## Score Aggregation

### Default Weights

```python
final_score = (
    0.25 * trend_strength +
    0.20 * volatility_quality +
    0.15 * liquidity_score +
    0.15 * regime_score +
    0.10 * symbol_winrate_score +
    0.10 * spread_score +
    0.05 * noise_score
)
```

### Custom Weights

You can override weights to match your trading style:

```python
# Example: Prioritize trend and liquidity
custom_weights = {
    'trend_strength': 0.40,
    'liquidity_score': 0.25,
    'volatility_quality': 0.15,
    'regime_score': 0.10,
    'symbol_winrate_score': 0.05,
    'spread_score': 0.03,
    'noise_score': 0.02,
}

ranker = OpportunityRanker(
    ...,
    weights=custom_weights
)
```

**Important**: Weights must sum to 1.0

---

## Usage Examples

### Basic Usage

```python
from opportunity_ranker import OpportunityRanker

# Initialize with dependencies
ranker = OpportunityRanker(
    market_data=your_market_data_client,
    trade_logs=your_trade_log_repo,
    regime_detector=your_regime_detector,
    opportunity_store=your_opportunity_store,
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"],
    timeframe="1h",
    candle_limit=200,
    min_score_threshold=0.5,  # Only include symbols >= 0.5
)

# Compute and store rankings
rankings = ranker.update_rankings()

# Output: {'BTCUSDT': 0.87, 'SOLUSDT': 0.82, 'ETHUSDT': 0.79}
```

### Get Top N Symbols

```python
# Get top 5 symbols for trading
top_symbols = ranker.get_top_n(n=5)

# Output: ['BTCUSDT', 'SOLUSDT', 'ETHUSDT', 'AVAXUSDT', 'BNBUSDT']
```

### Detailed Metrics Analysis

```python
# Get full metric breakdown
metrics_dict = ranker.compute_symbol_scores()

for symbol, metrics in metrics_dict.items():
    print(f"{symbol}:")
    print(f"  Trend Strength:     {metrics.trend_strength:.3f}")
    print(f"  Volatility Quality: {metrics.volatility_quality:.3f}")
    print(f"  Liquidity:          {metrics.liquidity_score:.3f}")
    print(f"  Final Score:        {metrics.final_score:.3f}")
```

### Periodic Updates

```python
import schedule

def update_opportunity_rankings():
    """Scheduled task to update rankings."""
    rankings = ranker.update_rankings()
    print(f"Updated rankings: {len(rankings)} symbols")

# Update every 15 minutes
schedule.every(15).minutes.do(update_opportunity_rankings)

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## Integration Guide

### Step 1: Implement Protocol Interfaces

```python
# Implement MarketDataClient
class BinanceMarketData:
    def get_latest_candles(self, symbol, timeframe, limit):
        # Fetch from Binance API
        ...
    
    def get_spread(self, symbol):
        # Get orderbook spread
        ...
    
    def get_liquidity(self, symbol):
        # Get 24h volume
        ...

# Implement TradeLogRepository
class PostgresTradeLog:
    def get_symbol_winrate(self, symbol, last_n=200):
        # Query database for historical trades
        ...

# Implement OpportunityStore
class RedisOpportunityStore:
    def update(self, rankings):
        # Store in Redis with TTL
        ...
    
    def get(self):
        # Retrieve from Redis
        ...
```

### Step 2: Wire into Backend

```python
# In your FastAPI backend
from opportunity_ranker import OpportunityRanker

app = FastAPI()

# Initialize ranker (singleton or dependency injection)
ranker = OpportunityRanker(
    market_data=BinanceMarketData(),
    trade_logs=PostgresTradeLog(),
    regime_detector=global_regime_detector,
    opportunity_store=RedisOpportunityStore(),
    symbols=config.TRADEABLE_SYMBOLS,
)

@app.get("/api/opportunities")
async def get_opportunities():
    """Endpoint to retrieve current opportunity rankings."""
    rankings = ranker.opportunity_store.get()
    return {"rankings": rankings}

@app.post("/api/opportunities/refresh")
async def refresh_opportunities():
    """Manually trigger ranking update."""
    rankings = ranker.update_rankings()
    return {"status": "success", "count": len(rankings)}
```

### Step 3: Schedule Updates

```python
# Background task
async def periodic_ranking_update():
    while True:
        try:
            rankings = ranker.update_rankings()
            logger.info(f"Updated rankings: {len(rankings)} symbols")
        except Exception as e:
            logger.error(f"Ranking update failed: {e}")
        
        await asyncio.sleep(900)  # 15 minutes

# Start on app startup
@app.on_event("startup")
async def start_background_tasks():
    asyncio.create_task(periodic_ranking_update())
```

### Step 4: Consume in Other Services

```python
# In Orchestrator
class OrchestratorPolicy:
    def should_allow_trade(self, signal):
        # Get current rankings
        rankings = opportunity_store.get()
        
        symbol_score = rankings.get(signal.symbol, 0.0)
        
        # Block trade if symbol score is too low
        if symbol_score < 0.5:
            return False, "Symbol opportunity score too low"
        
        return True, "Allowed"

# In Strategy Engine
class StrategyEngine:
    def get_active_symbols(self):
        # Only trade top 10 symbols
        rankings = opportunity_store.get()
        return list(rankings.keys())[:10]
```

---

## Configuration

### Recommended Settings

| Use Case | Timeframe | Candle Limit | Min Threshold | Update Frequency |
|----------|-----------|--------------|---------------|------------------|
| Day Trading | 15m | 200 | 0.6 | 5 minutes |
| Swing Trading | 1h | 200 | 0.5 | 15 minutes |
| Position Trading | 4h | 200 | 0.4 | 1 hour |
| High-Frequency | 5m | 100 | 0.7 | 1 minute |

### Tuning Weights

**Conservative (low risk)**:
```python
weights = {
    'liquidity_score': 0.30,    # Prioritize liquidity
    'spread_score': 0.20,       # Low costs
    'volatility_quality': 0.20, # Stable volatility
    'trend_strength': 0.15,
    'regime_score': 0.08,
    'symbol_winrate_score': 0.05,
    'noise_score': 0.02,
}
```

**Aggressive (high risk/reward)**:
```python
weights = {
    'trend_strength': 0.35,     # Strong trends
    'volatility_quality': 0.25, # High movement
    'regime_score': 0.15,
    'symbol_winrate_score': 0.10,
    'liquidity_score': 0.10,
    'spread_score': 0.03,
    'noise_score': 0.02,
}
```

---

## Testing

Run unit tests:

```bash
python -m pytest test_opportunity_ranker.py -v
```

Run example demonstrations:

```bash
python opportunity_ranker_example.py
```

---

## Performance Considerations

### Computational Complexity

- **Per symbol**: O(n) where n = candle_limit (typically 200)
- **Total**: O(m × n) where m = number of symbols
- **Typical runtime**: ~0.1–0.5 seconds per symbol
- **For 50 symbols**: ~5–25 seconds total

### Optimization Tips

1. **Parallel processing** (if needed for 100+ symbols):
   ```python
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=10) as executor:
       futures = [executor.submit(compute_metrics, symbol) for symbol in symbols]
       results = [f.result() for f in futures]
   ```

2. **Cache candle data** (if updating frequently)
3. **Use async I/O** for market data fetching
4. **Store in Redis** with 15-minute TTL

---

## Monitoring

### Key Metrics to Track

- Number of symbols passing threshold
- Average opportunity score
- Top symbol stability (how often top 5 changes)
- Distribution of scores (histogram)
- Correlation between score and actual trade profitability

### Logging

```python
logger.info(f"Rankings updated: {len(rankings)} symbols passed threshold")
logger.info(f"Top 3: {list(rankings.keys())[:3]}")
logger.info(f"Average score: {np.mean(list(rankings.values())):.3f}")
```

---

## Limitations & Future Enhancements

### Current Limitations

1. **No multi-timeframe analysis** (only single timeframe)
2. **No correlation filtering** (symbols may be correlated)
3. **No sector/category awareness**
4. **Static weights** (not adaptive to regime)

### Planned Enhancements

1. **Multi-timeframe scoring**: Combine 15m, 1h, 4h scores
2. **Correlation matrix**: Penalize highly correlated symbols
3. **Sector rotation**: Boost trending sectors
4. **Adaptive weights**: ML model to optimize weights per regime
5. **Symbol scoring history**: Track score evolution over time
6. **Anomaly detection**: Flag unusual market conditions per symbol

---

## FAQ

**Q: How often should I update rankings?**  
A: For most strategies, 10–15 minutes is optimal. More frequent updates waste compute; less frequent may miss regime shifts.

**Q: Should I trade all symbols with score > threshold?**  
A: Not necessarily. Use OppRank to filter the universe, then let Strategy Engine decide per-signal.

**Q: Can I use OppRank for crypto and stocks?**  
A: Yes, but calibrate thresholds (volatility, spread, liquidity) to asset class.

**Q: What if no symbols pass threshold?**  
A: Lower threshold or sit on sidelines. Low scores = poor market conditions.

**Q: How do I handle new symbols?**  
A: They'll have no historical winrate. Set default to 0.5 (neutral) in TradeLogRepository.

---

## Support & Contribution

For questions or improvements, contact the Quantum Trader development team.

---

**Module Version**: 1.0.0  
**Last Updated**: November 30, 2024  
**Author**: Quantum Trader Development Team
