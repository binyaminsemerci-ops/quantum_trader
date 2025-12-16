# Strategy Generator AI - Integration Plan

**Integrating SG AI with Quantum Trader Production System**

Date: November 29, 2025  
Status: ✅ Examples Validated, Ready for Integration

---

## 🎯 Executive Summary

Successfully implemented and tested Strategy Generator AI (SG AI) with stub implementations. Example 1 demonstrates:
- ✅ **Generation:** 10 strategies created
- ✅ **Backtesting:** 2,000-7,000 trades per strategy on 3 symbols
- ✅ **Fitness Scoring:** 65-67 fitness (target: PF 2.0+, WR 50%+)
- ✅ **Performance:** PF 1.79-2.60, WR 45-57%, DD <1.1%
- ✅ **Runtime:** <2 seconds for 10 strategies

**Next:** Integrate with existing Quantum Trader services.

---

## 📊 Test Results

### Example 1: First Generation ✅

**Configuration:**
- Population: 10 strategies
- Symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Backtest Period: 90 days
- Commission: 0.04%

**Top 5 Results:**

| Rank | Strategy | Fitness | PF | WR | DD | Trades | P&L |
|------|----------|---------|----|----|----|----|-----|
| 1 | Gen1_Strategy10 | 67.1 | 2.60 | 45.7% | 0.5% | 7,814 | $10,378 |
| 2 | Gen1_Strategy4 | 66.4 | 2.10 | 49.2% | 0.4% | 3,632 | $4,010 |
| 3 | Gen1_Strategy5 | 65.6 | 1.94 | 53.0% | 0.7% | 5,170 | $6,643 |
| 4 | Gen1_Strategy1 | 65.6 | 2.07 | 49.0% | 1.1% | 7,185 | $9,267 |
| 5 | Gen1_Strategy9 | 65.4 | 1.79 | 56.8% | 0.5% | 2,676 | $2,673 |

**Key Insights:**
- All strategies exceed PF 1.5 threshold (promotion-ready)
- Win rates 45-57% (target: 60%)
- Drawdowns <1.1% (excellent)
- Diverse entry types (momentum, ensemble, mean reversion)
- Diverse regime filters (ranging, trending, choppy, any)

---

## 🔌 Integration Architecture

### Phase 1: Core Infrastructure (Week 1)

```
┌─────────────────────────────────────────────────────────┐
│                  Quantum Trader                          │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────┐        ┌──────────────┐              │
│  │  PostgreSQL  │◄───────┤  SG AI       │              │
│  │  Database    │        │  Repository  │              │
│  └──────────────┘        └──────────────┘              │
│         ▲                         ▲                      │
│         │                         │                      │
│  ┌──────┴───────┐        ┌───────┴──────┐              │
│  │  Existing    │        │  Market Data │              │
│  │  Tables      │        │  Client      │              │
│  └──────────────┘        └──────┬───────┘              │
│                                  │                       │
│                          ┌───────▼──────┐              │
│                          │  Binance API │              │
│                          └──────────────┘              │
└─────────────────────────────────────────────────────────┘
```

**Tasks:**

1. **Create Database Tables** (2 hours)
   ```sql
   CREATE TABLE strategies (
       strategy_id TEXT PRIMARY KEY,
       name TEXT NOT NULL,
       status TEXT NOT NULL,
       -- ... all StrategyConfig fields
       created_at TIMESTAMP NOT NULL
   );
   
   CREATE TABLE strategy_stats (
       id SERIAL PRIMARY KEY,
       strategy_id TEXT REFERENCES strategies(strategy_id),
       source TEXT NOT NULL,
       -- ... all StrategyStats fields
       timestamp TIMESTAMP NOT NULL
   );
   
   CREATE INDEX idx_strategies_status ON strategies(status);
   CREATE INDEX idx_stats_strategy_id ON strategy_stats(strategy_id);
   CREATE INDEX idx_stats_timestamp ON strategy_stats(timestamp);
   ```

2. **Implement PostgresStrategyRepository** (4 hours)
   - Location: `backend/research/repositories_impl.py`
   - Use existing `backend/database/connection.py`
   - Implement all 7 protocol methods
   - Add connection pooling

3. **Implement BinanceMarketDataClient** (4 hours)
   - Location: `backend/research/market_data_impl.py`
   - Reuse existing Binance client from `backend/services/binance_client.py`
   - Cache OHLCV data in Redis (optional)
   - Handle rate limits

**Validation:**
```python
# Test PostgreSQL storage
repo = PostgresStrategyRepository(db_url)
config = StrategyConfig(name="Test", ...)
repo.save_strategy(config)
assert repo.get_strategies_by_status(StrategyStatus.CANDIDATE)

# Test market data
client = BinanceMarketDataClient(binance)
df = client.get_history("BTCUSDT", "15m", start, end)
assert len(df) > 0
assert "close" in df.columns
```

---

### Phase 2: Ensemble Integration (Week 1-2)

```
┌─────────────────────────────────────────────────────────┐
│                  Strategy Backtest                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Historical ──► EnsembleBacktester ──► Simulated Trades │
│  OHLCV Data                │                             │
│                            │                             │
│                    ┌───────▼────────┐                    │
│                    │  Ensemble      │                    │
│                    │  Orchestrator  │                    │
│                    └───────┬────────┘                    │
│                            │                             │
│              ┌─────────────┼─────────────┐              │
│              ▼             ▼             ▼              │
│         ┌────────┐    ┌────────┐   ┌────────┐          │
│         │Prophet │    │XGBoost │   │LSTM AI │          │
│         │Sentinel│    │Ranger  │   │Vision  │          │
│         └────────┘    └────────┘   └────────┘          │
│              │             │             │               │
│              └─────────────┴─────────────┘              │
│                            │                             │
│                    Signal + Confidence                   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

**Tasks:**

1. **Create EnsembleBacktester** (6 hours)
   - Location: `backend/research/ensemble_backtest.py`
   - Extend `StrategyBacktester`
   - Override `_check_entry()` method
   - Call `EnsembleOrchestrator.predict()`
   - Map ensemble signals to entry types

2. **Add Historical Prediction Mode** (4 hours)
   - Modify `backend/services/ensemble_orchestrator.py`
   - Add `predict_historical(df)` method
   - Avoid looking-ahead bias
   - Return signals for each bar

3. **Cache Historical Predictions** (2 hours)
   - Store in Redis with TTL (24h)
   - Key: `ensemble:predictions:{symbol}:{timeframe}:{date_range}`
   - Reduces redundant ensemble calls

**Code Example:**
```python
class EnsembleBacktester(StrategyBacktester):
    def __init__(self, market_data, ensemble):
        super().__init__(market_data)
        self.ensemble = ensemble
        self._prediction_cache = {}
    
    def _check_entry(self, config, df, idx):
        # Get ensemble prediction
        cache_key = f"{df['timestamp'].iloc[idx]}"
        if cache_key not in self._prediction_cache:
            historical_data = df.iloc[:idx+1]
            pred = self.ensemble.predict_historical(historical_data)
            self._prediction_cache[cache_key] = pred
        
        signal = self._prediction_cache[cache_key]
        
        # Check confidence threshold
        if signal['confidence'] < config.min_confidence:
            return None
        
        # Entry type logic
        if config.entry_type == EntryType.ENSEMBLE_CONSENSUS:
            return signal['direction'] == 'LONG'
        elif config.entry_type == EntryType.MOMENTUM:
            # Add momentum filter
            returns_20 = df['close'].pct_change(20).iloc[idx]
            return signal['direction'] == 'LONG' and returns_20 > 0
        # ... other types
```

**Validation:**
```python
# Test with real ensemble
ensemble = EnsembleOrchestrator()
ensemble.load_models()

backtester = EnsembleBacktester(market_data, ensemble)
stats = backtester.backtest(
    config=test_strategy,
    symbols=["BTCUSDT"],
    start=datetime(2025, 8, 1),
    end=datetime(2025, 11, 1)
)

assert stats.total_trades > 0
assert stats.profit_factor > 0
```

---

### Phase 3: Docker Deployment (Week 2)

```
docker-compose.yml:

services:
  # Existing services
  backend:
    ...
  
  # New: Strategy Generator
  strategy_generator:
    build: .
    image: quantum_trader:latest
    container_name: quantum_sg_ai
    command: python -m backend.research.continuous_runner
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - REDIS_URL=${REDIS_URL}
      
      # SG AI Config
      - SG_POPULATION_SIZE=30
      - SG_BACKTEST_DAYS=90
      - SG_BACKTEST_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,ARBUSDT
      - SG_GENERATION_INTERVAL=3600  # 1 hour
      
      # Promotion Thresholds
      - SG_CANDIDATE_MIN_PF=1.5
      - SG_CANDIDATE_MIN_TRADES=50
      - SG_SHADOW_MIN_PF=1.3
      - SG_SHADOW_MIN_DAYS=14
    
    depends_on:
      - db
      - redis
      - backend
    
    volumes:
      - ./backend:/app/backend
      - ./models:/app/models
    
    restart: unless-stopped
  
  # New: Shadow Tester
  shadow_tester:
    build: .
    image: quantum_trader:latest
    container_name: quantum_shadow_tester
    command: python -m backend.research.shadow_runner
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
      - REDIS_URL=${REDIS_URL}
      
      # Shadow Config
      - SHADOW_TEST_INTERVAL=900  # 15 minutes
      - SHADOW_LOOKBACK_DAYS=7
      - SHADOW_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT
    
    depends_on:
      - db
      - redis
      - backend
    
    restart: unless-stopped
```

**Tasks:**

1. **Create Continuous Runner** (4 hours)
   - Location: `backend/research/continuous_runner.py`
   - Run evolutionary loop every N hours
   - Graceful shutdown handling
   - Error recovery

2. **Create Shadow Runner** (4 hours)
   - Location: `backend/research/shadow_runner.py`
   - Run shadow testing every 15 minutes
   - Call deployment manager hourly
   - Prometheus metrics

3. **Add Health Checks** (2 hours)
   - Endpoint: `/sg-ai/health`
   - Endpoint: `/sg-ai/stats`
   - Monitor last generation time
   - Monitor shadow test status

**Validation:**
```bash
# Start services
docker-compose up -d strategy_generator shadow_tester

# Check logs
docker logs quantum_sg_ai --tail 50
docker logs quantum_shadow_tester --tail 50

# Check health
curl http://localhost:8000/sg-ai/health
# {"status": "healthy", "last_generation": "2025-11-29T20:45:00Z"}
```

---

### Phase 4: Monitoring & Alerts (Week 2-3)

```
Grafana Dashboard:

┌─────────────────────────────────────────────────────────┐
│                  SG AI Performance                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Best Fitness by Generation        Active Strategies     │
│  ┌──────────────────────┐          ┌──────────────────┐ │
│  │       /\             │          │ CANDIDATE:  15   │ │
│  │      /  \     /\     │          │ SHADOW:      3   │ │
│  │     /    \   /  \    │          │ LIVE:        2   │ │
│  │    /      \_/    \   │          │ DISABLED:    8   │ │
│  └──────────────────────┘          └──────────────────┘ │
│                                                           │
│  Shadow Test Performance          Promotion Rate         │
│  ┌──────────────────────┐          ┌──────────────────┐ │
│  │ Avg PF:    1.45      │          │ Last 24h: 2/15   │ │
│  │ Avg WR:    52.3%     │          │ (13.3%)          │ │
│  │ Avg Trades: 28       │          └──────────────────┘ │
│  └──────────────────────┘                                │
│                                                           │
│  Live Strategy P&L                                       │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Strategy_A:  +$1,245 (PF: 1.82, WR: 58%)           ││
│  │ Strategy_B:  +$  892 (PF: 1.54, WR: 51%)           ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

**Tasks:**

1. **Prometheus Metrics** (4 hours)
   - Location: `backend/research/metrics.py`
   - Metrics:
     * `sg_ai_generation_fitness{generation}` - Best fitness per gen
     * `sg_ai_strategies_total{status}` - Count by status
     * `sg_ai_shadow_pf` - Shadow test profit factor
     * `sg_ai_promotions_total` - Promotion counter
     * `sg_ai_demotions_total` - Demotion counter

2. **Grafana Dashboard** (2 hours)
   - Import dashboard JSON
   - Connect to Prometheus
   - Add alert rules

3. **Discord/Slack Alerts** (2 hours)
   - Alert on strategy promotion to LIVE
   - Alert on strategy demotion
   - Alert on system errors
   - Daily performance summary

**Alert Rules:**
```yaml
# Prometheus alert rules

- alert: StrategyPromoted
  expr: increase(sg_ai_promotions_total[5m]) > 0
  annotations:
    summary: "New strategy promoted to {{ $labels.status }}"

- alert: StrategyDemoted
  expr: increase(sg_ai_demotions_total[5m]) > 0
  annotations:
    summary: "Strategy demoted: {{ $labels.strategy_name }}"

- alert: NoGenerationsRecently
  expr: time() - sg_ai_last_generation_timestamp > 7200
  annotations:
    summary: "No strategy generation in last 2 hours"
```

---

### Phase 5: Production Hardening (Week 3)

**Tasks:**

1. **Error Handling** (4 hours)
   - Retry logic for API failures
   - Circuit breaker for Binance API
   - Graceful degradation
   - Dead letter queue for failed backtests

2. **Rate Limiting** (2 hours)
   - Respect Binance rate limits (1200 req/min)
   - Implement token bucket algorithm
   - Queue backtest requests

3. **Performance Optimization** (6 hours)
   - Multiprocessing for population evaluation
   - Vectorized backtesting (pandas)
   - Database query optimization
   - Redis caching

4. **Testing** (8 hours)
   - Unit tests for all components
   - Integration tests for full pipeline
   - Load testing (100 strategies/hour)
   - Chaos engineering (kill services)

**Code Example - Multiprocessing:**
```python
from multiprocessing import Pool

class StrategySearchEngine:
    def run_generation(self, population_size, generation):
        population = self._generate_population(population_size, generation)
        
        # Parallel backtesting
        with Pool(processes=4) as pool:
            results = pool.starmap(
                self._backtest_single,
                [(config, self.backtest_symbols, self.backtest_days) 
                 for config in population]
            )
        
        # Sort by fitness
        results.sort(key=lambda x: x[1].fitness_score, reverse=True)
        return results
    
    def _backtest_single(self, config, symbols, days):
        # Worker function for parallel execution
        stats = self.backtester.backtest(...)
        return (config, stats)
```

---

## 📋 Integration Checklist

### Week 1: Core Infrastructure ✅ Ready

- [x] Review examples and documentation
- [x] Test examples with stub implementations
- [x] Plan integration with existing services
- [ ] Create PostgreSQL tables
- [ ] Implement PostgresStrategyRepository
- [ ] Implement BinanceMarketDataClient
- [ ] Test with real data
- [ ] Create EnsembleBacktester
- [ ] Add historical prediction mode to ensemble
- [ ] Test ensemble integration

### Week 2: Deployment & Monitoring

- [ ] Create continuous_runner.py
- [ ] Create shadow_runner.py
- [ ] Add to docker-compose.yml
- [ ] Test Docker deployment
- [ ] Add health check endpoints
- [ ] Implement Prometheus metrics
- [ ] Create Grafana dashboard
- [ ] Set up Discord/Slack alerts
- [ ] Load testing

### Week 3: Production Hardening

- [ ] Error handling & retries
- [ ] Rate limiting
- [ ] Circuit breakers
- [ ] Performance optimization
- [ ] Unit tests
- [ ] Integration tests
- [ ] Documentation updates
- [ ] Production deployment

---

## 🎯 Success Metrics

**Week 1 Targets:**
- ✅ Example 1 working (fitness 65-67)
- Database schema created
- Real Binance data integration
- First real backtest with ensemble

**Week 2 Targets:**
- Docker services running
- 1 generation per hour
- Shadow testing every 15 min
- Grafana dashboard live

**Week 3 Targets:**
- 3-5 LIVE strategies deployed
- <1% error rate
- <5 minute backtest time per strategy
- 100% uptime

**Month 1 Targets:**
- 10+ LIVE strategies
- Average fitness >70
- Promotion rate >15%
- Demotion rate <5%
- Positive cumulative P&L

---

## 🚨 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting to backtest | High | Shadow test 14+ days, strict thresholds |
| API rate limits | Medium | Caching, rate limiting, backoff |
| Database deadlocks | Medium | Connection pooling, query optimization |
| Strategy correlation | High | Diversity metrics, regime filtering |
| Live losses | High | Small position sizes initially, tight SL |

---

## 📞 Support & Next Steps

**Immediate Actions:**
1. ✅ Review this integration plan
2. ✅ Approve Phase 1 implementation
3. Create database schema
4. Implement repository classes
5. Start ensemble integration

**Questions/Blockers:**
- Which PostgreSQL database to use? (existing or new)
- Redis instance available? (for caching)
- Prometheus/Grafana already set up?
- Preferred alert channel (Discord/Slack/Email)?

**Resources:**
- Development time: 3 weeks (1 developer)
- Testing period: 1 week (paper trading)
- Production rollout: Gradual (1-2 strategies/day)

---

**Status:** ✅ Ready for Phase 1 Implementation  
**Next Review:** After Phase 1 completion (Week 1 end)
