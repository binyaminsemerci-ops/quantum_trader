# OpportunityRanker Quick Reference

## 🚀 Quick Start

```bash
# 1. Start backend
python backend/main.py

# 2. Validate integration
python validate_opportunity_ranker.py

# 3. Test API
curl http://localhost:8000/opportunities/rankings
```

---

## 📡 API Endpoints

```bash
# Get all rankings
GET /opportunities/rankings

# Get top N (default: 10)
GET /opportunities/rankings/top?n=5&min_score=0.6

# Get specific symbol
GET /opportunities/rankings/BTCUSDT

# Get detailed breakdown
GET /opportunities/rankings/BTCUSDT/details

# Force refresh
POST /opportunities/refresh
```

---

## 🐍 Python API

```python
# Access ranker
ranker = app.state.opportunity_ranker

# Get all rankings
rankings = ranker.get_rankings()

# Get top opportunities
top_10 = ranker.get_top_opportunities(n=10, min_score=0.6)

# Get specific symbol
btc = ranker.get_ranking_for_symbol("BTCUSDT")

# Update weights
ranker.update_weights({
    "trend_strength": 0.30,
    "volatility_quality": 0.20,
    "liquidity": 0.20,
    "regime_compatibility": 0.15,
    "symbol_winrate": 0.10,
    "spread": 0.03,
    "noise": 0.02
})

# Manually trigger ranking
await ranker.rank_opportunities(["BTCUSDT", "ETHUSDT"])
```

---

## 📊 Metrics

| Metric | Weight | Description | Good | Poor |
|--------|--------|-------------|------|------|
| **Trend Strength** | 0.25 | ADX + Supertrend | >0.7 | <0.3 |
| **Volatility Quality** | 0.20 | ATR analysis | 0.6-0.8 | >0.9 or <0.1 |
| **Liquidity** | 0.15 | 24h volume | >$50M | <$10M |
| **Regime** | 0.15 | Market regime match | 1.0 | 0.0 |
| **Winrate** | 0.10 | Historical trades | >60% | <40% |
| **Spread** | 0.10 | Bid-ask spread | <0.05% | >0.2% |
| **Noise** | 0.05 | Price action quality | >0.7 | <0.3 |

**Overall Score = Σ(metric × weight)**

---

## 🎯 Score Interpretation

| Score | Grade | Interpretation |
|-------|-------|----------------|
| 0.8 - 1.0 | **A** | Excellent opportunity |
| 0.6 - 0.8 | **B** | Good opportunity |
| 0.4 - 0.6 | **C** | Moderate opportunity |
| 0.2 - 0.4 | **D** | Poor opportunity |
| 0.0 - 0.2 | **F** | Very poor opportunity |

---

## ⚙️ Environment Variables

```bash
# Enable/disable
QT_OPPORTUNITY_RANKER_ENABLED=true

# Refresh interval (seconds)
QT_OPPORTUNITY_REFRESH_INTERVAL=300

# Redis configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Binance credentials (optional for public data)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Integration thresholds
QT_MIN_OPPORTUNITY_SCORE=0.5        # Orchestrator filter
QT_STRATEGY_TOP_SYMBOLS=10           # Strategy Engine
QT_STRATEGY_MIN_SCORE=0.6            # Strategy Engine
```

---

## 🔌 Integration Examples

### Orchestrator (Trade Filtering)
```python
def should_allow_trade(self, symbol: str) -> bool:
    ranker = self.app_state.opportunity_ranker
    ranking = ranker.get_ranking_for_symbol(symbol)
    
    if ranking and ranking.overall_score < 0.5:
        logger.info(f"Trade blocked: {symbol} score={ranking.overall_score:.3f}")
        return False
    
    return True
```

### Strategy Engine (Symbol Selection)
```python
def get_active_symbols(self) -> list[str]:
    ranker = self.app_state.opportunity_ranker
    rankings = ranker.get_top_opportunities(n=10, min_score=0.6)
    return [r.symbol for r in rankings]
```

### MSC AI (Risk Adjustment)
```python
def adjust_risk_mode(self):
    ranker = self.app_state.opportunity_ranker
    high_quality = ranker.get_top_opportunities(n=20, min_score=0.7)
    
    if len(high_quality) >= 5:
        self.set_risk_mode("AGGRESSIVE")
    elif len(high_quality) >= 2:
        self.set_risk_mode("NORMAL")
    else:
        self.set_risk_mode("DEFENSIVE")
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `backend/services/opportunity_ranker.py` | Core module (700 lines) |
| `backend/clients/binance_market_data_client.py` | Binance integration |
| `backend/repositories/postgres_trade_log_repository.py` | PostgreSQL integration |
| `backend/stores/redis_opportunity_store.py` | Redis storage |
| `backend/integrations/opportunity_ranker_factory.py` | Factory & wiring |
| `backend/routes/opportunity_routes.py` | FastAPI routes |
| `backend/main.py` | Startup integration |
| `OPPORTUNITY_RANKER_COMPLETE.md` | Full documentation |

---

## 🧪 Testing

```bash
# Unit tests
python -m pytest backend/services/test_opportunity_ranker.py -v

# Integration validation
python validate_opportunity_ranker.py

# Manual API test
curl http://localhost:8000/opportunities/rankings | jq '.'
```

---

## 🔧 Troubleshooting

### OpportunityRanker not starting
```bash
# Check logs
grep "OpportunityRanker" logs/backend.log

# Verify dependencies
pip install redis ccxt pandas numpy

# Check environment
echo $QT_OPPORTUNITY_RANKER_ENABLED
```

### No rankings computed
```bash
# Wait for initialization (30s)
# Check Redis connection
redis-cli ping

# Force refresh
curl -X POST http://localhost:8000/opportunities/refresh
```

### API returning 404
```bash
# Verify route registration
grep "opportunity_routes" backend/main.py

# Check OPPORTUNITY_RANKER_AVAILABLE flag
# Restart backend
```

---

## 📈 Performance

- **Initialization:** ~30s
- **Single Symbol:** ~200-500ms
- **20 Symbols:** ~5-10s
- **API Response:** <100ms (cached)
- **Memory:** ~50MB (20 symbols)

---

## ✅ Checklist

- [ ] Backend starts successfully
- [ ] Logs show "OpportunityRanker: ENABLED"
- [ ] API endpoints respond
- [ ] Rankings computed (check `/rankings`)
- [ ] Top symbols identified (check `/rankings/top`)
- [ ] Redis storage working
- [ ] Refresh works (`POST /refresh`)
- [ ] Integration with Orchestrator (optional)
- [ ] Integration with Strategy Engine (optional)
- [ ] Integration with MSC AI (optional)

---

## 🎉 Success Indicators

```
[OK] OpportunityRanker integration available
[SEARCH] Initializing OpportunityRanker...
[OK] RegimeDetector loaded for OpportunityRanker
[OK] Initial rankings: 20 symbols
   #1: BTCUSDT = 0.753
   #2: ETHUSDT = 0.698
📊 OPPORTUNITY RANKER: ENABLED (refreshes every 300s)
[OK] OpportunityRanker API endpoints registered
```

---

## 📞 Support

- **Documentation:** `OPPORTUNITY_RANKER_COMPLETE.md`
- **Integration Guide:** `OPPORTUNITY_RANKER_INTEGRATION_GUIDE.md`
- **Architecture:** `OPPORTUNITY_RANKER_ARCHITECTURE.md`
- **Examples:** `backend/services/opportunity_ranker_example.py`

---

**Version:** 1.0.0  
**Status:** Production-Ready ✅  
**Last Updated:** 2025-06-15
