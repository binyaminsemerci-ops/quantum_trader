# AI Engine Service

**Service #3 of 7 - Sprint 2 Microservices Architecture**

## Purpose

AI inference brain for trading decisions: ensemble voting, meta-strategy selection, RL position sizing, and trade intent generation.

## Responsibilities

- **AI Model Inference:** 4-model ensemble (XGBoost, LightGBM, N-HiTS, PatchTST)
- **Ensemble Voting:** Weighted voting + consensus checking (3/4 agreement required)
- **Meta-Strategy Selection:** RL-based strategy selection (9 strategies)
- **RL Position Sizing:** Q-learning based position sizing + TP/SL optimization
- **Market Regime Detection:** Volatility + trend classification
- **Memory State Management:** 24-hour trading history memory
- **Model Supervision:** Bias detection (block if >70% SHORT/LONG bias)
- **Trade Intent Generation:** Final `ai.decision.made` event → execution-service

## Architecture

### Port
- **8001** (HTTP REST API)

### Events IN (Subscriptions)
- `market.tick` - Real-time price updates (main trigger)
- `market.klines` - Candle data for regime detection
- `trade.closed` - Trade outcomes for continuous learning
- `policy.updated` - Policy changes from risk-safety-service

### Events OUT (Publications)
- `ai.signal_generated` - Ensemble inference result (intermediate)
- `strategy.selected` - Meta-strategy selection (intermediate)
- `sizing.decided` - RL position sizing result (intermediate)
- **`ai.decision.made`** - **FINAL TRADE INTENT** (consumed by execution-service)

### REST API Endpoints

#### Health
- `GET /health` - Service health + component status

#### Signal Generation
- `POST /api/ai/signal` - Manual signal generation for a symbol

#### Metrics
- `GET /api/ai/metrics/ensemble` - Ensemble performance metrics
- `GET /api/ai/metrics/meta-strategy` - Meta-strategy performance
- `GET /api/ai/metrics/rl-sizing` - RL sizing performance

## Dependencies

### Internal (Sprint 1 Modules)
- **D1: PolicyStore** - Via risk-safety-service API (readonly)
- **D2: EventBus** - For event-driven communication

### External Services
- **risk-safety-service** (:8003) - PolicyStore queries
- **Redis** - EventBus backend
- **marketdata-service** (:8007) - Market data (future)

### AI Modules (Migrated)
- **ai_engine/ensemble_manager.py** (1224 lines) - 4-model ensemble
- **backend/services/ai/meta_strategy_selector.py** (676 lines) - Strategy selection
- **backend/services/ai/rl_position_sizing_agent.py** (882 lines) - Position sizing
- **backend/services/ai/regime_detector.py** - Market regime detection
- **backend/services/ai/memory_state_manager.py** - Memory state
- **backend/services/ai/model_supervisor.py** - Bias detection

## Configuration

See `config.py` for all settings. Key configs:

```python
# Ensemble
ENSEMBLE_MODELS = ["xgb", "lgbm", "nhits", "patchtst"]
ENSEMBLE_WEIGHTS = {"xgb": 0.25, "lgbm": 0.25, "nhits": 0.30, "patchtst": 0.20}
MIN_CONSENSUS = 3  # 3/4 models must agree

# Meta-Strategy
META_STRATEGY_ENABLED = True
META_STRATEGY_EPSILON = 0.10  # 10% exploration

# RL Sizing
RL_SIZING_ENABLED = True
RL_SIZING_EPSILON = 0.15  # 15% exploration

# Confidence thresholds
MIN_SIGNAL_CONFIDENCE = 0.65  # Block signals <65%
HIGH_CONFIDENCE_THRESHOLD = 0.85  # Flag high-confidence signals
```

## AI Pipeline Flow

```
1. Market Tick Event → ai-engine-service
   └─ market.tick with symbol + price

2. Ensemble Inference (4 models)
   └─ XGBoost, LightGBM, N-HiTS, PatchTST vote
   └─ Weighted voting: 25%, 25%, 30%, 20%
   └─ Consensus check: 3/4 models must agree
   └─ Output: action (BUY/SELL/HOLD) + confidence (0-1)

3. Confidence Filter
   └─ Reject if confidence < 0.65

4. Meta-Strategy Selection
   └─ Detect market regime (high_vol_trending, low_vol_ranging, etc.)
   └─ Select strategy using Q-learning (9 strategies available)
   └─ Output: strategy_id (aggressive, scalping, swing, etc.)

5. RL Position Sizing
   └─ Calculate position size based on:
      - Signal confidence
      - Market regime
      - Portfolio exposure
      - Historical performance (Q-table)
   └─ Output: position_size_usd, leverage, TP%, SL%

6. Trade Intent Generation
   └─ Build ai.decision.made event with:
      - symbol, side, confidence
      - entry_price, quantity, leverage
      - stop_loss, take_profit
      - model, strategy, regime metadata

7. Event Publication
   └─ ai.decision.made → EventBus → execution-service
```

## Ensemble Models

### 1. XGBoost (25% weight)
- **Type:** Gradient boosted trees
- **Strengths:** Feature interactions, non-linearity
- **Model:** `models/xgb_futures_model.joblib`

### 2. LightGBM (25% weight)
- **Type:** Gradient boosted trees (optimized)
- **Strengths:** Fast inference, sparse features
- **Model:** `models/lgbm_model.txt`

### 3. N-HiTS (30% weight)
- **Type:** Neural Hierarchical Interpolation for Time Series
- **Strengths:** Multi-rate temporal patterns, best for volatility
- **Model:** `models/nhits_model.pt`

### 4. PatchTST (20% weight)
- **Type:** Patch Time Series Transformer
- **Strengths:** Long-range dependencies, attention mechanism
- **Model:** `models/patchtst_model.pt`

### Voting Mechanism
- **Weighted average** of confidence scores
- **Consensus:** 3/4 models must vote same action (BUY/SELL)
- **Fallback:** HOLD if no consensus

## Meta-Strategy Selector

**9 Strategies:**
1. **Aggressive:** High leverage, tight stops, trending markets
2. **Scalping:** Quick in/out, low leverage, high frequency
3. **Swing:** Multi-day holds, moderate leverage
4. **Mean Revert:** Contrarian, range-bound markets
5. **Momentum:** Trend-following, breakout confirmation
6. **Breakout:** High volatility, strong directional moves
7. **Conservative:** Low leverage, wide stops, risk-averse
8. **Adaptive:** Dynamic adjustment based on conditions
9. **Default:** Balanced approach

**Selection Algorithm:**
- **Q-learning** with contextual states (symbol, regime, confidence)
- **Exploration:** 10% random strategy selection
- **Exploitation:** 90% best Q-value strategy
- **Learning:** Updates Q-table from trade outcomes (trade.closed events)

## RL Position Sizing

**Q-learning State Space:**
- Market regime (5 states)
- Signal confidence buckets (5 buckets)
- Portfolio exposure (4 states)

**Action Space:**
- Position size: 100 - 2000 USD (10 buckets)
- Leverage: 1x - 30x (5 buckets)
- Risk: 1% - 5% of portfolio (5 buckets)

**Reward Function:**
- R-multiple (PnL / risk)
- Win rate bonus
- Drawdown penalty

**TP/SL Optimization:**
- Uses Trading Mathematician (Kelly Criterion + ATR)
- Dynamic TP: 3% - 10% based on volatility
- Dynamic SL: 1% - 5% based on ATR

## Running the Service

### Local Development
```bash
cd microservices/ai_engine
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8001
```

### Docker
```bash
docker build -t ai-engine-service .
docker run -p 8001:8001 \
  -e REDIS_HOST="redis" \
  -v $(pwd)/../../models:/app/models \
  ai-engine-service
```

### Docker Compose (Microservices Stack)
```bash
cd quantum_trader
docker-compose --profile microservices up ai-engine
```

**Services Started:**
- redis (dependency)
- risk-safety (dependency)
- ai-engine (main service)

## Testing

```bash
# Health check
curl http://localhost:8001/health

# Generate signal
curl -X POST http://localhost:8001/api/ai/signal \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "timeframe": "5m", "include_reasoning": true}'

# Get ensemble metrics
curl http://localhost:8001/api/ai/metrics/ensemble
```

## Integration with Other Services

### execution-service (:8002)
- **Consumes:** `ai.decision.made` (main output event)
- **Flow:** ai-engine → trade intent → execution-service → order placement

### risk-safety-service (:8003)
- **PolicyStore queries:** Read risk limits, leverage caps
- **Flow:** ai-engine → HTTP GET /api/policy/{key} → risk-safety

### marketdata-service (:8007)
- **Provides:** market.tick, market.klines events
- **Flow:** marketdata → EventBus → ai-engine → inference

### portfolio-intelligence-service (:8004)
- **Provides:** Portfolio state (exposure, correlation)
- **Flow:** portfolio → EventBus → ai-engine → RL sizing considers exposure

## Known Limitations (MVP)

1. **Market data:** Currently uses placeholder data, needs integration with marketdata-service
2. **Continuous learning:** trade.closed event handler implemented but Q-table updates TODO
3. **Model retraining:** Continuous Learning Manager integration TODO
4. **HFOS orchestration:** AI-OS coordination TODO
5. **Shadow models:** Shadow model integration TODO

## Future Enhancements (Post-Sprint 2)

- [ ] WebSocket for real-time market data
- [ ] Automatic model retraining pipeline
- [ ] HFOS (AI-OS) orchestration layer
- [ ] Shadow model A/B testing
- [ ] Drift detection integration
- [ ] Covariate shift handling
- [ ] Multi-timeframe analysis
- [ ] Prometheus metrics exporter

## Sprint 2 Status

✅ **COMPLETE** (Service #3 of 7)
- [x] Boilerplate (main.py, config.py, models.py, service.py, api.py)
- [x] AI module integration (ensemble, meta-strategy, RL sizing)
- [x] Event handlers (market.tick → ai.decision.made)
- [x] Signal generation pipeline
- [x] REST API endpoints
- [x] Test suite (8 test cases)
- [x] Dockerfile
- [x] Documentation

**Next:** Service #4 (portfolio-intelligence-service)
