# AI/ML Integration Guide

## Overview

Quantum Trader now includes comprehensive AI/scikit-learn integration for intelligent position management. The AI engine uses XGBoost ensemble models with 100+ technical indicators to predict optimal entry/exit points with confidence scoring.

## Architecture

### Core Components

1. **AI Agent** (`ai_engine/agents/xgb_agent.py`)
   - XGBoost-based prediction with ensemble support
   - 100+ technical indicators via feature engineering
   - BUY/SELL/HOLD classification with confidence scores
   - Fallback to EMA heuristics when model unavailable

2. **AI Trading Engine** (`backend/services/ai_trading_engine.py`)
   - Position-aware signal processing
   - Confidence-based position sizing (0.5x-1.5x multiplier)
   - Execution outcome tracking for continuous learning
   - Performance metrics and model retraining infrastructure

3. **Selection Engine** (`backend/services/selection_engine.py`)
   - Blends liquidity metrics with AI predictions
   - Confidence-weighted scoring
   - Signal distribution tracking (BUY/SELL/HOLD percentages)

4. **Execution Service** (`backend/services/execution.py`)
   - AI-adjusted order sizing
   - Signal-based intent filtering
   - Automatic outcome recording for model improvement

## AI Signal Processing

### Prediction Logic

```python
# XGBoost returns float prediction:
if prediction > 0.01:  action = "BUY"
elif prediction < -0.01:  action = "SELL"
else:  action = "HOLD"
```

### Position-Aware Adjustments

The AI engine considers current positions when generating signals:

- **Already LONG + AI says BUY**: Requires confidence > 0.7 to add position
- **Already LONG + AI says SELL**: Confidence > 0.6 triggers position close
- **No position + AI says BUY**: Standard confidence threshold (0.5)
- **SHORT positions**: Similar logic with reversed signals

### Confidence-Based Sizing

```python
size_multiplier = 0.5 + confidence * 1.0
# Clamped to range [0.3, 2.0]
```

Examples:
- Confidence 0.4 → 0.9x size (reduce risk)
- Confidence 0.6 → 1.1x size (standard)
- Confidence 0.9 → 1.4x size (high conviction)

## Integration Flow

### Liquidity Selection Phase

1. `scan_top_by_volume_from_api()` fetches top symbols by volume
2. For each symbol, AI agent predicts BUY/SELL/HOLD with confidence
3. Signals blended with liquidity metrics:
   ```python
   final_score = liquidity_weight * volume_component + 
                 model_weight * (ai_score * confidence)
   ```
4. Top symbols selected for portfolio allocation

### Execution Phase

1. `run_portfolio_rebalance()` generates order intents from allocations
2. AI engine processes intents with current positions
3. Size multipliers applied based on AI confidence
4. Conflicting signals filtered (e.g., AI says SELL but intent is BUY)
5. Orders submitted with adjusted quantities
6. Outcomes recorded for continuous learning

## API Endpoints

### Get AI Status
```
GET /ai/live-status
X-Admin-Token: <your-token>
```

Returns:
```json
{
  "status": "active",
  "model": {
    "loaded": true,
    "type": "xgboost",
    "feature_count": 100,
    "ensemble_available": false
  },
  "predictions": {
    "total": 37,
    "buy_signals": 5,
    "sell_signals": 8,
    "hold_signals": 24,
    "timestamp": "2024-11-12T10:30:00"
  },
  "recent_signals": [
    {
      "symbol": "SOLUSDC",
      "weight": 0.15,
      "score": 0.82,
      "reason": "AI=BUY(0.75)x1.25; momentum=0.68"
    }
  ]
}
```

### Get AI Predictions
```
POST /ai/predict
X-Admin-Token: <your-token>
{
  "symbols": ["BTCUSDC", "ETHUSDC"]
}
```

### Trigger Model Retraining
```
POST /ai/retrain
X-Admin-Token: <your-token>
{
  "lookback_days": 30,
  "min_samples": 1000
}
```

## Configuration

### Enable AI Scoring

Edit `backend/config/liquidity.py`:
```python
LIQUIDITY_MODEL_WEIGHT = 0.5  # 50% AI, 50% liquidity metrics
LIQUIDITY_VOLUME_WEIGHT = 0.5
```

Higher `MODEL_WEIGHT` → More AI-driven selection  
Higher `VOLUME_WEIGHT` → More liquidity-driven selection

### AI Model Path

```bash
QT_AI_MODEL_PATH=ai_engine/models/xgb_model.pkl
QT_AI_SCALER_PATH=ai_engine/models/scaler.pkl
```

### Feature Engineering

```bash
QT_AI_USE_ADVANCED_FEATURES=1  # Enable 100+ indicators
QT_AI_USE_ENSEMBLE=1           # Use 6-model ensemble (requires catboost)
```

## Monitoring

### Signal Distribution

Check logs for AI signal breakdown:
```
INFO: AI signals: BUY=5 (13.5%), SELL=8 (21.6%), HOLD=24 (64.9%)
```

### Execution Adjustments

```
INFO: AI adjusted BTCUSDC: qty=0.1000->0.1250, mult=1.25
INFO: AI skipping BUY intent for ETHUSDC: AI=SELL confidence=0.85
```

### Performance Tracking

AI engine tracks prediction accuracy:
- Predicted action vs actual outcome
- Entry price vs exit price (for closed positions)
- Win rate by confidence bucket

## Continuous Learning

### Automatic Outcome Recording

Every executed order records:
- AI predicted action + confidence
- Actual executed side
- Entry price and timestamp

### Model Retraining (TODO)

Planned workflow:
1. Collect outcomes from last N days
2. Fetch OHLCV data for executed symbols
3. Generate features using `compute_all_indicators`
4. Label: BUY/SELL/HOLD based on actual P&L
5. Retrain XGBoost with new samples
6. Validate on holdout set
7. Replace live model if improved

## Current Status

### What's Working ✅

- XGBoost model loaded and active
- AI signals integrated in liquidity selection
- Position-aware signal processing
- Confidence-based position sizing
- Execution outcome tracking
- API endpoints for monitoring

### Known Issues ⚠️

1. **All signals showing HOLD**: Model may need retraining with live data
   - Current model trained on historical data
   - Real-time market conditions differ
   - Solution: Collect outcomes and retrain

2. **Ensemble unavailable**: Missing catboost dependency
   - Fallback to single XGBoost model works
   - Solution: `pip install catboost` (optional)

3. **High margin usage**: Existing SOLUSDC position uses ~4688 USDC
   - Limits available capital for new AI-driven trades
   - Solution: Close or reduce position to free margin

## Next Steps

### Immediate Actions

1. **Test AI Integration**
   ```powershell
   # Close existing positions to free margin
   # (Manual via Binance dashboard or API)
   
   # Trigger liquidity refresh to get fresh AI signals
   .\backend\trigger_liquidity.ps1
   
   # Run execution with AI adjustments
   .\backend\trigger_execution.ps1
   ```

2. **Monitor AI Status**
   ```powershell
   curl http://localhost:8000/ai/live-status `
     -H "X-Admin-Token: your-secret-admin-token"
   ```

3. **Check Signal Distribution**
   ```powershell
   # View backend logs for AI signal breakdown
   # Look for lines starting with "INFO: AI signals:"
   ```

### Medium-Term Improvements

1. **Collect Training Data**: Run live for 1-2 weeks to accumulate outcomes
2. **Retrain Model**: Use actual P&L to improve predictions
3. **A/B Testing**: Compare AI-adjusted vs non-AI performance
4. **Feature Tuning**: Analyze which indicators are most predictive

### Long-Term Goals

1. **Multi-timeframe Analysis**: Combine 1m, 5m, 15m predictions
2. **Sentiment Integration**: Use Twitter/news sentiment (already in agent)
3. **Reinforcement Learning**: Learn optimal sizing from portfolio returns
4. **Risk-Adjusted Predictions**: Factor in volatility and correlation

## Troubleshooting

### AI Shows All HOLD Signals

**Cause**: Model not sufficiently trained or features too weak

**Solutions**:
- Check model file exists: `ai_engine/models/xgb_model.pkl`
- Verify scaler file: `ai_engine/models/scaler.pkl`
- Review training data quality in model logs
- Consider retraining with recent market data

### Size Multipliers Not Applied

**Cause**: Confidence too low or signals conflict with intents

**Solutions**:
- Check AI confidence scores in logs
- Verify `MODEL_WEIGHT` > 0 in liquidity config
- Review intent reasons for "AI=" prefix
- Ensure `get_trading_signals` called in execution

### Performance Metrics Empty

**Cause**: No execution outcomes recorded yet

**Solutions**:
- Run at least one execution cycle with orders
- Check ExecutionJournal for entries with "AI=" in reason
- Verify database write permissions
- Allow time for positions to close for P&L calculation

## References

- **Feature Engineering**: `ai_engine/feature_engineer.py`
- **XGBoost Agent**: `ai_engine/agents/xgb_agent.py`
- **AI Engine**: `backend/services/ai_trading_engine.py`
- **Selection Logic**: `backend/services/selection_engine.py`
- **Execution Flow**: `backend/services/execution.py`
- **API Routes**: `backend/routes/ai.py`

---

**Created**: 2024-11-12  
**Last Updated**: 2024-11-12  
**Status**: Active Integration - Testing Phase
