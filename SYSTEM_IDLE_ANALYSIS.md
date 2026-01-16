# System Idle Analysis - Nov 19, 2025

## Problem Statement
Hybrid Agent loaded successfully with TFT + XGBoost, but **system generates 0 trades** despite:
- ✅ Backend healthy
- ✅ Paper trading enabled
- ✅ AI engine running every ~60s
- ✅ 222 symbols scanned
- ✅ Signals generated (max confidence 0.65)

## Root Cause Analysis

### The Pipeline
```
XGBAgent → ML prediction (confidence < 0.55)
   ↓
   Falls back to _rule_based_fallback()
   ↓
   Returns model="rule_fallback_rsi" (confidence ≤ 0.65)
   ↓
HybridAgent combines signals
   ↓
   Returns model="hybrid_ensemble" (weighted confidence)
   ↓
AITradingEngine processes (threshold 0.64)
   ↓
EventDrivenExecutor filters (if model == "rule_fallback_rsi": skip)
   ↓
   🚫 ALL SIGNALS FILTERED OUT
```

### Key Findings

1. **ML Confidence Gap**
   - XGBoost model returns confidence < 0.55 for most symbols
   - File: `ai_engine/agents/xgb_agent.py:420-421`
   - When confidence < 0.55, it falls back to rule-based RSI signals

2. **Fallback Filter**
   - File: `backend/services/event_driven_executor.py:143-145`
   - Blocks all signals with `model == "rule_fallback_rsi"`
   - Was intended to ensure ML-only trading
   - But most symbols return rule_fallback due to low ML confidence

3. **Threshold Mismatch**
   - ML threshold for fallback: **0.55** (line 420, 467 in xgb_agent.py)
   - Execution threshold: **0.64** (QT_CONFIDENCE_THRESHOLD)
   - Rule fallback max confidence: **0.65** (line 589, 598 in xgb_agent.py)
   - Gap: 0.64-0.65 is too narrow, most signals at 0.50-0.63

### Evidence

**From Logs:**
```
17:56:51 | AI signals generated for 222 symbols: BUY=22 SELL=67 HOLD=133 | conf avg=0.49 max=0.65
17:56:51 | Found 0 high-confidence signals (>= 0.64)
17:56:51 | ⚠️ BTCUSDT: Skipping - using fallback rules (not trained ML)
17:56:51 | ⚠️ ETHUSDT: Skipping - using fallback rules (not trained ML)
...
[1069 symbols total filtered as "not trained ML"]
```

**Signal Distribution:**
- Generated: BUY=22, SELL=67, HOLD=133
- Max confidence: 0.65
- Avg confidence: 0.49
- Passing filter (>= 0.64): **0**

### Why Models Have Low Confidence

1. **Old Training Data**
   - Last trained: Nov 11, 2025 (ensemble_model.pkl)
   - TFT model: Nov 18, 2025 (tft_model.pth - 2.70MB)
   - XGBoost models: 400+ versioned files, latest Nov 17
   - Market conditions changed, models stale

2. **CoinGecko Rate Limits**
   - Training interrupted immediately (429 errors)
   - Cannot retrain with current setup
   - Missing: market cap, social metrics, volume data

3. **Feature Mismatch**
   - Models trained on specific feature set
   - Current features may differ → low confidence

## Solution Options

### Option 1: Lower Confidence Threshold (QUICK FIX)
**Impact: IMMEDIATE trading within 1 minute**

```env
QT_CONFIDENCE_THRESHOLD=0.55  # Down from 0.64
```

**Pros:**
- Immediate results
- Tests if system works end-to-end
- Allows rule-based signals to trigger

**Cons:**
- Lower quality signals
- Higher risk trades
- Still using fallback rules, not ML

**Risk Level:** Medium (paper trading mitigates)

---

### Option 2: Remove Fallback Filter (MODERATE)
**Impact: Trading starts within 1 minute**

File: `backend/services/event_driven_executor.py:143-145`

```python
# REMOVE these lines:
if model == "rule_fallback_rsi":
    logger.debug(f"⚠️ {symbol}: Skipping - using fallback rules (not trained ML)")
    continue
```

**Pros:**
- Allows all high-confidence signals through
- No threshold change needed
- Rule-based signals can still be useful

**Cons:**
- Defeats purpose of ML-only filter
- Lower signal quality
- Not using trained models

**Risk Level:** Medium-High

---

### Option 3: Retrain Models with Binance-Only Data (RECOMMENDED)
**Impact: 30-60 minutes setup + training time**

Steps:
1. Modify `ai_engine/train_and_save.py` to skip CoinGecko entirely
2. Use only Binance OHLCV + technical indicators
3. Train on 10-15 symbols first (avoid rate limits)
4. Expand to 45 symbols gradually

**Features to Keep:**
- OHLCV (Close, Volume, High, Low)
- RSI, EMA, MACD (technical indicators)
- Volatility, momentum (calculated from OHLCV)

**Features to Remove:**
- Market cap (CoinGecko)
- Social metrics (CoinGecko)
- Coingecko volume (redundant with Binance)

**Pros:**
- Real ML predictions with high confidence
- No API rate limits
- Fresh models for current market
- Sustainable long-term

**Cons:**
- Takes time to implement
- Need to modify training pipeline
- May lose some predictive features

**Risk Level:** Low (proper solution)

---

### Option 4: Hybrid Approach (BEST)
**Impact: Immediate trading + improved quality over time**

**Phase 1 (5 minutes):**
```env
QT_CONFIDENCE_THRESHOLD=0.58  # Allow some rule-based signals
```

**Phase 2 (1 hour):**
- Retrain with Binance-only data
- Target: Get ML confidence > 0.64 for top symbols

**Phase 3 (ongoing):**
- Gradually raise threshold back to 0.64 as models improve
- Monitor performance and adjust

**Pros:**
- Immediate system validation
- Time to fix training pipeline
- Smooth transition to better models

**Cons:**
- Two-phase implementation
- Need monitoring during transition

**Risk Level:** Low-Medium

---

## Current System Status

### Environment Variables
```
QT_PAPER_TRADING=true ✅
QT_CONFIDENCE_THRESHOLD=0.64 ⚠️ (too high)
QT_MIN_CONFIDENCE=0.65 ⚠️ (contradicts 0.64?)
QT_ENABLE_AI_TRADING=true ✅
QT_ENABLE_EXECUTION=true ✅
QT_EVENT_DRIVEN_MODE=true ✅
```

### Models Available
```
ensemble_model.pkl    3.09MB   Nov 11, 2025
tft_model.pth         2.70MB   Nov 18, 2025
xgb_model.json        2.11MB   Oct 03, 2025
xgb_model.pkl         0.20MB   Nov 19, 2025
+ 400 versioned scaler/xgb files
```

### Active Positions
**0 positions** (all closed)

### AI Engine Activity
- Scanning: 222 symbols every ~60s
- Signals: BUY=22, SELL=67, HOLD=133
- Max confidence: 0.65
- Filtered: 100% (0 pass threshold)

## Recommendations

### Immediate Action (Next 5 minutes)
```bash
# Test with lower threshold
systemctl stop backend
# Edit .env or systemctl.yml:
# QT_CONFIDENCE_THRESHOLD=0.58
systemctl up -d backend

# Monitor for signals
python monitor_hybrid.py -i 5
```

### Short-term (Next 1 hour)
1. Create `train_binance_only.py` script
2. Remove CoinGecko dependencies
3. Train on 10 high-volume symbols
4. Deploy new models
5. Verify confidence > 0.64

### Long-term (Next 1 week)
1. Set up continuous training with Binance data only
2. Monitor model performance daily
3. Gradually expand symbol universe
4. Implement model versioning/rollback
5. Add confidence monitoring dashboard

## Code Changes Needed

### Quick Fix (5 min)
```yaml
# systemctl.yml or .env
QT_CONFIDENCE_THRESHOLD=0.58  # Test threshold
```

### Proper Fix (1 hour)
```python
# ai_engine/train_binance_only.py
async def train_models_binance_only(symbols: List[str]):
    """Train models using only Binance data (no CoinGecko)"""
    features = []
    for symbol in symbols:
        # Fetch OHLCV from Binance
        ohlcv = await binance_client.get_ohlcv(symbol, limit=600)
        
        # Calculate technical indicators
        df = calculate_indicators(ohlcv)
        
        # NO CoinGecko calls
        features.append(df)
    
    # Train XGBoost
    xgb_model = train_xgboost(features, targets)
    
    # Train TFT
    tft_model = train_tft(features, targets)
    
    # Save models
    save_model(xgb_model, "xgb_model.pkl")
    save_model(tft_model, "tft_model.pth")
```

## Testing Plan

### Validation Steps
1. ✅ Backend healthy
2. ✅ Paper trading enabled
3. ⚠️ Signal generation (low confidence)
4. ❌ Signal filtering (all blocked)
5. ❌ Trade execution (never reached)

### After Fix
1. Lower threshold to 0.58
2. Monitor for 5 minutes → should see first signals
3. Verify paper trades execute
4. Check execution logs for success
5. Monitor PnL in paper account

### Success Criteria
- [ ] At least 1 signal passes filter within 5 min
- [ ] Paper order placed successfully
- [ ] Position tracking works
- [ ] TP/SL orders set correctly
- [ ] No live trading executed

## Next Steps

**Choose your path:**

A. **Quick Test** → Lower threshold to 0.58, monitor for 5 min
B. **Proper Fix** → Create Binance-only training pipeline (1 hour)
C. **Hybrid** → Do both (test now, fix properly in parallel)

**My Recommendation: Option C (Hybrid)**
- Lower threshold now to 0.58
- Start training script development in parallel
- Monitor results from both approaches
- Transition smoothly once new models ready

---

**Status:** 📊 Analysis complete, awaiting user decision
**Risk:** ⚠️ Medium (paper trading, but system not tested end-to-end)
**Urgency:** 🚨 High (system not producing any trades)

