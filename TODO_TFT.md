# 🎯 QUANTUM TRADER - TODO LIST
**Updated: November 18, 2025**

---

## 🔥 CRITICAL - DO FIRST

### 1. ⚠️ Debug TFT Training Error
**Priority: URGENT**  
**Time: 30 min - 1 time**

Training crashed with exit code 1. Need to identify and fix error.

**Steps:**
```bash
# Stop backend to avoid database locks
docker-compose stop backend

# Start backend in isolation
docker-compose start backend
sleep 5

# Try training with verbose error logging
docker exec quantum_backend python /app/train_tft.py 2>&1 | tee tft_training_full.log

# If still fails, try with smaller dataset
docker exec quantum_backend python -c "
import sys; sys.path.insert(0, '/app')
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
db = SessionLocal()
count = db.query(AITrainingSample).filter(AITrainingSample.outcome_known==True).count()
print(f'Samples available: {count}')
db.close()
"
```

**Possible issues:**
- [ ] Database locked (continuous training running)
- [ ] Out of memory (reduce batch size)
- [ ] Missing dependency
- [ ] Tensor shape mismatch
- [ ] SQLite timeout

**Solutions to try:**
1. Stop continuous training before TFT training
2. Reduce batch size from 128 to 64
3. Train on subset (10K samples) first
4. Add more error handling in train_tft.py
5. Check available RAM: `docker stats quantum_backend`

---

### 2. 🚀 Complete TFT Model Training
**Priority: HIGH**  
**Time: 10-30 min (actual training)**

Once debugging is done, complete the full training.

**Command:**
```bash
docker exec quantum_backend python /app/train_tft.py
```

**Expected:**
- Train sequences: 253,412
- Val sequences: 63,354
- Epochs: 20 (with early stopping)
- Output: `ai_engine/models/tft_model.pth` (~5-10 MB)
- Target accuracy: ≥55%

**Success Checklist:**
- [ ] Training completes without errors
- [ ] Validation accuracy ≥55% (minimum)
- [ ] Model file created: `tft_model.pth`
- [ ] Normalization stats saved: `tft_normalization.json`
- [ ] Training log shows improvement over epochs

---

### 3. 🔌 Integrate TFT Agent in Backend
**Priority: HIGH**  
**Time: 1-2 timer**

Replace XGBoost agent with TFT agent in production code.

**File to modify:** `backend/services/ai_coordinator.py`

**Changes needed:**
```python
# Add import
from ai_engine.agents.tft_agent import TFTAgent

# In AICoordinator.__init__:
self.tft_agent = TFTAgent()
if not self.tft_agent.load_model():
    logger.warning("TFT model not available, using XGBoost fallback")

# In get_ai_signals():
# Try TFT first, fallback to XGBoost
if self.tft_agent.model is not None:
    predictions = self.tft_agent.batch_predict(symbols_features, confidence_threshold=0.65)
else:
    # Fallback to XGBoost
    predictions = self.xgb_agent.batch_predict(symbols_features)
```

**Testing:**
```bash
# Restart backend after changes
docker-compose restart backend

# Watch logs for TFT initialization
docker logs quantum_backend --tail 50 | grep -i "tft\|temporal"

# Test predictions
docker logs quantum_backend --tail 100 --follow | grep "AI signals"
```

**Checklist:**
- [ ] Import TFTAgent in ai_coordinator.py
- [ ] Initialize TFT agent in constructor
- [ ] Load model at startup
- [ ] Replace prediction calls
- [ ] Add fallback to XGBoost
- [ ] Add error handling
- [ ] Test locally
- [ ] Restart backend
- [ ] Verify no errors in logs

---

## 📊 TESTING & VALIDATION

### 4. ✅ Test TFT Predictions
**Priority: MEDIUM**  
**Time: 30 min**

Verify TFT agent works correctly before production deployment.

**Unit Tests:**
```bash
# Test 1: Model loading
docker exec quantum_backend python -c "
from ai_engine.agents.tft_agent import TFTAgent
agent = TFTAgent()
success = agent.load_model()
print(f'Model loaded: {success}')
"

# Test 2: Single prediction
docker exec quantum_backend python -c "
from ai_engine.agents.tft_agent import TFTAgent
agent = TFTAgent()
agent.load_model()

features = {
    'Close': 50000, 'Volume': 1000000, 'EMA_10': 50100,
    'EMA_50': 49800, 'RSI': 55, 'MACD': 0.5,
    'MACD_signal': 0.3, 'BB_upper': 51000, 'BB_middle': 50000,
    'BB_lower': 49000, 'ATR': 500, 'volume_sma_20': 950000,
    'price_change_pct': 0.2, 'high_low_range': 1000
}

# Add 60 samples to history
for i in range(60):
    agent.add_to_history('BTCUSDT', features)

action, conf, meta = agent.predict('BTCUSDT', features)
print(f'Action: {action}, Confidence: {conf:.2%}')
print(f'Metadata: {meta}')
"

# Test 3: Batch predictions
docker exec quantum_backend python -c "
from ai_engine.agents.tft_agent import TFTAgent
agent = TFTAgent()
agent.load_model()

symbols_features = {
    'BTCUSDT': {'Close': 50000, 'Volume': 1000000, 'RSI': 55, ...},
    'ETHUSDT': {'Close': 3000, 'Volume': 500000, 'RSI': 45, ...},
}

results = agent.batch_predict(symbols_features)
for symbol, (action, conf, meta) in results.items():
    print(f'{symbol}: {action} ({conf:.1%})')
"
```

**Checklist:**
- [ ] Model loads successfully
- [ ] Single predictions work
- [ ] Batch predictions work
- [ ] Confidence thresholds work
- [ ] Insufficient history handled correctly
- [ ] Predictions are not all HOLD
- [ ] Inference time <50ms per symbol

---

### 5. 📈 Monitor WIN Rate Improvement
**Priority: MEDIUM**  
**Time: 24-48 timer (continuous monitoring)**

After integration, monitor performance to ensure improvement.

**Commands:**
```bash
# Check dataset stats
docker exec quantum_backend python /app/check_dataset.py

# Watch live predictions
docker logs quantum_backend --tail 100 --follow | grep "AI signals"

# Check WIN rate after N hours
docker exec quantum_backend python -c "
from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample
from sqlalchemy import func
db = SessionLocal()

# Last 24 hours
import datetime
cutoff = datetime.datetime.now() - datetime.timedelta(hours=24)

samples = db.query(AITrainingSample).filter(
    AITrainingSample.outcome_known == True,
    AITrainingSample.created_at >= cutoff
).all()

wins = sum(1 for s in samples if s.realized_pnl and s.realized_pnl > 0)
losses = sum(1 for s in samples if s.realized_pnl and s.realized_pnl < 0)
total = wins + losses

if total > 0:
    win_rate = wins / total * 100
    print(f'Last 24h: {wins} wins, {losses} losses')
    print(f'WIN RATE: {win_rate:.1f}%')
else:
    print('Not enough data yet')
"
```

**Target Metrics:**
- [ ] WIN rate: 60-75% (vs 42% before)
- [ ] BUY signals: 25-35%
- [ ] SELL signals: 25-35%
- [ ] HOLD signals: 30-50%
- [ ] Average confidence: >0.65
- [ ] Prediction latency: <50ms

**Monitor for:**
- [ ] 100 predictions collected
- [ ] 1000 predictions collected
- [ ] 24 hours of operation
- [ ] 7 days of operation

---

## 🔧 OPTIMIZATION (if needed)

### 6. 🎯 Tune Hyperparameters
**Priority: LOW**  
**Time: 2-4 timer**

If WIN rate is below 55%, optimize model parameters.

**What to tune:**
```python
# In train_tft.py:
- sequence_length: 60 → Try 30, 90, 120
- batch_size: 128 → Try 64, 256
- hidden_size: 128 → Try 96, 192, 256
- num_heads: 8 → Try 4, 16
- learning_rate: 0.001 → Try 0.0005, 0.002
- dropout: 0.1 → Try 0.05, 0.2

# In tft_agent.py:
- confidence_threshold: 0.65 → Try 0.6, 0.7, 0.75
```

**Process:**
1. Change one parameter at a time
2. Retrain model
3. Test for 24h
4. Compare WIN rate
5. Keep if improvement, revert if worse

**Checklist:**
- [ ] Try different sequence lengths
- [ ] Experiment with confidence thresholds
- [ ] Test different learning rates
- [ ] Validate with cross-validation
- [ ] Document best parameters

---

### 7. 📊 Feature Engineering
**Priority: LOW**  
**Time: 2-3 timer**

Add more features if WIN rate needs improvement.

**New features to consider:**
```python
# Temporal features
- 'returns_1h': 1-hour return
- 'returns_4h': 4-hour return
- 'returns_1d': 1-day return
- 'volatility_24h': Rolling 24h volatility

# Market structure
- 'order_imbalance': Bid/ask imbalance
- 'spread_pct': Bid-ask spread
- 'depth_ratio': Bid depth / ask depth

# Momentum indicators
- 'momentum_12h': 12-hour momentum
- 'roc_24h': 24-hour rate of change

# Multi-timeframe
- 'rsi_1h': RSI on 1h chart
- 'rsi_4h': RSI on 4h chart
- 'macd_1h': MACD on 1h chart
```

**Implementation:**
1. Add features to `FeatureEngineer`
2. Regenerate dataset with new features
3. Update `input_size` in TFT model
4. Retrain
5. Test

---

## 📦 DEPLOYMENT & PRODUCTION

### 8. 🚀 Production Deployment
**Priority: MEDIUM**  
**Time: 1-2 timer**

Deploy to production after testing is successful.

**Pre-deployment checklist:**
- [ ] TFT model trained and validated
- [ ] Integration tested locally
- [ ] WIN rate >55% confirmed
- [ ] No errors in logs for 24h
- [ ] Fallback to XGBoost working
- [ ] Performance metrics acceptable

**Deployment steps:**
```bash
# 1. Backup current state
docker exec quantum_backend cp -r /app/ai_engine/models /app/ai_engine/models.backup

# 2. Copy TFT model to production
docker cp ai_engine/models/tft_model.pth quantum_backend:/app/ai_engine/models/
docker cp ai_engine/models/tft_normalization.json quantum_backend:/app/ai_engine/models/

# 3. Update backend code
docker cp backend/services/ai_coordinator.py quantum_backend:/app/backend/services/

# 4. Restart services
docker-compose restart backend

# 5. Monitor logs
docker logs quantum_backend --tail 100 --follow
```

**Post-deployment:**
- [ ] Verify TFT model loaded
- [ ] Check predictions are generated
- [ ] Monitor for errors
- [ ] Watch WIN rate
- [ ] Set up alerts

---

### 9. 📊 Set Up Monitoring Dashboard
**Priority: LOW**  
**Time: 2-3 timer**

Create dashboard to monitor TFT performance.

**Metrics to track:**
```python
# Real-time
- Current WIN rate (1h, 24h, 7d)
- Prediction confidence distribution
- Signal distribution (BUY/SELL/HOLD)
- Prediction latency
- Model version in use

# Historical
- WIN rate over time (chart)
- Cumulative profit/loss
- Sharpe ratio
- Max drawdown
- False positive rate
```

**Implementation:**
1. Add TFT metrics endpoint
2. Update frontend dashboard
3. Add Grafana/Prometheus (optional)
4. Set up alerts (email/Slack)

---

## 🐛 KNOWN ISSUES TO FIX

### Issue 1: Training Crashed
- [ ] Debug training error
- [ ] Fix database locking
- [ ] Handle OOM errors
- [ ] Add better logging

### Issue 2: Database Locked During Training
- [ ] Stop continuous training before TFT training
- [ ] Use separate database for training
- [ ] Add retry logic
- [ ] Implement proper locking

### Issue 3: Normalization Stats Not Saved
- [ ] Save feature_mean and feature_std to JSON
- [ ] Load during agent initialization
- [ ] Add to model checkpoint

---

## 📚 DOCUMENTATION TO UPDATE

### Update Required:
- [x] TFT_IMPLEMENTATION_STATUS.md (DONE)
- [x] README.md (DONE)
- [x] TODO.md (THIS FILE)
- [ ] ARCHITECTURE.md (add TFT section)
- [ ] API.md (add TFT endpoints)
- [ ] DEPLOYMENT.md (add TFT deployment steps)

---

## 🎯 SUCCESS METRICS

### Minimum Viable (MVP):
- [ ] TFT model trained
- [ ] Validation accuracy ≥55%
- [ ] Integration complete
- [ ] No crashes
- [ ] WIN rate >50%

### Production Ready:
- [ ] Validation accuracy ≥60%
- [ ] WIN rate ≥55% (24h)
- [ ] Latency <50ms
- [ ] 0 errors in 24h
- [ ] Proper monitoring

### Optimal:
- [ ] Validation accuracy ≥65%
- [ ] WIN rate ≥60% (7d)
- [ ] Latency <30ms
- [ ] Sharpe ratio >1.5
- [ ] Max drawdown <10%

---

## 📅 TIMELINE

| Task | Est. Time | Status |
|------|-----------|--------|
| Debug training | 1h | ⏳ **NEXT** |
| Complete training | 30min | ⏳ Pending |
| Integration | 2h | ⏳ Pending |
| Testing | 1h | ⏳ Pending |
| Monitoring setup | 24h | ⏳ Pending |
| Optimization | 4h | Optional |
| **TOTAL** | **32h** | **0% Complete** |

---

## 🏁 QUICK START (Next Steps)

```bash
# Step 1: Debug and train
docker-compose stop backend
docker-compose start backend
docker exec quantum_backend python /app/train_tft.py

# Step 2: Verify model
docker exec quantum_backend ls -lh /app/ai_engine/models/tft_model.pth

# Step 3: Integrate
# (Edit ai_coordinator.py as described above)

# Step 4: Deploy
docker-compose restart backend

# Step 5: Monitor
docker logs quantum_backend --tail 100 --follow
```

---

*Updated: November 18, 2025*  
*Next Review: After TFT training completes*
