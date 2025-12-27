# 🚀 Continuous Learning Quick Start

## ✅ System Status: LIVE & READY

### Current Setup
- ✅ Database tables created (`ai_training_samples`, `ai_model_versions`)
- ✅ Backend running with continuous learning enabled
- ✅ AI model active and generating predictions
- ✅ API endpoints functional
- ✅ Automatic retraining scheduled (daily 03:00 UTC)

---

## 📊 Monitor System

### Check Training Data
```powershell
# See all training samples
curl -X GET "http://localhost:8000/ai/training-samples?limit=20" `
  -H "X-Admin-Token: live-admin-token"

# See only completed samples (with outcomes)
curl -X GET "http://localhost:8000/ai/training-samples?outcome_known=true&limit=50" `
  -H "X-Admin-Token: live-admin-token"
```

### Check Model Versions
```powershell
# List all trained models
curl -X GET "http://localhost:8000/ai/models" `
  -H "X-Admin-Token: live-admin-token"
```

### Check AI Status
```powershell
# See current AI predictions
curl -X GET "http://localhost:8000/ai/live-status" `
  -H "X-Admin-Token: live-admin-token"
```

---

## 🔄 Manual Retraining

### When You Have 100+ Completed Trades:
```powershell
# Trigger retraining
curl -X POST "http://localhost:8000/ai/retrain?min_samples=100" `
  -H "X-Admin-Token: live-admin-token"
```

### Expected Response:
```json
{
  "status": "success",
  "version_id": "v20251112_150000",
  "training_samples": 250,
  "validation_samples": 62,
  "train_accuracy": 0.68,
  "validation_accuracy": 0.62,
  "train_mae": 0.0245,
  "validation_mae": 0.0312,
  "model_path": "ai_engine/models/xgb_model_v20251112_150000.pkl"
}
```

---

## ⚙️ Activate New Model

### After Reviewing Metrics:
```powershell
# 1. List models and find best version
curl -X GET "http://localhost:8000/ai/models" `
  -H "X-Admin-Token: live-admin-token"

# 2. Activate the best model
curl -X POST "http://localhost:8000/ai/activate-model/v20251112_150000" `
  -H "X-Admin-Token: live-admin-token"

# 3. Restart backend to load new model
Get-Process | Where-Object {$_.Path -like "*python*" -and (Get-NetTCPConnection -OwningProcess $_.Id -ErrorAction SilentlyContinue | Where-Object LocalPort -eq 8000)} | Stop-Process -Force

cd c:\quantum_trader\backend
Start-Process pwsh -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command","cd c:\quantum_trader\backend; `$env:QT_AI_RETRAINING_ENABLED='1'; uvicorn main:app --host 0.0.0.0 --port 8000" -WindowStyle Minimized
```

---

## 📈 Understanding Metrics

### Key Metrics to Watch:
- **validation_accuracy** (most important!): >60% good, >65% great, >70% excellent
- **validation_mae**: Lower is better (Mean Absolute Error)
- **training_samples**: More samples = better model
- **live_accuracy**: Real-world performance after activation

### Decision Guide:
- ✅ Activate if: `validation_accuracy > current_model` AND `samples > 200`
- ⚠️  Review if: `validation_accuracy` only slightly better
- ❌ Skip if: `validation_accuracy < current_model` OR `samples < 100`

---

## 🛠️ Direct Database Queries

### SQLite Commands:
```powershell
cd c:\quantum_trader\backend

# Count total samples
sqlite3 data/trades.db "SELECT COUNT(*) FROM ai_training_samples"

# Count completed samples
sqlite3 data/trades.db "SELECT COUNT(*) FROM ai_training_samples WHERE outcome_known=1"

# See recent samples
sqlite3 data/trades.db "SELECT symbol, predicted_action, prediction_confidence, realized_pnl FROM ai_training_samples ORDER BY timestamp DESC LIMIT 10"

# Check model versions
sqlite3 data/trades.db "SELECT version_id, training_samples, validation_accuracy, is_active FROM ai_model_versions ORDER BY trained_at DESC"
```

---

## 📝 How It Works

### Automatic Learning Cycle:

1. **Prediction Phase** (every 30 min)
   - AI analyzes market data
   - Makes BUY/SELL/HOLD predictions
   - Features saved to `ai_training_samples`

2. **Execution Phase**
   - Orders sent to Binance
   - Entry price/quantity recorded
   - `executed=True` flag set

3. **Outcome Phase** (when position closes)
   - Exit price recorded
   - P&L calculated
   - `outcome_known=True` flag set
   - Target label (% return) calculated

4. **Retraining Phase** (daily 03:00 UTC)
   - Checks if 100+ completed samples exist
   - Builds dataset from features + outcomes
   - Trains new XGBoost model
   - Validates on holdout set (20%)
   - Saves as new version

5. **Activation Phase** (manual)
   - You review new model metrics
   - Activate if better than current
   - Backend restart loads new model

---

## 🎯 Timeline Expectations

### Week 1-2: Data Collection
- System collects 50-100 samples
- Not enough for retraining yet
- Monitor: Are samples being created?

### Week 3-4: First Retraining
- 100+ completed samples available
- First automatic retraining at 03:00 UTC
- Review first model version

### Month 2+: Continuous Improvement
- New model every few days/weeks
- Compare performance trends
- AI gets smarter over time 📈

---

## ⚡ Quick Troubleshooting

### "insufficient_samples" Error
**Cause**: Not enough completed trades  
**Solution**: Wait longer or lower `min_samples` temporarily

### No Samples Being Created
**Cause**: AI not executing trades (all HOLD signals)  
**Solution**: Check AI thresholds (currently ±0.001), consider adjusting

### Model Performance Worse
**Cause**: Overfitting or bad training data  
**Solution**: Don't activate, wait for more diverse samples

### Backend Not Starting
**Cause**: Port 8000 in use or module errors  
**Solution**: Kill existing process, check logs

---

## 📚 Documentation

- **Full Guide**: `CONTINUOUS_LEARNING.md`
- **Implementation Summary**: `AI_CONTINUOUS_LEARNING_SUMMARY.md`
- **AI Integration**: `AI_INTEGRATION.md`

---

## 🎉 You're All Set!

Your AI will now:
- ✅ Learn from every trade automatically
- ✅ Retrain daily when enough data is available
- ✅ Track performance metrics
- ✅ Allow safe model versioning and rollback

**Just let it run and check back in 1-2 weeks!** 🚀
