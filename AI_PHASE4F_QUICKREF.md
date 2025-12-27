# Phase 4F: Quick Reference Card

## 🚀 Status

✅ **DEPLOYED & OPERATIONAL**  
✅ **11 Models Active**  
✅ **Autonomous Retraining Every 4 Hours**

---

## 📊 Quick Health Check

```bash
# Check retrainer status
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'curl -s http://localhost:8001/health | jq ".metrics.adaptive_retrainer"'

# Expected output:
{
  "enabled": true,
  "retrain_interval_seconds": 14400,
  "retrain_count": 0,
  "time_until_next_seconds": 14300,
  "last_losses": {},
  "model_paths": {
    "patchtst": "/app/models/patchtst_adaptive.pth",
    "nhits": "/app/models/nhits_adaptive.pth"
  }
}
```

---

## 🔍 Monitoring Commands

### Watch Retraining Logs
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'docker logs -f quantum_ai_engine | grep -E "Retrainer|PatchTST|N-HiTS|Epoch"'
```

### Check Model Files
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'docker exec quantum_ai_engine ls -lh /app/models/ | grep adaptive'
```

### View All Active Modules
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'docker logs quantum_ai_engine | grep "models active"'
```

---

## ⏰ Retraining Schedule

- **Interval:** 4 hours (14400 seconds)
- **First Cycle:** ~4h after deployment
- **Ongoing:** Every 4h automatically
- **Data:** 24h lookback (≥5000 points)

---

## 📁 Key Files

| File | Location |
|------|----------|
| Retrainer Module | `backend/microservices/ai_engine/services/adaptive_retrainer.py` |
| Service Integration | `microservices/ai_engine/service.py` |
| Model Storage | `/app/models/*.pth` (in container) |
| Training Data | `/app/adaptive_training/` (in container) |

---

## 🎯 What Happens Every 4 Hours

1. ⏰ **Check:** Has interval passed?
2. 📊 **Fetch:** Get 24h market data
3. 🔧 **Prepare:** Create training dataset
4. 🧠 **Train PatchTST:** 2 epochs
5. 🧠 **Train N-HiTS:** 2 epochs
6. 💾 **Save:** Update model files
7. 📝 **Log:** Record metrics and history

---

## 🔥 Complete Phase 4 Stack

```
Phase 4A-C: Foundation (6 modules) ✅
Phase 4D: Model Supervisor ✅
Phase 4E: Predictive Governance ✅
Phase 4F: Adaptive Retraining ✅

Total: 11 AI Modules Active 🎉
```

---

## ⚙️ Configuration

```python
retrain_interval = 14400  # 4 hours
min_data_points = 5000    # Minimum data
max_epochs = 2            # Training epochs
learning_rate = 1e-4      # Adam LR
batch_size = 64           # Training batch
window_size = 128         # Sequence length
```

---

## 🚨 Troubleshooting

### No Models Generated?
- Check: `time_until_next_seconds` in health
- Wait: First cycle runs after 4h
- Logs: `docker logs quantum_ai_engine | grep Retrainer`

### Retraining Failed?
```bash
# Check errors
docker logs quantum_ai_engine | grep -A 5 "Retrainer.*error"

# Verify data availability
docker exec quantum_ai_engine ls -lh /app/adaptive_training/
```

### Health Endpoint Not Responding?
```bash
# Check container status
docker ps | grep quantum_ai_engine

# Restart if needed
docker restart quantum_ai_engine
```

---

## 📈 Expected Logs

### Initialization
```
[Retrainer] Initialized - Interval: 14400s, Min data: 5000, Epochs: 2
[AI-ENGINE] ✅ Adaptive Retraining Pipeline active
[PHASE 4F] Adaptive Retrainer initialized - Interval: 4h
```

### During Retraining
```
[Retrainer] 🔄 Initiating adaptive retraining cycle...
[Retrainer] Fetching 24h data for BTCUSDT...
[Retrainer] ✅ Fetched 5847 data points for BTCUSDT
[Retrainer] Prepared dataset: X=(5719, 128, 5), y=(5719,)
[Retrainer][PatchTST] Starting retraining...
[Retrainer][PatchTST] Epoch 1/2, Loss: 0.003456
[Retrainer][PatchTST] Epoch 2/2, Loss: 0.002891
[Retrainer][PatchTST] ✅ Model saved to /app/models/patchtst_adaptive.pth
[Retrainer][N-HiTS] Starting retraining...
[Retrainer][N-HiTS] Epoch 1/2, Loss: 0.004123
[Retrainer][N-HiTS] Epoch 2/2, Loss: 0.003567
[Retrainer][N-HiTS] ✅ Model saved to /app/models/nhits_adaptive.pth
[Retrainer] ✅ Cycle complete - Next cycle in ~4h
```

---

## 🎉 Success!

Your system now:
- ✅ Learns autonomously every 4 hours
- ✅ Adapts to market conditions
- ✅ Requires zero manual intervention
- ✅ Tracks all metrics and history

**True Adaptive Trading Intelligence! 🤖📈**
