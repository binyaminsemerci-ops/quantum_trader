# Phase 4D + 4E: Quick Reference

## 🚀 Deployment Commands

```bash
# From Windows/WSL - Deploy to VPS
cd /mnt/c/quantum_trader
chmod +x scripts/deploy_phase4d_4e.sh
./scripts/deploy_phase4d_4e.sh

# Validate deployment
chmod +x scripts/validate_phase4d_4e.sh
./scripts/validate_phase4d_4e.sh
```

## 📊 Monitoring Commands

```bash
# Check governance status
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'curl -s http://localhost:8001/health | jq ".metrics.governance"'

# Watch live logs
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'docker logs -f quantum_ai_engine | grep -E "Governance|Supervisor|Drift"'

# Check weights
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'journalctl -u quantum_ai_engine.service --tail 50 | grep "Adjusted weights"'
```

## 🧪 Testing Commands

```bash
# Generate test signal
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'curl -X POST http://localhost:8001/api/ai/signal \
   -H "Content-Type: application/json" \
   -d "{\"symbol\":\"BTCUSDT\"}"'

# View full health
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'curl -s http://localhost:8001/health | jq .'
```

## ✅ Expected Log Output

```
[AI-ENGINE] 🧠 Initializing Model Supervisor & Governance...
[Supervisor] ✅ Registered model: PatchTST
[Supervisor] ✅ Registered model: NHiTS
[Supervisor] ✅ Registered model: XGBoost
[Supervisor] ✅ Registered model: LightGBM
[PHASE 4D+4E] Supervisor + Predictive Governance active
[Governance] 📊 Adjusted weights: PatchTST=0.23, NHiTS=0.28, XGBoost=0.25, LightGBM=0.24
```

## 🎯 Key Metrics

- **Drift Threshold:** 5% MAPE
- **Retrain Interval:** 1 hour (3600s)
- **Smoothing Factor:** 0.3
- **Models Tracked:** 4 (PatchTST, NHiTS, XGBoost, LightGBM)

## 🔧 Troubleshooting

### Issue: Models not registered
```bash
# Check ensemble is loaded
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'journalctl -u quantum_ai_engine.service | grep "Ensemble loaded"'
```

### Issue: Governance not running
```bash
# Verify initialization
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'journalctl -u quantum_ai_engine.service | grep "Governance active"'
```

### Issue: No weight adjustments
```bash
# Check if cycles are running
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 \
  'journalctl -u quantum_ai_engine.service | grep "Cycle complete"'
```

## 📁 Key Files

- **Service:** `backend/microservices/ai_engine/services/model_supervisor_governance.py`
- **Integration:** `microservices/ai_engine/service.py`
- **Deploy:** `scripts/deploy_phase4d_4e.sh`
- **Validate:** `scripts/validate_phase4d_4e.sh`
- **Docs:** `AI_PHASE4D_4E_IMPLEMENTATION.md`

## 🎉 Success Indicators

✅ All 4 models registered  
✅ Governance cycle running after predictions  
✅ Weights adjusting dynamically  
✅ Health endpoint shows governance metrics  
✅ Drift detection operational  
✅ Auto-retraining configured  

**System is self-regulating! 🤖**

