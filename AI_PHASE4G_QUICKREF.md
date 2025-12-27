# 🧪 PHASE 4G QUICK REFERENCE

**Model Validation Layer** - Automatic Quality Control for Retrained Models

---

## 🎯 What It Does

Automatically validates retrained models before promoting them to production.

**Process:**
1. Adaptive retrainer completes 4h cycle
2. Validator loads old (production) + new (retrained) models
3. Evaluates both on 12h validation dataset
4. Compares MAPE and Sharpe ratio
5. **ACCEPT:** Replace production if both metrics improve
6. **REJECT:** Delete retrained model if criteria not met
7. Log decision to audit trail

---

## 📊 Validation Criteria

**NEW MODEL PROMOTED ONLY IF:**

✅ `new_mape < old_mape * 0.97` (3%+ MAPE improvement)  
**AND**  
✅ `new_sharpe > old_sharpe` (Better risk-adjusted returns)

**BOTH conditions must be met. If either fails → Model rejected.**

---

## 🔧 Key Files

```
backend/microservices/ai_engine/services/
└── model_validation_layer.py      # Validation engine (8.8 KB)

/app/models/
├── patchtst.pth                    # Production
├── patchtst_adaptive.pth          # Candidate (post-retraining)
├── nhits.pth                      # Production
└── nhits_adaptive.pth             # Candidate (post-retraining)

/app/logs/
└── model_validation.log           # Audit trail
```

---

## 📈 Metrics Explained

### MAPE (Mean Absolute Percentage Error)
- **Lower is better**
- Measures prediction accuracy
- Threshold: New ≤ 97% of old (3%+ improvement required)

### Sharpe Ratio
- **Higher is better**
- Measures risk-adjusted returns
- Threshold: New > old (any improvement required)

### PnL (Profit & Loss)
- Directional profit from predictions
- **Not used in decision criteria** (informational only)

---

## 🔍 Monitoring Commands

### Check Validator Status
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254

# Health endpoint
curl http://localhost:8001/health | jq '.metrics.model_validator'

# Recent validation decisions (last 20 lines)
docker exec quantum_ai_engine tail -20 /app/logs/model_validation.log

# Validator logs
docker logs quantum_ai_engine | grep Validator
```

### Expected Health Response
```json
{
  "model_validator": {
    "enabled": true,
    "validation_log_path": "/app/logs/model_validation.log",
    "recent_validations": [
      "2025-12-20T12:00:00.123 [Validator] patchtst: ... → ✅ ACCEPT",
      "2025-12-20T12:00:15.456 [Validator] nhits: ... → ❌ REJECT"
    ],
    "criteria": {
      "mape_improvement_required": "3%",
      "sharpe_improvement_required": true
    }
  }
}
```

---

## 📝 Log Format

```
TIMESTAMP [Validator] MODEL: old(MAPE=X, PnL=Y, Sharpe=Z) → new(MAPE=X, PnL=Y, Sharpe=Z) → MAPE_improvement=X% → DECISION
TIMESTAMP [Validator] ACTION
```

**Example:**
```
2025-12-20T12:00:00 [Validator] patchtst: old(MAPE=0.0450, PnL=125.30, Sharpe=1.25) → new(MAPE=0.0420, PnL=145.80, Sharpe=1.37) → MAPE_improvement=6.7% → ✅ ACCEPT
2025-12-20T12:00:15 [Validator] ✅ Promoted new patchtst model to production
```

---

## 🔄 Workflow Timeline

```
T=0h00m: Retraining completes
         └─> patchtst_adaptive.pth, nhits_adaptive.pth created

T=0h01m: Validation triggers
         └─> Fetch 12h validation data
         └─> Load production + adaptive models
         └─> Evaluate both on same dataset

T=0h03m: PatchTST validation
         └─> MAPE: 0.045 → 0.042 (✅ 6.7% improvement)
         └─> Sharpe: 1.25 → 1.37 (✅ improved)
         └─> Decision: ✅ ACCEPT
         └─> Action: Replace patchtst.pth with adaptive

T=0h04m: N-HiTS validation
         └─> MAPE: 0.048 → 0.051 (❌ worse)
         └─> Sharpe: 1.18 → 1.15 (❌ worse)
         └─> Decision: ❌ REJECT
         └─> Action: Delete nhits_adaptive.pth

T=0h05m: Validation complete
         └─> Log: {"patchtst": True, "nhits": False}
         └─> System continues with improved PatchTST

T=4h00m: Next retraining cycle begins...
```

---

## 🛡️ Safety Features

✅ **Strict Criteria:** Both MAPE and Sharpe must improve  
✅ **Automatic Rollback:** Poor models deleted, never reach production  
✅ **Audit Trail:** Every decision logged with full metrics  
✅ **Zero Risk:** Production models only replaced when proven better  
✅ **Graceful Degradation:** Errors keep production models intact  

---

## 📊 Integration Status

| Phase | Status | Description |
|-------|--------|-------------|
| 4D+4E | ✅ Active | Model Supervisor & Governance |
| 4F | ✅ Active | Adaptive Retraining (4h cycles) |
| 4G | ✅ Active | Model Validation Layer ← **NEW** |

**Complete Loop:**
```
Generate Predictions → Track Performance → Detect Drift →
Retrain Models → Validate Candidates → Promote Best → Loop
```

---

## 🎯 Success Indicators

✅ **Validator initialized** - Log: "[PHASE 4G] Validator initialized"  
✅ **Health endpoint active** - `model_validator.enabled = true`  
⏳ **First validation** - Waiting for 4h retraining cycle  
⏳ **Audit log populated** - Will appear after first validation  

---

## 🚨 Troubleshooting

### No validation log entries after 4h
**Check:** Did retrainer complete?
```bash
docker logs quantum_ai_engine | grep "Retrainer.*complete"
```

### All models rejected
**Normal!** Validator is working correctly - market conditions may have changed, making retraining ineffective this cycle.

### Validator error in logs
**Check:** Are model files present?
```bash
docker exec quantum_ai_engine ls -lh /app/models/
```

---

## 📞 Quick Commands

```bash
# Full health check
curl -s http://localhost:8001/health | python3 -m json.tool | grep -A15 model_validator

# Recent validations (compact)
docker exec quantum_ai_engine tail -10 /app/logs/model_validation.log

# Count validations
docker exec quantum_ai_engine wc -l /app/logs/model_validation.log

# Check active models
curl -s http://localhost:8001/health | jq '.metrics.models_loaded'

# Live validator logs
docker logs -f quantum_ai_engine | grep Validator
```

---

## 🏆 What This Means

**Before Phase 4G:**
- Retrained models deployed blindly
- No quality control
- Risk of regression
- Manual validation required

**After Phase 4G:**
- Every retrained model scientifically evaluated
- Only proven-better models reach production
- Automatic rollback of poor models
- Full audit trail of all changes

**Result:** Self-learning + Self-validating AI system with zero manual oversight.

---

**Status:** ✅ DEPLOYED & OPERATIONAL  
**Location:** VPS quantum_ai_engine container  
**Next Event:** First validation ~4h after retraining  

