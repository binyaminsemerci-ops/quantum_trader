# P0.6 TRAINING FAILURE REPORT
**Date**: 2026-01-10 03:36 UTC  
**Commit**: bff8e6ae  
**Database**: `/opt/quantum/data/quantum_trader.db` (6,000 samples)  
**Status**: ❌ **FAILED SANITY CHECKS - NO DEPLOYMENT**

---

## EXECUTIVE SUMMARY

P0.6 training **completed** but **failed hard sanity checks**. Model collapsed to constant predictions (100% HOLD, confidence stddev = 0.0000). Gates 1 and 2 both failed. **DO NOT DEPLOY**.

---

## 1. PREFLIGHT (✅ PASSED)

### Git Status
- **Branch**: main
- **HEAD**: bff8e6ae
- **Script DB path**: Line 659: `/opt/quantum/data/quantum_trader.db` ✅

### Database Sanity
```
Total training samples: 6,000
  WIN:  3,614 (60.2%)
  LOSS: 2,386 (39.8%)

Per-symbol counts:
  ADAUSDT: 1,156 samples (19.3%)
  BNBUSDT: 1,237 samples (20.6%)
  BTCUSDT: 1,211 samples (20.2%)
  ETHUSDT: 1,142 samples (19.0%)
  SOLUSDT: 1,254 samples (20.9%)
```
✅ All checks passed, data ready for training

---

## 2. TRAINING EXECUTION

### Configuration
```
Database: /opt/quantum/data/quantum_trader.db
Window: Last 30 days (cutoff: 2025-12-11)
Epochs: 20
Batch size: 32
Learning rate: 0.0003
Label smoothing: epsilon=0.1 ([0,1] → [0.1, 0.9])
Class weights: Enabled (balanced)
```

### Data Loading
```
Loaded: 3,908 samples (from last 30 days)

BEFORE BALANCING:
  WIN:   2,361 (60.4%)
  LOSS:  1,547 (39.6%)

AFTER BALANCING (P0.6 fix):
  WIN:   1,547 (50.0%)  ← Undersampled
  LOSS:  1,547 (50.0%)
  Total: 3,094 samples

Sequences: (3,094, 128, 8)

Data splits:
  Train: 2,165 (70%)
  Val:   464 (15%)
  Test:  465 (15%)
```

✅ Balanced sampling worked correctly (50/50 split achieved)

### Training Progress
```
Epoch    Train Loss   Val Loss   Val Acc    Val F1    Conf Std
─────────────────────────────────────────────────────────────
1        0.7061       0.7102     51.72%     0.0000    0.0005
2        0.7023       0.6966     48.28%     0.6512    0.0002
3        0.6961       0.7200     48.28%     0.6512    0.0001
...
20       0.6947       0.6965     48.28%     0.6512    0.0000
─────────────────────────────────────────────────────────────
Best Val F1: 0.6512
Training time: 24.9s
```

**Analysis**: 
- Loss plateaued around 0.69 (not converging properly)
- Validation accuracy stuck at ~48-52% (random chance)
- Confidence stddev collapsed to 0.0000 by epoch 20
- Model did not learn meaningful patterns

---

## 3. TEST SET EVALUATION (❌ FAILED)

### Classification Metrics
```
Accuracy:  51.83%  (barely better than random)
Precision: 51.83%
Recall:    100.00%  (predicts all as WIN)
F1 Score:  0.6827
```

### Confusion Matrix
```
              Predicted
             LOSS    WIN
Actual LOSS     0    224  ← All LOSS predicted as WIN
       WIN      0    241  ← All WIN predicted as WIN
```

**Analysis**: Model predicts **everything as WIN** (binary class 1), which maps to **HOLD** action in inference.

### Action Distribution
```
📊 ACTION DISTRIBUTION (P0.6):
  BUY   :    0 (  0.0%)
  SELL  :    0 (  0.0%)
  HOLD  :  465 (100.0%)  ← DEGENERATE
```

❌ **CRITICAL FAILURE**: 100% HOLD predictions (no BUY/SELL)

### Confidence Statistics
```
📈 CONFIDENCE STATISTICS (P0.6):
  Mean:       0.5239
  Stddev:     0.0000  ← FLATLINED
  Min:        0.5238
  P10:        0.5239
  P50:        0.5239
  P90:        0.5239
  Max:        0.5239
  P10-P90:    0.0000  ← NO SPREAD
  Unique:     2 values only
```

❌ **CRITICAL FAILURE**: Confidence collapsed to constant (0.5239 for all predictions)

---

## 4. GATE CHECKS (❌ BOTH FAILED)

### Gate 1: Action Diversity
```
Requirement:
  ✓ No class >70%
  ✓ ≥2 classes >10%

Actual:
  ❌ Max action: HOLD (100.0% > 70%)
  ❌ Classes >10%: 1 (need ≥2)

Result: ❌ FAIL
```

### Gate 2: Confidence Spread
```
Requirement:
  ✓ Stddev ≥0.05
  ✓ P10-P90 ≥0.12

Actual:
  ❌ Stddev: 0.0000 (<0.02 minimum)
  ❌ P10-P90: 0.0000 (<0.05 minimum)

Result: ❌ FAIL
```

### Gate Summary
```
Gate 1 (Action Diversity):  ❌ FAIL (2/2 checks failed)
Gate 2 (Confidence Spread): ❌ FAIL (2/2 checks failed)
─────────────────────────────────────────────────────────
Overall:                     ❌ FAIL (4/4 checks failed)
```

---

## 5. ROOT CAUSE ANALYSIS

### What Went Wrong

1. **Model Collapsed to Constant Prediction**
   - Outputs constant probability (0.5239) for all inputs
   - Binary classifier "gave up" and predicts same value always
   - Label smoothing did not prevent collapse

2. **Sigmoid Saturation**
   - Final layer sigmoid outputs stuck at ~0.52
   - Not responding to input variations
   - Gradient flow problem during backprop

3. **Insufficient Learning**
   - 20 epochs not enough for complex time series
   - Learning rate 0.0003 too high (overshoots minimum)
   - Model parameters (612k) underutilized

4. **Feature Quality Issues** (suspected)
   - Only 8 features per timestep (very low)
   - Features may not have enough signal
   - Need to verify feature engineering

### Why Balanced Sampling Didn't Help

Balanced sampling **worked correctly** (50/50 split achieved), but:
- Model still learned to predict constant
- Issue is not class imbalance, but **feature quality** or **architecture**
- Label smoothing [0.1, 0.9] not enough to force diversity

---

## 6. COMPARISON: P0.4 vs P0.6

| Metric | P0.4 (Original) | P0.6 (This Attempt) | Target |
|--------|-----------------|---------------------|--------|
| BUY % | 87-89% | 0% | 35-45% |
| SELL % | 6% | 0% | 25-35% |
| HOLD % | 6% | 100% | 20-30% |
| Conf Std | 0.003 | 0.0000 | ≥0.05 |
| P10-P90 | 0.0018 | 0.0000 | ≥0.12 |

**Analysis**: P0.6 is **WORSE** than P0.4:
- P0.4: At least predicted BUY (though too much)
- P0.6: Completely degenerate (100% HOLD, zero confidence spread)

---

## 7. HYPERPARAMETER TUNING RECOMMENDATIONS

### Immediate Adjustments (P0.7 Attempt)

1. **Label Smoothing**: Increase epsilon to 0.15 or 0.2
   ```python
   LABEL_SMOOTHING = 0.15  # [0,1] → [0.15, 0.85]
   ```

2. **Dropout**: Increase to prevent overfitting
   ```python
   DROPOUT = 0.2  # Was likely 0.1
   ```

3. **Learning Rate**: Reduce for stable convergence
   ```python
   LEARNING_RATE = 0.0001  # Was 0.0003
   ```

4. **Epochs**: Increase training time
   ```python
   EPOCHS = 30  # Was 20
   ```

5. **Add Confidence Loss Term**: Penalize low confidence spread
   ```python
   # Add to loss function:
   confidence_penalty = -torch.std(outputs)  # Encourage diversity
   total_loss = bce_loss + 0.1 * confidence_penalty
   ```

### Architectural Changes (P0.8+)

1. **Feature Engineering**:
   - Currently only 8 features (very low)
   - Add: RSI, MACD, Bollinger Bands, volume indicators
   - Target: 20-30 features minimum

2. **Model Depth**:
   - Add more layers or attention heads
   - Increase hidden dimension

3. **Ensemble Training**:
   - Train multiple models with different seeds
   - Average predictions to increase diversity

---

## 8. DEPLOYMENT DECISION

### Hard Stop Conditions Met
- ❌ Gate 1 failed (action diversity)
- ❌ Gate 2 failed (confidence spread)
- ❌ Model is degenerate (constant predictions)

### Decision Matrix
```
IF Gate 1 PASS AND Gate 2 PASS:
  → Deploy to shadow mode
ELSE:
  → STOP, do not deploy
```

**Result**: ❌ **DO NOT DEPLOY P0.6**

### Rationale
1. Model produces no actionable signals (100% HOLD)
2. Zero confidence differentiation (cannot filter by confidence)
3. Worse than P0.4 (which at least predicted BUY, even if biased)
4. Would waste compute in shadow mode (no learning possible)

---

## 9. CURRENT STATE

### VPS Status
- **Training log**: `/tmp/patchtst_p06_training.log`
- **Model file**: NOT saved (script exited without saving)
- **AI engine**: Still running P0.4 model in shadow mode ✅
- **Service**: `quantum-ai-engine.service` active, unchanged

### No Changes Made
- ❌ No model copied to `/opt/quantum/ai_engine/models/`
- ❌ No env file updated
- ❌ No service restart
- ✅ P0.4 shadow mode still active and safe

---

## 10. NEXT STEPS

### Immediate (P0.7 Retraining)

1. **Investigate Features**
   ```bash
   # Check what 8 features are being used
   sqlite3 /opt/quantum/data/quantum_trader.db \
     "SELECT feature_names FROM ai_training_samples LIMIT 1"
   ```

2. **Adjust Hyperparameters** (apply recommendations from Section 7)

3. **Add Confidence Regularization** (custom loss term)

4. **Retrain with P0.7 config**

### Medium-Term (Feature Engineering)

1. **Expand Feature Set**:
   - Technical indicators (RSI, MACD, BB, ATR)
   - Volume profile
   - Order book features
   - Funding rate

2. **Verify Feature Quality**:
   - Check correlation with target
   - Remove redundant features
   - Normalize properly

### Long-Term (Architecture)

1. **Experiment with Architecture**:
   - Try different patch sizes
   - Adjust attention heads
   - Add more encoder layers

2. **Consider Alternative Models**:
   - LSTM with attention
   - Transformer encoder-only
   - TabNet for tabular features

---

## 11. LESSONS LEARNED

### What Worked
✅ Balanced sampling (50/50 split achieved)  
✅ Database path fix (6,000 samples loaded)  
✅ Sanity checks caught failure (prevented bad deployment)  
✅ Systemd-only workflow (no Docker needed)

### What Didn't Work
❌ Label smoothing alone insufficient  
❌ Class weights didn't prevent collapse  
❌ 20 epochs too short for convergence  
❌ Learning rate too high (0.0003)  
❌ Only 8 features (too sparse)

### Key Insight
**Class imbalance was NOT the only problem**. Even with 50/50 balanced data, model collapsed due to:
- Insufficient feature richness
- Hyperparameter choices
- Possible architecture limitations

The original P0.4 BUY bias (87%) was partly due to imbalance, but also partly due to the model **learning a strong directional pattern** from features. P0.6 balanced data removed that pattern, but model couldn't learn anything else.

---

## 12. FINAL PROOF PACK

### Training Execution
```
✅ Git HEAD: bff8e6ae
✅ Database: /opt/quantum/data/quantum_trader.db
✅ Samples loaded: 3,908 (30-day window)
✅ Balanced: 3,094 (50/50 WIN/LOSS)
✅ Training completed: 24.9s (20 epochs)
```

### Gate Results
```
Gate 1 (Action Diversity):
  Max action: HOLD 100.0% (❌ >70%)
  Classes >10%: 1 (❌ need ≥2)
  Result: ❌ FAIL

Gate 2 (Confidence Spread):
  Stddev: 0.0000 (❌ <0.02)
  P10-P90: 0.0000 (❌ <0.05)
  Result: ❌ FAIL

Overall: ❌ 0/2 gates passed
```

### Model Status
```
❌ Model file: NOT SAVED (sanity checks failed)
❌ Deployment: BLOCKED (gates failed)
✅ Current: P0.4 still running in shadow mode
```

### Service Status
```
✅ quantum-ai-engine.service: active
✅ PatchTST shadow mode: enabled
✅ Model: patchtst_v20260109_233444.pth (P0.4)
✅ No changes made to production
```

---

## CONCLUSION

**P0.6 training FAILED and was NOT deployed.**

Gates 1 and 2 both failed due to degenerate model behavior (100% HOLD, zero confidence spread). The model collapsed to constant predictions despite balanced sampling and label smoothing.

**Recommendation**: Adjust hyperparameters (increase label smoothing to 0.15, reduce learning rate to 0.0001, increase epochs to 30, add confidence regularization) and retrain as **P0.7**.

**Current State**: P0.4 remains deployed in shadow mode. System is safe and stable.

---

**Report Generated**: 2026-01-10 03:45 UTC  
**Status**: ✅ FAILURE DOCUMENTED - READY FOR P0.7 PLANNING
