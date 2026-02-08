# META-AGENT V2: Production Guide

## Overview

Meta-Agent V2 is a **regime-aware stacked ensemble** that learns when to trust base model predictions vs when to defer to weighted ensemble voting.

### Key Features

- ✅ **Safety-first design**: Only overrides when confident
- ✅ **Explainable**: Every decision includes reason  
- ✅ **Regime-aware**: Adapts to market conditions
- ✅ **Self-monitoring**: Tracks override rate and fallback reasons
- ✅ **Fail-safe**: Multiple safety checks prevent silent failures

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRADING SIGNAL FLOW                      │
└─────────────────────────────────────────────────────────────┘

    Market Data
        ↓
┌──────────────────┐
│  Base Agents     │
│  - XGBoost       │
│  - LightGBM      │
│  - N-HiTS        │
│  - PatchTST      │
└────────┬─────────┘
         │
         ↓
┌────────────────────────────────────┐
│  Meta-Agent V2                     │
│  ┌──────────────────────────────┐  │
│  │ Feature Extraction           │  │
│  │ - Base signals (4 models)    │  │
│  │ - Aggregate stats            │  │
│  │ - Voting distribution        │  │
│  │ - Regime features            │  │
│  └──────────┬───────────────────┘  │
│             ↓                      │
│  ┌──────────────────────────────┐  │
│  │ Decision Logic               │  │
│  │ - Check strong consensus     │  │
│  │ - Compute meta confidence    │  │
│  │ - Apply threshold            │  │
│  └──────────┬───────────────────┘  │
│             ↓                      │
│  ┌──────────────────────────────┐  │
│  │ Output Decision              │  │
│  │ - use_meta: bool             │  │
│  │ - action: SELL|HOLD|BUY      │  │
│  │ - confidence: float          │  │
│  │ - reason: str                │  │
│  └──────────────────────────────┘  │
└────────────────────────────────────┘
         │
         ↓
    Final Signal
```

---

## Quick Start

### Prerequisites

- ✅ All 4 base agents must be active (XGBoost, LightGBM, N-HiTS, PatchTST)
- ✅ Prediction logs available (JSONL format)
- ✅ Trade outcome data (optional but recommended)
- ✅ Minimum 1000 training samples

### 1. Training

```bash
# Activate environment
source /opt/quantum/venvs/ai-engine/bin/activate

# Run training script
cd /home/qt/quantum_trader
python ops/retrain/train_meta_v2.py
```

**Expected output:**
```
[INFO] Loading prediction logs...
[INFO] Loaded 2534 prediction records
[INFO] Date range: 2026-02-05 to 2026-02-06
[INFO] Extracting features for training...
[INFO] Feature matrix shape: (2534, 26)
[INFO] Time-series CV (5 splits)...
[INFO]   Fold 1: accuracy=0.6842
[INFO]   Fold 2: accuracy=0.7021
[INFO]   Fold 3: accuracy=0.6954
[INFO]   Fold 4: accuracy=0.7103
[INFO]   Fold 5: accuracy=0.6889
[INFO] CV accuracy: 0.6962 ± 0.0098
[INFO] Training final model...
[INFO] Calibrating probabilities...
[INFO] Test accuracy: 0.6845
✅ Model saved: /home/qt/quantum_trader/ai_engine/models/meta_v2/meta_model.pkl
✅ Scaler saved: /home/qt/quantum_trader/ai_engine/models/meta_v2/scaler.pkl
✅ Metadata saved: /home/qt/quantum_trader/ai_engine/models/meta_v2/metadata.json
```

### 2. Deployment

Meta-Agent V2 is auto-loaded by ensemble_manager.py when available.

#### Enable Meta-Agent

```bash
# Edit service environment
sudo nano /etc/systemd/system/quantum-ai-engine.service

# Add environment variables:
Environment="META_AGENT_ENABLED=true"
Environment="META_OVERRIDE_THRESHOLD=0.65"
Environment="META_FAIL_OPEN=true"

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart quantum-ai-engine
```

#### Verify Deployment

```bash
# Check logs for meta-agent initialization
journalctl -u quantum-ai-engine -n 50 | grep META

# Expected output:
[✅ META-V2] Meta-learning agent loaded (5th layer)
   └─ Override threshold: 0.65
   └─ Version: v2
```

### 3. Monitoring

```bash
# Watch live meta decisions
journalctl -u quantum-ai-engine -f | grep 'META-V2'

# Output examples:
[META-V2] BTCUSDT: OVERRIDE (base=HOLD@0.68 → meta=BUY@0.82) | Reason: meta_override_buy
[META-V2] ETHUSDT: FALLBACK (keeping base=HOLD@0.75) | Reason: strong_consensus_hold
[META-V2] BNBUSDT: FALLBACK (keeping base=BUY@0.70) | Reason: meta_low_confidence_0.58
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `META_AGENT_ENABLED` | `false` | Enable/disable meta-agent |
| `META_OVERRIDE_THRESHOLD` | `0.65` | Minimum confidence for override (V2: 0.65, V1: 0.99) |
| `META_FAIL_OPEN` | `true` | On error: use base ensemble (true) or force HOLD (false) |
| `META_V2_MODEL_DIR` | `/home/qt/quantum_trader/ai_engine/models/meta_v2` | Model directory |

### Model Parameters

Edit `train_meta_v2.py` CONFIG section:

```python
CONFIG = {
    # Training window
    'min_date': '2026-02-05',  # Only data after PyTorch fix
    
    # Model hyperparameters
    'C': 0.1,  # L2 regularization (smaller = stronger)
    'class_weight': 'balanced',  # Handle class imbalance
    
    # Thresholds
    'meta_threshold': 0.65,  # Override threshold
    'consensus_threshold': 0.75,  # Strong consensus (defer to ensemble)
    'max_override_rate': 0.40,  # Warn if meta overrides >40%
    
    # Validation
    'min_samples': 1000,  # Minimum training samples
    'min_accuracy': 0.55,  # Minimum acceptable accuracy
}
```

---

## Decision Logic

### When Meta Overrides

Meta-agent will **override** base ensemble when:

1. ✅ Model is loaded and ready
2. ✅ No strong consensus (< 75% agreement)
3. ✅ Meta confidence >= threshold (default 0.65)
4. ✅ All safety checks pass

### When Meta Defers (Fallback)

Meta-agent will **defer** to base ensemble when:

| Reason | Description |
|--------|-------------|
| `model_not_loaded` | Model file not found or failed validation |
| `strong_consensus_buy/sell/hold` | Base agents agree >= 75% |
| `meta_low_confidence_X` | Meta confidence < threshold |
| `feature_dim_mismatch` | Input feature count != expected |
| `invalid_confidence` | Meta confidence outside [0, 1] |
| `feature_error` | Feature extraction failed |
| `prediction_error` | Model prediction failed |

---

## Feature Engineering

### Input Features (26 total)

**Base Agent Signals (16 features):**
- XGBoost: [sell, hold, buy, confidence]
- LightGBM: [sell, hold, buy, confidence]
- N-HiTS: [sell, hold, buy, confidence]
- PatchTST: [sell, hold, buy, confidence]

**Aggregate Statistics (4 features):**
- mean_confidence
- max_confidence
- min_confidence
- confidence_std

**Voting Distribution (4 features):**
- vote_buy_pct (% of agents voting BUY)
- vote_sell_pct (% of agents voting SELL)
- vote_hold_pct (% of agents voting HOLD)
- disagreement (1 - most_common_count / total)

**Regime Features (2 features):**
- volatility (0.0-1.0)
- trend_strength (-1.0 to 1.0)

### Feature Example

```python
{
    'xgb_action': 'BUY',
    'xgb_conf': 0.70,
    'lgbm_action': 'HOLD',
    'lgbm_conf': 0.65,
    'nhits_action': 'BUY',
    'nhits_conf': 0.82,
    'patchtst_action': 'BUY',
    'patchtst_conf': 0.75,
    'mean_confidence': 0.73,
    'max_confidence': 0.82,
    'min_confidence': 0.65,
    'confidence_std': 0.068,
    'vote_buy_pct': 0.75,     # 3/4 agents vote BUY
    'vote_sell_pct': 0.0,
    'vote_hold_pct': 0.25,
    'disagreement': 0.25,      # 1 - (3/4) = 0.25
    'volatility': 0.52,
    'trend_strength': 0.15
}
```

---

## Safety Guarantees

### Runtime Checks

Meta-Agent V2 implements multiple safety layers:

1. **Parameter Count Validation**
   ```python
   param_count = sum(p.numel() for p in model.parameters())
   assert param_count > 0
   ```

2. **Non-Constant Output Test**
   ```python
   out1 = model(random_input_1)
   out2 = model(random_input_2)
   assert not torch.allclose(out1, out2)
   ```

3. **Feature Dimension Check**
   ```python
   if X.shape[1] != expected_feature_dim:
       fallback_to_ensemble()
   ```

4. **Confidence Bounds**
   ```python
   assert 0.0 <= confidence <= 1.0
   ```

5. **Override Rate Monitoring**
   ```python
   if override_rate > max_override_rate:
       logger.warning("Meta may be too aggressive")
   ```

### Fail-Safe Modes

**Fail-Open (default):**
- On any error → use base ensemble
- System continues operating
- Error logged for investigation

**Fail-Closed (optional):**
- On any error → force HOLD
- Conservative but may miss opportunities
- Set `META_FAIL_OPEN=false`

---

## Monitoring & Diagnostics

### Key Metrics

```bash
# Override rate (should be 10-30%)
journalctl -u quantum-ai-engine --since='1 hour ago' | grep 'META-V2.*OVERRIDE' | wc -l

# Fallback reasons distribution
journalctl -u quantum-ai-engine --since='1 hour ago' | grep 'META-V2.*FALLBACK' | grep -oP 'Reason: \K\w+' | sort | uniq -c
```

### Health Checks

**Model Loaded:**
```bash
curl http://localhost:8001/api/ai/meta/status

# Expected response:
{
  "version": "v2",
  "model_ready": true,
  "total_predictions": 1523,
  "meta_overrides": 387,
  "override_rate": 0.254,
  "meta_threshold": 0.65,
  "consensus_threshold": 0.75
}
```

**Statistics Reset:**
```bash
curl -X POST http://localhost:8001/api/ai/meta/reset_stats
```

### Performance Monitoring

Track meta-agent contribution to system performance:

```python
# In ensemble_manager.py logs, compare:
# - base_ensemble_action vs meta_action
# - Actual outcome (win/loss)

# Good meta-agent should:
# - Override when base ensemble would lose
# - Defer when base ensemble is correct
# - Maintain override_rate: 15-30%
```

---

## Troubleshooting

### Meta-Agent Not Loading

**Symptoms:**
```
[SKIP] Meta agent not available (install required)
```

**Solution:**
```bash
# Check model files exist
ls -lh /home/qt/quantum_trader/ai_engine/models/meta_v2/

# Should show:
meta_model.pkl
scaler.pkl
metadata.json

# If missing, retrain:
python ops/retrain/train_meta_v2.py
```

### High Override Rate (>40%)

**Symptoms:**
```
[MetaV2] ⚠️  High override rate: 52% (threshold: 40%)
```

**Solution:**
```bash
# Increase meta threshold
Environment="META_OVERRIDE_THRESHOLD=0.75"  # Was 0.65

# OR retrain with stricter regularization
# Edit train_meta_v2.py:
CONFIG['C'] = 0.05  # Was 0.1 (stronger regularization)
```

### Constant Fallback

**Symptoms:**
```
[META-V2] FALLBACK ... | Reason: meta_low_confidence_0.42
[META-V2] FALLBACK ... | Reason: meta_low_confidence_0.38
```

**Solution:**
```bash
# Lower meta threshold
Environment="META_OVERRIDE_THRESHOLD=0.55"  # Was 0.65

# OR retrain with more data
# Collect more samples from live trading logs
```

### Model Validation Failed

**Symptoms:**
```
[MetaV2] ❌ VALIDATION FAILED: Model produces constant output
```

**Solution:**
```bash
# Model is broken - retrain from scratch
rm -rf /home/qt/quantum_trader/ai_engine/models/meta_v2/*
python ops/retrain/train_meta_v2.py

# Check training accuracy >= 0.55
# If below, collect more diverse training data
```

---

## Retraining Schedule

### When to Retrain

- ✅ **Weekly**: Normal schedule for continuous improvement
- ✅ **After base model update**: When XGBoost/LightGBM/N-HiTS/PatchTST retrained
- ✅ **Performance degradation**: If meta accuracy drops >5% in production
- ✅ **Market regime shift**: After significant market structure change

### Retraining Procedure

```bash
# 1. Backup current model
cp -r /home/qt/quantum_trader/ai_engine/models/meta_v2 \
      /home/qt/quantum_trader/ai_engine/models/meta_v2.backup_$(date +%Y%m%d)

# 2. Run training
python ops/retrain/train_meta_v2.py

# 3. Validate new model
python ai_engine/tests/test_meta_agent_v2.py

# 4. Deploy (restart service)
sudo systemctl restart quantum-ai-engine

# 5. Monitor for 1 hour
journalctl -u quantum-ai-engine -f | grep META-V2
```

---

## Performance Expectations

### Accuracy Targets

| Metric | Target | Interpretation |
|--------|--------|----------------|
| CV Accuracy | ≥ 0.55 | Above random (0.33 for 3 classes) |
| Test Accuracy | ≥ 0.55 | Generalization check |
| Override Rate | 15-30% | Balance between intervention and trust |
| Consensus Respect | 100% | Always defer to strong base agreement |

### Regime-Specific Performance

Meta-agent should maintain accuracy across:
- **Low volatility**: ≥ 0.53
- **Medium volatility**: ≥ 0.57
- **High volatility**: ≥ 0.52
- **Trending markets**: ≥ 0.55
- **Ranging markets**: ≥ 0.56

---

## API Reference

### Prediction Interface

```python
from ai_engine.meta.meta_agent_v2 import MetaAgentV2

# Initialize
meta = MetaAgentV2(
    model_dir='/path/to/models/meta_v2',
    meta_threshold=0.65,
    consensus_threshold=0.75,
    enable_regime_features=True
)

# Predict
result = meta.predict(
    base_predictions={
        'xgb': {'action': 'BUY', 'confidence': 0.70},
        'lgbm': {'action': 'HOLD', 'confidence': 0.65},
        'nhits': {'action': 'BUY', 'confidence': 0.82},
        'patchtst': {'action': 'BUY', 'confidence': 0.75}
    },
    regime_info={
        'volatility': 0.52,
        'trend_strength': 0.15
    },
    symbol='BTCUSDT'
)

# Result format:
{
    'use_meta': True,
    'action': 'BUY',
    'confidence': 0.78,
    'reason': 'meta_override_buy',
    'meta_confidence': 0.78,
    'base_ensemble_action': 'BUY',
    'base_ensemble_confidence': 0.72
}
```

### Statistics Interface

```python
# Get statistics
stats = meta.get_statistics()
{
    'total_predictions': 1523,
    'meta_overrides': 387,
    'override_rate': 0.254,
    'fallback_reasons': {
        'strong_consensus': 612,
        'low_meta_confidence': 524,
        'model_not_loaded': 0
    },
    'model_ready': True,
    'meta_threshold': 0.65,
    'consensus_threshold': 0.75
}

# Reset statistics
meta.reset_statistics()
```

---

## Version History

### v2.0.0 (2026-02-06)
- ✅ Initial release
- ✅ Regime-aware decision making
- ✅ Strong consensus detection
- ✅ Multiple safety checks
- ✅ Explainable decisions
- ✅ Runtime statistics tracking

### Roadmap

**v2.1.0 (Planned):**
- Adaptive threshold based on recent performance
- Per-symbol meta models
- Volatility-adaptive confidence calibration

**v2.2.0 (Planned):**
- Online learning (incremental updates)
- Multi-horizon predictions (1m, 5m, 15m)
- Ensemble diversity metrics

---

## Support

**Issues & Questions:**
- GitHub: https://github.com/binyaminsemerci-ops/quantum_trader
- Documentation: /home/qt/quantum_trader/docs/meta_agent_v2.md

**Logs:**
```bash
# Meta-agent specific logs
journalctl -u quantum-ai-engine | grep META-V2

# Full AI engine logs
journalctl -u quantum-ai-engine -f
```

**Model Files:**
```bash
/home/qt/quantum_trader/ai_engine/models/meta_v2/
├── meta_model.pkl      # Trained logistic regression model
├── scaler.pkl          # Feature scaler (StandardScaler)
└── metadata.json       # Training metadata and metrics
```

---

**Last Updated:** 2026-02-06  
**Version:** 2.0.0  
**Status:** ✅ Production Ready
