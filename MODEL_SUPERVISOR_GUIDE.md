# MODEL SUPERVISOR — Complete Guide

**Version:** 1.0  
**Date:** November 23, 2025  
**Status:** ✅ DEPLOYED & OPERATIONAL

---

## 🎯 MISSION

The **Model Supervisor** oversees AI models and ensembles for Quantum Trader. It monitors performance, detects drift, ranks models, and recommends retraining or reweighting.

**Key Principle:** Supervise MODELS, not trades. Ensure models stay performant and well-calibrated.

---

## 📊 WHAT DOES MODEL SUPERVISOR DO?

### Core Responsibilities

1. **Per-Model Performance Tracking**
   - Winrate, Avg R, Total R
   - Calibration quality (confidence vs reality)
   - Regime-specific performance
   - Performance trends (improving/stable/degrading)

2. **Ensemble Weight Optimization**
   - Overall weights based on performance
   - Regime-specific weights (TRENDING vs RANGING)
   - Dynamic reweighting recommendations

3. **Drift Detection**
   - Performance degradation over time
   - Calibration breakdown
   - Regime-specific weaknesses

4. **Retraining Recommendations**
   - Which models need retraining
   - Priority levels (URGENT/HIGH/MEDIUM/LOW)
   - Specific improvement targets
   - Actionable suggestions

---

## 🏗️ ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL SUPERVISOR                          │
└─────────────────────────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │      INPUT: SIGNAL LOGS + OUTCOMES      │
        ├────────────────────────────────────────┤
        │ • model_id, prediction, confidence     │
        │ • realized_R, realized_pnl             │
        │ • regime_tag, vol_level                │
        │ • timestamps                           │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │      PER-MODEL METRICS COMPUTATION      │
        ├────────────────────────────────────────┤
        │ • Winrate, Avg R, Total R              │
        │ • Calibration quality                  │
        │ • Regime-specific performance          │
        │ • Recent performance (7 days)          │
        │ • Performance trend                    │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │         MODEL HEALTH ASSESSMENT         │
        ├────────────────────────────────────────┤
        │ • HEALTHY: All metrics good            │
        │ • DEGRADED: Some issues                │
        │ • CRITICAL: Multiple failures          │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │           MODEL RANKING                 │
        ├────────────────────────────────────────┤
        │ • Score = 0.35*WR + 0.35*R + 0.20*Cal  │
        │ • Sort by score (highest = best)       │
        │ • Assign recommended weights           │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │      ENSEMBLE WEIGHT SUGGESTIONS        │
        ├────────────────────────────────────────┤
        │ • Overall weights                      │
        │ • Regime-specific weights              │
        │ • Reasoning & justification            │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │     RETRAIN RECOMMENDATIONS             │
        ├────────────────────────────────────────┤
        │ • Priority (URGENT/HIGH/MEDIUM/LOW)    │
        │ • Reasons for retraining               │
        │ • Target improvements                  │
        │ • Actionable suggestions               │
        └────────────────────────────────────────┘
```

---

## 🔧 INTEGRATION

### Step 1: Import

```python
from backend.services.model_supervisor import (
    ModelSupervisor,
    SignalLog,
    ModelMetrics,
    ModelHealth
)
```

### Step 2: Initialize

```python
supervisor = ModelSupervisor(
    data_dir="/app/data",
    analysis_window_days=30,  # Analyze last 30 days
    recent_window_days=7      # "Recent" = last 7 days
)
```

### Step 3: Prepare Signal Logs

```python
# Collect signal logs with outcomes
signal_logs = []

for trade in completed_trades:
    signal_logs.append({
        "timestamp": trade.entry_time,
        "model_id": trade.model_used,  # e.g., "xgboost_v1"
        "symbol": trade.symbol,
        "prediction": trade.signal_action,  # BUY/SELL/HOLD
        "confidence": trade.signal_confidence,
        "regime_tag": trade.regime_at_entry,
        "vol_level": trade.volatility_at_entry,
        "realized_R": trade.R_multiple,
        "realized_pnl": trade.pnl,
        "outcome_known": True,
        "outcome_timestamp": trade.exit_time
    })
```

### Step 4: Run Analysis

```python
output = supervisor.analyze_models(
    signal_logs=signal_logs,
    model_metadata=None  # Optional: training dates, hyperparams
)
```

### Step 5: Use Output

```python
# Check model health
print(f"Healthy: {output.healthy_models}")
print(f"Degraded: {output.degraded_models}")
print(f"Critical: {output.critical_models}")

# Update ensemble weights
for model_id, weight in output.ensemble_weights.overall_weights.items():
    ensemble.set_weight(model_id, weight)

# Regime-specific weights
for regime, weights in output.ensemble_weights.regime_weights.items():
    ensemble.set_regime_weights(regime, weights)

# Handle retrain recommendations
for rec in output.retrain_recommendations:
    if rec.priority == "URGENT":
        logger.error(f"🚨 URGENT: Retrain {rec.model_id}")
        # Trigger retraining immediately
        await retrain_model(rec.model_id)
    elif rec.priority == "HIGH":
        # Schedule retraining soon
        schedule_retrain(rec.model_id, priority="high")
```

---

## 📊 PERFORMANCE THRESHOLDS

### Health Criteria

| Metric | Minimum | Impact |
|--------|---------|--------|
| **Winrate** | 50% | Critical if below |
| **Avg R** | 0.0 | Critical if negative |
| **Calibration** | 70% | Degraded if below |
| **Sample Count** | 20 | Warning if below |

### Health Status

| Status | Criteria | Action |
|--------|----------|--------|
| **HEALTHY** | All metrics above minimums | Continue normal operation |
| **DEGRADED** | 1 metric below OR 2+ warnings | Monitor closely, consider retrain |
| **CRITICAL** | 2+ metrics below minimums | Retrain urgently or disable |

### Performance Trends

| Trend | Detection | Meaning |
|-------|-----------|---------|
| **IMPROVING** | +15% R in second half | Model getting better |
| **STABLE** | ±15% R between halves | Model consistent |
| **DEGRADING** | -15% R in second half | Model drifting, needs retrain |

---

## 🎯 MODEL RANKING FORMULA

```python
overall_score = (
    winrate * 0.35 +              # 35% weight on winrate
    max(0, avg_R) * 0.35 +        # 35% weight on avg R (non-negative)
    calibration_quality * 0.20 +   # 20% weight on calibration
    (profit_factor / 10) * 0.10    # 10% weight on profit factor
)

# Penalties
if predictions < MIN_SAMPLES:
    score *= 0.5
if health == "DEGRADED":
    score *= 0.8
if health == "CRITICAL":
    score *= 0.5

# Rank by score (highest = best)
recommended_weight = score / total_score
```

---

## 📈 ENSEMBLE WEIGHT SUGGESTIONS

### Overall Weights

Based on model rankings:
```python
{
    "xgboost_v1": 0.40,    # Best performer
    "ensemble_v2": 0.35,   # Second best
    "lstm_v1": 0.15,       # Weak, reduced weight
    "transformer_v1": 0.10 # New, conservative weight
}
```

### Regime-Specific Weights

Optimized per regime:
```python
"TRENDING": {
    "xgboost_v1": 0.50,    # Strongest in trends
    "ensemble_v2": 0.35,
    "lstm_v1": 0.15
},
"RANGING": {
    "lstm_v1": 0.45,       # Better in ranges
    "xgboost_v1": 0.35,
    "ensemble_v2": 0.20
}
```

---

## 🔍 CALIBRATION QUALITY

**Calibration** measures how well confidence scores match reality.

### Perfect Calibration
- 70% confidence → 70% success rate
- 80% confidence → 80% success rate
- 90% confidence → 90% success rate

### Poor Calibration Examples
- 70% confidence → 40% success rate (overconfident)
- 70% confidence → 95% success rate (underconfident)

### Calibration Formula

```python
# Group predictions by confidence buckets (0.6, 0.7, 0.8, etc.)
for confidence_level, outcomes in buckets:
    actual_success_rate = sum(outcomes) / len(outcomes)
    error = abs(confidence_level - actual_success_rate)

calibration_quality = 1.0 - mean(errors)
```

---

## 🚨 RETRAIN PRIORITIES

### URGENT
**Triggers:**
- Model health = CRITICAL
- Winrate < 40%
- Avg R < -0.5

**Action:** Retrain immediately or disable model

### HIGH
**Triggers:**
- Model health = DEGRADED
- Winrate < 50%
- Calibration < 60%

**Action:** Schedule retrain within 1-3 days

### MEDIUM
**Triggers:**
- Performance = DEGRADING
- Calibration < 70%

**Action:** Schedule retrain within 1-2 weeks

### LOW
**Triggers:**
- Specific regime weakness
- Minor calibration issues

**Action:** Include in next scheduled retrain cycle

---

## 📊 OUTPUT STRUCTURE

### SupervisorOutput

```python
{
    "timestamp": "2025-11-23T12:00:00Z",
    "analysis_period_days": 30,
    
    "model_metrics": {
        "xgboost_v1": {
            "total_predictions": 150,
            "predictions_with_outcome": 120,
            "winrate": 0.60,
            "avg_R": 0.18,
            "calibration_quality": 0.75,
            "health_status": "HEALTHY",
            "performance_trend": "STABLE",
            "regime_performance": {
                "TRENDING": {"winrate": 0.65, "avg_R": 0.25},
                "RANGING": {"winrate": 0.55, "avg_R": 0.10}
            }
        }
    },
    
    "model_rankings": [
        {
            "model_id": "xgboost_v1",
            "rank": 1,
            "overall_score": 0.456,
            "recommended_weight": 0.40
        }
    ],
    
    "ensemble_weights": {
        "overall_weights": {"xgboost_v1": 0.40, "lstm_v1": 0.35},
        "regime_weights": {
            "TRENDING": {"xgboost_v1": 0.50},
            "RANGING": {"lstm_v1": 0.45}
        }
    },
    
    "healthy_models": ["xgboost_v1", "ensemble_v2"],
    "degraded_models": [],
    "critical_models": ["lstm_v1"],
    
    "retrain_recommendations": [
        {
            "model_id": "lstm_v1",
            "priority": "URGENT",
            "reasons": ["Winrate below minimum", "Avg R negative"],
            "suggested_actions": [
                "Focus on improving prediction accuracy",
                "Review risk/reward ratios"
            ]
        }
    ],
    
    "summary": {
        "total_models": 3,
        "healthy_models": 2,
        "critical_models": 1,
        "avg_winrate": 0.58,
        "best_model": "xgboost_v1"
    }
}
```

---

## 🔄 RECOMMENDED WORKFLOW

### Daily Analysis

```python
# Run daily at 03:00 UTC
async def daily_model_analysis():
    # 1. Collect signal logs from last 30 days
    signal_logs = await get_signal_logs_with_outcomes(days=30)
    
    # 2. Run analysis
    output = supervisor.analyze_models(signal_logs)
    
    # 3. Check for critical issues
    if output.critical_models:
        await alert_admin(f"Critical models: {output.critical_models}")
    
    # 4. Update ensemble weights
    await update_ensemble_weights(output.ensemble_weights)
    
    # 5. Process retrain recommendations
    for rec in output.retrain_recommendations:
        if rec.priority == "URGENT":
            await trigger_retrain(rec.model_id, urgent=True)
        elif rec.priority == "HIGH":
            await schedule_retrain(rec.model_id, days=2)
    
    # 6. Save report
    await save_model_health_report(output)
```

### Weekly Review

```python
# Review trends weekly
async def weekly_model_review():
    # Compare last week vs previous weeks
    current = await get_latest_supervisor_output()
    previous = await get_supervisor_output(days_ago=7)
    
    # Check for deteriorating models
    for model_id in current.model_metrics:
        current_wr = current.model_metrics[model_id].winrate
        previous_wr = previous.model_metrics[model_id].winrate
        
        if current_wr < previous_wr - 0.10:  # 10% drop
            logger.warning(f"{model_id} winrate dropped {previous_wr:.1%} → {current_wr:.1%}")
            await consider_emergency_retrain(model_id)
```

---

## 🧪 TESTING

Run tests:

```bash
cd backend
python services/model_supervisor.py
```

Expected output:
```
Model Rankings:
  1. ensemble_v2: Score=0.482, WR=60.0%, R=0.200, Weight=46.6%
  2. xgboost_v1: Score=0.411, WR=60.0%, R=0.180, Weight=39.7%
  3. lstm_v1: Score=0.143, WR=40.0%, R=-0.120, Weight=13.8%

⚠️ RETRAIN RECOMMENDATIONS:
  [URGENT] lstm_v1: Winrate below minimum, Avg R negative
  [MEDIUM] xgboost_v1: Poor calibration, Performance degrading
```

---

## 📁 FILES

| File | Purpose |
|------|---------|
| `backend/services/model_supervisor.py` | Main implementation |
| `/app/data/model_supervisor_output.json` | Latest analysis output |
| `MODEL_SUPERVISOR_GUIDE.md` | This guide |

---

## 💡 BEST PRACTICES

1. **Run Daily** - Catch drift early
2. **Monitor Calibration** - Poor calibration = unreliable confidence
3. **Act on URGENT** - Don't delay critical retrains
4. **Update Weights** - Apply ensemble weight suggestions
5. **Track Trends** - Weekly reviews catch slow degradation

---

## 🚨 COMMON ISSUES

### Model Always CRITICAL
**Cause:** Poor training data or wrong hyperparameters  
**Fix:** Retrain with better data, tune hyperparameters

### Poor Calibration
**Cause:** Confidence scores not calibrated properly  
**Fix:** Apply calibration scaling (Platt scaling, isotonic regression)

### Degrading Performance
**Cause:** Market regime change, model drift  
**Fix:** Retrain with recent data, add new features

### Low Sample Count
**Cause:** Not enough completed trades  
**Fix:** Wait for more data (need 20+ samples minimum)

---

**Status:** ✅ READY FOR DEPLOYMENT  
**Version:** 1.0  
**Last Updated:** November 23, 2025
