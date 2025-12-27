# RETRAINING SYSTEM ORCHESTRATOR — Complete Guide

**Version:** 1.0  
**Date:** November 23, 2025  
**Status:** ✅ DEPLOYED & OPERATIONAL

---

## 🎯 MISSION

The **Retraining System Orchestrator** automates model retraining based on performance drift, schedule, and regime changes. It manages the complete lifecycle: trigger evaluation, training coordination, versioning, and safe deployment.

**Key Principle:** Retrain MODELS automatically when needed, with safe deployment policies and rollback capabilities.

---

## 📊 WHAT DOES THE ORCHESTRATOR DO?

### Core Responsibilities

1. **Trigger Evaluation**
   - Performance-driven (model degrading)
   - Time-driven (periodic schedule)
   - Regime-driven (market regime changed)
   - Drift detection
   - Manual override

2. **Training Pipeline Coordination**
   - Select data window (rolling)
   - Configure features and hyperparameters
   - Launch training jobs
   - Run cross-validation
   - Evaluate out-of-sample performance

3. **Model Versioning**
   - Track all model versions
   - Maintain training metadata
   - Store performance metrics
   - Enable rollback to previous versions

4. **Deployment Decisions**
   - Compare new vs old model
   - Safety checks (minimum thresholds)
   - Performance improvement analysis
   - Canary testing for moderate improvements
   - Auto-deploy or manual review

---

## 🏗️ ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────┐
│              RETRAINING SYSTEM ORCHESTRATOR                  │
└─────────────────────────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │      INPUT: MODEL SUPERVISOR OUTPUT     │
        ├────────────────────────────────────────┤
        │ • model_metrics (WR, R, calibration)   │
        │ • retrain_recommendations              │
        │ • health_status per model              │
        │ • performance_trends                   │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │         TRIGGER EVALUATION              │
        ├────────────────────────────────────────┤
        │ PERFORMANCE:                           │
        │   - Health CRITICAL → URGENT          │
        │   - Health DEGRADED → HIGH            │
        │   - Performance DEGRADING → MEDIUM    │
        │                                        │
        │ TIME:                                  │
        │   - Periodic (7 days) → LOW           │
        │                                        │
        │ REGIME:                                │
        │   - Regime changed → MEDIUM           │
        │                                        │
        │ MANUAL:                                │
        │   - Admin trigger → URGENT/HIGH/LOW   │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │        CREATE RETRAINING PLAN           │
        ├────────────────────────────────────────┤
        │ • Group triggers by model_id           │
        │ • Prioritize: URGENT > HIGH > ...     │
        │ • Create RetrainingJob per model       │
        │ • Estimate duration                    │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │       EXECUTE RETRAINING JOBS           │
        ├────────────────────────────────────────┤
        │ For each job:                          │
        │   1. Select training script            │
        │   2. Build command with hyperparams    │
        │   3. Launch subprocess                 │
        │   4. Monitor progress                  │
        │   5. Load new model version            │
        │   6. Register in version history       │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │       EVALUATE DEPLOYMENT               │
        ├────────────────────────────────────────┤
        │ Compare old vs new:                    │
        │   - Winrate improvement                │
        │   - Avg R improvement                  │
        │   - Calibration improvement            │
        │                                        │
        │ Safety checks:                         │
        │   - WR >= 50%                          │
        │   - Avg R >= 0.0                       │
        │   - Calibration >= 70%                 │
        │                                        │
        │ Decision:                              │
        │   - Improvement >= 5% → DEPLOY        │
        │   - Improvement 2-5% → CANARY         │
        │   - Improvement < 0% → KEEP OLD       │
        │   - Failed safety → KEEP OLD          │
        └────────────────────────────────────────┘
                           ▼
        ┌────────────────────────────────────────┐
        │           DEPLOY OR ROLLBACK            │
        ├────────────────────────────────────────┤
        │ • Deploy new version to production     │
        │ • Update ensemble weights              │
        │ • Keep old version for rollback        │
        │ • Log deployment event                 │
        └────────────────────────────────────────┘
```

---

## 🔧 INTEGRATION

### Step 1: Import

```python
from backend.services.retraining_orchestrator import (
    RetrainingOrchestrator,
    TriggerType,
    DeploymentDecision,
    ModelVersion
)
```

### Step 2: Initialize

```python
orchestrator = RetrainingOrchestrator(
    data_dir="/app/data",
    models_dir="/app/ai_engine/models",
    scripts_dir="/app/scripts",
    
    # Performance thresholds
    min_winrate=0.50,
    min_avg_r=0.0,
    min_calibration=0.70,
    
    # Deployment thresholds
    min_improvement_pct=0.05,  # 5% improvement to deploy
    canary_threshold_pct=0.02,  # 2-5% = canary test
    
    # Schedule
    periodic_retrain_days=7  # Weekly
)
```

### Step 3: Evaluate Triggers

```python
# Get Model Supervisor output
supervisor_output = model_supervisor.analyze_models(signal_logs)

# Evaluate triggers
triggers = orchestrator.evaluate_triggers(
    supervisor_output=supervisor_output,
    current_regime="TRENDING"  # Optional
)

print(f"Found {len(triggers)} retraining triggers")
```

### Step 4: Create & Execute Plan

```python
# Create retraining plan
plan = orchestrator.create_retraining_plan(
    triggers=triggers,
    batch_size=3  # Train 3 models in parallel
)

print(f"Plan: {plan.total_jobs} jobs, ~{plan.estimated_duration_minutes:.0f} min")

# Execute jobs (can run async)
for job in plan.jobs:
    completed_job = orchestrator.execute_retraining_job(job)
    
    if completed_job.status == RetrainingStatus.COMPLETED:
        print(f"[OK] {job.model_id} retrained successfully")
        
        # Evaluate deployment
        recommendation = orchestrator.evaluate_deployment(
            new_version=completed_job.new_version
        )
        
        if recommendation.decision == DeploymentDecision.DEPLOY_IMMEDIATELY:
            orchestrator.deploy_model(completed_job.new_version)
            print(f"[OK] Deployed {job.model_id} {completed_job.new_version.version_tag}")
        elif recommendation.decision == DeploymentDecision.RUN_CANARY:
            print(f"[INFO] Run canary test for {job.model_id}")
            # Run canary logic here
        elif recommendation.decision == DeploymentDecision.KEEP_OLD:
            print(f"[SKIP] Keeping old version for {job.model_id}")
```

### Step 5: Rollback if Needed

```python
# Rollback to previous version
success = orchestrator.rollback_model(
    model_id="xgboost_v1",
    target_version_tag="v20251120_120000"
)

if success:
    print("Rollback successful")
```

---

## 📊 TRIGGER TYPES

### 1. Performance-Driven

**Triggers:**
- Model health = CRITICAL → URGENT
- Model health = DEGRADED → HIGH
- Performance trend = DEGRADING → MEDIUM

**Example:**
```python
# Model Supervisor detects critical health
{
    "xgboost_v1": {
        "winrate": 0.45,  # Below 50% minimum
        "avg_R": -0.05,   # Negative
        "health_status": "CRITICAL"
    }
}

# Orchestrator creates URGENT trigger
trigger = RetrainingTrigger(
    trigger_type=TriggerType.PERFORMANCE_DRIVEN,
    model_id="xgboost_v1",
    reason="Model health CRITICAL: WR=45.0%, R=-0.05",
    priority="URGENT"
)
```

### 2. Time-Driven

**Triggers:**
- Periodic schedule (default: 7 days) → LOW

**Example:**
```python
# Model deployed 8 days ago
deployed_date = "2025-11-15T12:00:00Z"
current_date = "2025-11-23T12:00:00Z"
days_since = 8  # > 7 days threshold

# Orchestrator creates time-driven trigger
trigger = RetrainingTrigger(
    trigger_type=TriggerType.TIME_DRIVEN,
    model_id="ensemble_v2",
    reason="Periodic retrain: 8 days since last deploy",
    priority="LOW"
)
```

### 3. Regime-Driven

**Triggers:**
- Market regime changed and sustained → MEDIUM

**Example:**
```python
# Regime detector identifies change
old_regime = "RANGING"
new_regime = "TRENDING"

# Orchestrator creates regime-driven trigger
trigger = RetrainingTrigger(
    trigger_type=TriggerType.REGIME_DRIVEN,
    model_id="lstm_v1",
    reason="Regime changed: RANGING → TRENDING",
    priority="MEDIUM"
)
```

### 4. Manual Override

**Triggers:**
- Admin manually triggers retrain → URGENT/HIGH/LOW

**Example:**
```python
# Admin wants to force retrain
trigger = RetrainingTrigger(
    trigger_type=TriggerType.MANUAL,
    model_id="patchtst_v1",
    reason="Admin: Testing new features",
    priority="MEDIUM"
)
```

---

## 🔄 TRAINING PIPELINE

### Data Window Selection

| Trigger Type | Data Window |
|--------------|-------------|
| Performance-driven | 60 days (2 months) |
| Regime-driven | 30 days (1 month) |
| Time-driven | 90 days (3 months) |
| Manual | Configurable |

### Training Scripts

| Model Type | Script |
|------------|--------|
| XGBoost | `scripts/train_futures_xgboost.py` |
| LightGBM | `scripts/train_lightgbm.py` |
| N-HiTS | `scripts/train_futures_nhits.py` |
| PatchTST | `scripts/train_futures_patchtst.py` |

### Hyperparameters

Hyperparameters are inherited from current deployed version, or use defaults:

**XGBoost:**
```python
{
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1
}
```

**N-HiTS:**
```python
{
    "hidden_size": 128,
    "num_blocks": 3,
    "epochs": 50
}
```

**PatchTST:**
```python
{
    "d_model": 128,
    "nhead": 8,
    "num_layers": 3,
    "epochs": 50
}
```

---

## 📦 MODEL VERSIONING

### Version Structure

```python
ModelVersion(
    model_id="xgboost_v1",
    version_tag="v20251123_120000",
    training_date="2025-11-23T12:00:00Z",
    
    # Data
    data_range_start="2025-10-23T00:00:00Z",
    data_range_end="2025-11-23T00:00:00Z",
    sample_count=6000,
    feature_count=50,
    
    # Config
    hyperparameters={"n_estimators": 100, "max_depth": 6},
    features_used=["ohlcv", "technical", "sentiment"],
    
    # Metrics
    train_metrics={"winrate": 0.60, "avg_R": 0.18},
    validation_metrics={"winrate": 0.58, "avg_R": 0.15, "calibration": 0.78},
    
    # Deployment
    is_deployed=True,
    deployment_date="2025-11-23T12:30:00Z",
    
    # Files
    model_path="/app/ai_engine/models/xgb_model.pkl",
    scaler_path="/app/ai_engine/models/scaler.pkl",
    metadata_path="/app/ai_engine/models/metadata.json"
)
```

### Version History

Orchestrator keeps:
- **Last 10 versions** per model
- **All deployed versions** (never deleted)
- **Last 5 non-deployed versions** (older ones deleted)

### Rollback

```python
# Get all versions
versions = orchestrator.get_model_versions("xgboost_v1")
print(f"Found {len(versions)} versions")

# Rollback to specific version
orchestrator.rollback_model(
    model_id="xgboost_v1",
    target_version_tag="v20251120_120000"
)
```

---

## 🚦 DEPLOYMENT DECISIONS

### Decision Logic

```python
# Calculate overall improvement
overall_improvement = (
    winrate_improvement * 0.35 +
    avg_R_improvement * 0.35 +
    calibration_improvement * 0.30
)

# Safety checks
is_safe = (
    new_winrate >= 0.50 AND
    new_avg_R >= 0.0 AND
    new_calibration >= 0.70
)

# Make decision
if not is_safe:
    decision = KEEP_OLD
elif overall_improvement >= 0.05:  # 5%
    decision = DEPLOY_IMMEDIATELY
elif overall_improvement >= 0.02:  # 2-5%
    decision = RUN_CANARY
elif overall_improvement < 0:
    decision = KEEP_OLD
else:
    decision = REQUIRES_REVIEW
```

### Decision Types

| Decision | Condition | Action |
|----------|-----------|--------|
| **DEPLOY_IMMEDIATELY** | Improvement >= 5%, Safe | Deploy to production |
| **RUN_CANARY** | Improvement 2-5%, Safe | Run alongside old model, A/B test |
| **KEEP_OLD** | Improvement < 0%, Unsafe | Reject new version |
| **REQUIRES_REVIEW** | Improvement 0-2% | Manual review needed |

### Example Output

```python
DeploymentRecommendation(
    model_id="xgboost_v1",
    decision=DeploymentDecision.DEPLOY_IMMEDIATELY,
    performance_improvement=0.25,  # 25% improvement
    is_safe_to_deploy=True,
    requires_canary=False,
    reasons=[
        "Performance improved by 25.0%",
        "All safety checks passed"
    ],
    concerns=[],
    min_improvement_threshold=0.05
)
```

---

## 📈 RETRAINING WORKFLOW

### Daily Automated Workflow

```python
async def daily_retraining_workflow():
    """Run daily at 03:00 UTC"""
    
    # 1. Get Model Supervisor output
    signal_logs = await get_signal_logs_with_outcomes(days=30)
    supervisor_output = model_supervisor.analyze_models(signal_logs)
    
    # 2. Evaluate triggers
    current_regime = await get_current_regime()
    triggers = orchestrator.evaluate_triggers(
        supervisor_output=supervisor_output,
        current_regime=current_regime
    )
    
    if not triggers:
        logger.info("No retraining triggers found")
        return
    
    # 3. Create plan
    plan = orchestrator.create_retraining_plan(triggers, batch_size=3)
    logger.info(f"Retraining plan: {plan.total_jobs} jobs")
    
    # 4. Execute jobs
    for job in plan.jobs:
        logger.info(f"Starting retrain: {job.model_id}")
        completed_job = orchestrator.execute_retraining_job(job)
        
        if completed_job.status == RetrainingStatus.FAILED:
            await alert_admin(f"Retraining failed: {job.model_id}")
            continue
        
        # 5. Evaluate deployment
        recommendation = orchestrator.evaluate_deployment(
            new_version=completed_job.new_version
        )
        
        # 6. Handle decision
        if recommendation.decision == DeploymentDecision.DEPLOY_IMMEDIATELY:
            orchestrator.deploy_model(completed_job.new_version)
            await update_ensemble_weights(completed_job.new_version)
            logger.info(f"Deployed: {job.model_id}")
        
        elif recommendation.decision == DeploymentDecision.RUN_CANARY:
            await start_canary_test(completed_job.new_version)
            logger.info(f"Canary test started: {job.model_id}")
        
        elif recommendation.decision == DeploymentDecision.REQUIRES_REVIEW:
            await request_manual_review(recommendation)
            logger.info(f"Manual review required: {job.model_id}")
        
        else:
            logger.info(f"Kept old version: {job.model_id}")
    
    # 7. Save report
    await save_retraining_report(plan)
```

---

## 🧪 TESTING

Run standalone test:

```bash
cd backend
python services/retraining_orchestrator.py
```

Expected output:
```
============================================================
RETRAINING ORCHESTRATOR - Standalone Test
============================================================

[OK] Orchestrator initialized
  Models dir: ai_engine\models
  Min winrate: 50.0%
  Min improvement: 5.0%

============================================================
TEST 1: Evaluate Triggers
============================================================

[OK] Found 2 triggers:
  [URGENT] xgboost_v1: Model health CRITICAL: WR=45.0%, R=-0.05
  [HIGH] ensemble_v2: Model health DEGRADED: STABLE

============================================================
TEST 2: Create Retraining Plan
============================================================

[OK] Created plan: plan_20251123_001559
  Total jobs: 2
  Estimated duration: 15 minutes

  Jobs:
    - xgboost_v1: Model health CRITICAL: WR=45.0%, R=-0.05
    - ensemble_v2: Model health DEGRADED: STABLE

============================================================
TEST 3: Evaluate Deployment
============================================================

[OK] Deployment recommendation for xgboost_v1:
  Decision: deploy_immediately
  Performance improvement: 25.0%
  Safe to deploy: True
  Requires canary: False

  Reasons:
    - Performance improved by 25.0%

============================================================
[OK] All tests completed successfully!
============================================================
```

---

## 📁 FILES

| File | Purpose |
|------|---------|
| `backend/services/retraining_orchestrator.py` | Main implementation |
| `/app/data/retraining_orchestrator_state.json` | State persistence |
| `/app/data/retraining_orchestrator_test.json` | Test output |
| `RETRAINING_ORCHESTRATOR_GUIDE.md` | This guide |

---

## 💡 BEST PRACTICES

1. **Run Daily** - Check triggers every day at 03:00 UTC
2. **Prioritize URGENT** - Always handle CRITICAL model health first
3. **Monitor Deployments** - Track new version performance for 24h
4. **Keep Rollback Ready** - Never delete deployed versions
5. **Test Canary First** - For moderate improvements, run canary before full deploy
6. **Log Everything** - Maintain detailed logs of all retraining events

---

## 🚨 COMMON ISSUES

### Training Job Fails
**Cause:** Missing data, incorrect hyperparameters, script errors  
**Fix:** Check logs in `job.error_message`, validate training script exists

### New Model Underperforms
**Cause:** Insufficient data, overfitting, regime mismatch  
**Fix:** Increase data window, tune hyperparameters, check regime stability

### Deployment Rejected
**Cause:** Failed safety checks, negative improvement  
**Fix:** Review validation metrics, consider manual hyperparameter tuning

### Version Limit Reached
**Cause:** Too many versions stored (> 10)  
**Fix:** System auto-cleans old non-deployed versions

---

## 🔗 INTEGRATION WITH OTHER SYSTEMS

### Model Supervisor → Orchestrator

```python
# Model Supervisor identifies issues
supervisor_output = model_supervisor.analyze_models(signal_logs)

# Orchestrator evaluates triggers
triggers = orchestrator.evaluate_triggers(supervisor_output)
```

### Orchestrator → Event-Driven Executor

```python
# Orchestrator deploys new version
orchestrator.deploy_model(new_version)

# Executor uses new model
executor.reload_ai_agent()  # Reload ensemble with new weights
```

### Orchestrator → Portfolio Balancer

```python
# Orchestrator updates model performance
portfolio_balancer.update_model_confidence(
    model_id="xgboost_v1",
    new_winrate=0.58,
    new_avg_r=0.15
)
```

---

**Status:** ✅ READY FOR DEPLOYMENT  
**Version:** 1.0  
**Last Updated:** November 23, 2025
