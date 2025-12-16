# ML/AI Pipeline - Complete Implementation

## Overview

Complete production-ready continuous learning system for Quantum Trader with:
- **Automated retraining** (FULL/PARTIAL/INCREMENTAL)
- **Drift detection** (feature/prediction/performance)
- **Shadow testing** with auto-promotion
- **RL agent lifecycle management**
- **Model registry** with versioning
- **Performance monitoring** and alerting

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ContinuousLearningManager (CLM)                 │
│                     Central Orchestrator                         │
└────────────┬────────────────────────────────────────────────────┘
             │
             ├──► HistoricalDataFetcher (50+ indicators)
             ├──► FeatureEngineer (technical indicators)
             ├──► ModelTraining (XGBoost, LightGBM, N-HiTS, PatchTST)
             ├──► ModelRegistry (versioning, metadata)
             ├──► ShadowTester (parallel evaluation)
             ├──► DriftDetector (KS-test, PSI)
             ├──► ModelSupervisor (winrate, calibration)
             ├──► RetrainingOrchestrator (workflow engine)
             ├──► RLMetaStrategy (PPO strategy selection)
             └──► RLPositionSizing (SAC position/leverage/TPSL)
```

## Components

### Core Modules (11 files, ~9,300 lines)

1. **data_pipeline.py** (650 lines)
   - `HistoricalDataFetcher`: Fetch OHLCV from database
   - `FeatureEngineer`: 50+ technical indicators

2. **model_training.py** (1,200 lines)
   - `train_xgboost()`: Tree-based gradient boosting
   - `train_lightgbm()`: Fast gradient boosting
   - `train_nhits()`: Neural Hierarchical Interpolation (PyTorch)
   - `train_patchtst()`: Patch Time-Series Transformer (PyTorch)

3. **model_registry.py** (600 lines)
   - `ModelRegistry`: Version control for models
   - Status: TRAINING → SHADOW → ACTIVE → RETIRED
   - Postgres metadata + file storage (pickle)

4. **shadow_tester.py** (700 lines)
   - `ShadowTester`: Parallel predictions
   - Track outcomes, compare performance
   - Auto-promote better models

5. **rl_meta_strategy.py** (600 lines)
   - `RLMetaStrategy`: PPO-based strategy selector
   - 4 strategies: momentum, mean-reversion, trend-following, breakout
   - Regime-aware allocation

6. **rl_position_sizing.py** (700 lines)
   - `RLPositionSizing`: SAC-based sizing agent
   - Outputs: size, leverage, TP/SL
   - Risk-adjusted position management

7. **drift_detector.py** (500 lines)
   - `DriftDetector`: KS-test for distribution shifts
   - 3 types: FEATURE, PREDICTION, PERFORMANCE
   - Severity: LOW, MEDIUM, HIGH, CRITICAL

8. **model_supervisor.py** (500 lines)
   - `ModelSupervisor`: Performance tracking
   - Winrate with confidence intervals
   - Calibration error, directional bias

9. **retraining.py** (900 lines)
   - `RetrainingOrchestrator`: Workflow automation
   - FULL: Retrain all models from scratch
   - PARTIAL: Retrain subset of models
   - INCREMENTAL: Fine-tune on new data

10. **clm.py** (1,000 lines)
    - `ContinuousLearningManager`: Central coordinator
    - Event-driven architecture
    - Scheduled tasks:
      * Retraining: Every 168 hours (weekly)
      * Drift checks: Every 24 hours (daily)
      * Performance monitoring: Every 6 hours
      * Shadow promotions: Every 1 hour

11. **api_endpoints.py** (550 lines)
    - REST API for external control
    - 15+ endpoints for management
    - FastAPI integration

### Database Schema (schema.sql, 400 lines)

**6 Tables:**
1. `model_registry`: Model versions and metadata
2. `shadow_test_results`: Shadow predictions and outcomes
3. `rl_versions`: RL agent checkpoints
4. `drift_events`: Distribution drift logs
5. `model_performance_logs`: Time-series metrics
6. `retraining_jobs`: Workflow status

**4 Views:**
- `active_models_summary`
- `shadow_models_summary`
- `recent_drift_events`
- `model_performance_dashboard`

**2 Maintenance Functions:**
- `cleanup_old_shadow_results()` (90-day retention)
- `archive_retired_models()` (180-day archival)

### Integration Tests (4 suites, ~1,000 lines)

1. **test_full_retraining_workflow.py**
   - Full/partial/incremental retraining
   - Failure handling
   - Concurrent job prevention
   - Data validation

2. **test_drift_detection_workflow.py**
   - Feature/prediction/performance drift
   - Auto-retraining triggers
   - Severity thresholds
   - Scheduled checks

3. **test_shadow_promotion_workflow.py**
   - Shadow evaluation
   - Auto-promotion logic
   - Minimum prediction requirements
   - Multi-shadow best selection

4. **test_end_to_end_pipeline.py**
   - Complete lifecycle test
   - Resilience to failures
   - Multi-model coordination

## Installation

### 1. Database Setup

```sql
-- Run schema.sql
psql -U postgres -d quantum_trader -f backend/domains/learning/schema.sql
```

### 2. Python Dependencies

```bash
pip install xgboost lightgbm torch torchmetrics stable-baselines3 scikit-learn pandas numpy sqlalchemy asyncpg
```

### 3. Initialize CLM

```python
from backend.domains.learning.clm import create_clm, CLMConfig
from backend.core.database import get_db_session
from backend.core.event_bus import get_event_bus
from backend.core.policy_store import get_policy_store

# Configure
config = CLMConfig(
    retraining_schedule_hours=168,  # Weekly
    drift_check_hours=24,  # Daily
    performance_check_hours=6,
    drift_trigger_threshold=0.05,
    shadow_min_predictions=100,
    auto_retraining_enabled=True,
    auto_promotion_enabled=True,
)

# Create instance
async def setup():
    db = await get_db_session()
    event_bus = get_event_bus()
    policy_store = get_policy_store()
    
    clm = await create_clm(db, event_bus, policy_store, config)
    await clm.start()
    
    return clm
```

### 4. Add to FastAPI

```python
# main.py
from backend.domains.learning.api_endpoints import router as learning_router, initialize_clm

app = FastAPI()

@app.on_event("startup")
async def startup():
    # Initialize CLM
    db = await get_db_session()
    event_bus = get_event_bus()
    policy_store = get_policy_store()
    
    await initialize_clm(db, event_bus, policy_store)

# Register routes
app.include_router(learning_router)
```

## Usage

### API Endpoints

**Model Management:**
```bash
# List all models
GET /api/v1/learning/models?model_type=xgboost&status=active

# Get model details
GET /api/v1/learning/models/{model_id}

# Retire model
POST /api/v1/learning/models/{model_id}/retire

# Promote shadow model
POST /api/v1/learning/models/promote
```

**Retraining:**
```bash
# Trigger full retraining
POST /api/v1/learning/retraining/trigger
{
  "retraining_type": "full",
  "trigger_reason": "manual",
  "days_of_data": 90
}

# Get job status
GET /api/v1/learning/retraining/{job_id}
```

**Shadow Testing:**
```bash
# Get shadow summary
GET /api/v1/learning/shadow-testing/summary?days=30
```

**Drift Monitoring:**
```bash
# Get drift events
GET /api/v1/learning/drift/events?days=7&drift_type=FEATURE

# Trigger drift check
POST /api/v1/learning/drift/check/xgboost
```

**CLM Status:**
```bash
# Get system status
GET /api/v1/learning/status

# Start/stop CLM
POST /api/v1/learning/start
POST /api/v1/learning/stop
```

**Performance:**
```bash
# Get model performance
GET /api/v1/learning/performance/{model_id}?days=30
```

### Python API

```python
# Trigger retraining
job_id = await clm.trigger_retraining(
    retraining_type=RetrainingType.FULL,
    trigger_reason="manual",
)

# Check drift
result = await clm.manual_trigger_drift_check(ModelType.XGBOOST)

# Promote shadow
success = await clm.manual_promote_shadow(ModelType.LIGHTGBM)

# Get status
status = await clm.get_system_status()
```

## Event-Driven Architecture

### Published Events

```python
# Drift detected
await event_bus.publish("learning.drift.detected", {
    "drift_type": "FEATURE",
    "severity": "HIGH",
    "model_type": "xgboost",
    "feature_name": "rsi_14",
})

# Retraining completed
await event_bus.publish("learning.retraining.completed", {
    "job_id": "retrain_123",
    "models_trained": 4,
    "models_succeeded": 4,
})

# Performance alert
await event_bus.publish("learning.performance.alert", {
    "model_id": "xgboost_v2.0",
    "metric": "winrate",
    "current_value": 0.45,
    "threshold": 0.60,
})

# Model promoted
await event_bus.publish("learning.model.updated", {
    "model_type": "xgboost",
    "old_model_id": "xgboost_v1.0",
    "new_model_id": "xgboost_v2.0",
    "reason": "shadow_promotion",
})
```

### Event Subscriptions

CLM subscribes to:
- `learning.drift.detected` → Trigger retraining
- `learning.retraining.completed` → Log completion
- `learning.performance.alert` → Alert monitoring
- `learning.model.promoted` → Notify trading engine

## Workflows

### 1. Initial Setup

```bash
# 1. Deploy database schema
psql -f schema.sql

# 2. Run initial training
curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{"retraining_type": "full", "trigger_reason": "initial_setup"}'

# 3. Monitor job
curl http://localhost:8000/api/v1/learning/retraining/{job_id}

# 4. Promote shadow models to active
curl -X POST http://localhost:8000/api/v1/learning/models/promote \
  -d '{"model_type": "xgboost"}'
```

### 2. Automated Operations

Once active, CLM runs automatically:

- **Weekly retraining** (every 168 hours)
- **Daily drift checks** (every 24 hours)
- **6-hourly performance monitoring**
- **Hourly shadow promotion evaluation**

### 3. Manual Intervention

```python
# Force retraining
job_id = await clm.trigger_retraining(
    retraining_type=RetrainingType.PARTIAL,
    model_types=[ModelType.XGBOOST, ModelType.LIGHTGBM],
    trigger_reason="performance_degradation",
)

# Check specific drift
result = await clm.manual_trigger_drift_check(ModelType.NHITS)

# Force promotion
await clm.manual_promote_shadow(ModelType.PATCHTST)
```

## Monitoring

### 1. Dashboard Views

```sql
-- Active models summary
SELECT * FROM active_models_summary;

-- Shadow testing progress
SELECT * FROM shadow_models_summary;

-- Recent drift events
SELECT * FROM recent_drift_events;

-- Performance trends
SELECT * FROM model_performance_dashboard;
```

### 2. API Status

```bash
curl http://localhost:8000/api/v1/learning/status
```

Returns:
```json
{
  "running": true,
  "last_retraining": "2024-01-15T10:30:00Z",
  "last_drift_check": "2024-01-20T08:00:00Z",
  "last_performance_check": "2024-01-20T14:00:00Z",
  "config": {...},
  "active_models": {
    "xgboost": "xgboost_v2.1.0",
    "lightgbm": "lightgbm_v2.0.5"
  },
  "shadow_models": {
    "nhits": "nhits_v3.0.0",
    "patchtst": "patchtst_v1.5.0"
  }
}
```

### 3. Logs

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("learning.clm")

# CLM logs all operations:
# - Scheduled task runs
# - Drift detections
# - Retraining triggers
# - Promotions
# - Errors
```

## Configuration

### CLMConfig Options

```python
CLMConfig(
    # Scheduling
    retraining_schedule_hours=168,  # Weekly
    drift_check_hours=24,  # Daily
    performance_check_hours=6,  # 4x daily
    
    # Thresholds
    drift_trigger_threshold=0.05,  # KS p-value
    shadow_min_predictions=100,  # Min for promotion
    
    # Automation
    auto_retraining_enabled=True,  # Auto-retrain on drift
    auto_promotion_enabled=True,  # Auto-promote shadows
)
```

### PolicyStore Integration

Store config in PolicyStore:

```python
await policy_store.set_policy(
    "learning.clm.config",
    {
        "retraining_schedule_hours": 168,
        "drift_check_hours": 24,
        "auto_retraining_enabled": True,
        "auto_promotion_enabled": True,
    }
)
```

## Testing

### Run Integration Tests

```bash
# All tests
pytest tests/integration/

# Specific suite
pytest tests/integration/test_full_retraining_workflow.py
pytest tests/integration/test_drift_detection_workflow.py
pytest tests/integration/test_shadow_promotion_workflow.py
pytest tests/integration/test_end_to_end_pipeline.py
```

### Test Coverage

- ✅ Full/partial/incremental retraining
- ✅ Drift detection (feature/prediction/performance)
- ✅ Shadow testing and auto-promotion
- ✅ RL agent training and checkpointing
- ✅ Error handling and recovery
- ✅ Multi-model coordination
- ✅ End-to-end pipeline

## Performance

### Training Times (estimated)

- **XGBoost**: ~5-10 minutes (90 days data)
- **LightGBM**: ~3-7 minutes
- **N-HiTS**: ~15-30 minutes (GPU recommended)
- **PatchTST**: ~20-40 minutes (GPU recommended)

**Full retraining**: ~1 hour (all 4 models in parallel)

### Database Storage

- **Model files**: ~10-50 MB per model (pickle)
- **Shadow predictions**: ~1 GB per month (retention: 90 days)
- **Drift events**: ~10 MB per month
- **Performance logs**: ~50 MB per month

## Maintenance

### Scheduled Cleanup

```sql
-- Run weekly
SELECT cleanup_old_shadow_results();  -- 90-day retention
SELECT archive_retired_models();  -- 180-day archival
```

### Manual Cleanup

```sql
-- Delete old retraining jobs
DELETE FROM retraining_jobs
WHERE completed_at < NOW() - INTERVAL '180 days';

-- Retire old models
UPDATE model_registry
SET status = 'RETIRED', retired_at = NOW()
WHERE status = 'ACTIVE'
  AND promoted_at < NOW() - INTERVAL '90 days';
```

## Troubleshooting

### CLM Not Starting

```python
# Check logs
logger.error("CLM startup failed")

# Verify database connection
await db.execute("SELECT 1")

# Verify EventBus connection
await event_bus.publish("test.event", {})
```

### Retraining Failures

```bash
# Check job status
curl http://localhost:8000/api/v1/learning/retraining/{job_id}

# Review error logs
SELECT * FROM retraining_jobs WHERE status = 'FAILED';
```

### Drift Not Detected

```python
# Lower threshold
config.drift_trigger_threshold = 0.1  # Less sensitive

# Check data quality
data = await data_fetcher.fetch_recent_data(days=7)
assert len(data) > 1000  # Sufficient data
```

### Shadow Not Promoting

```python
# Check prediction count
summary = await shadow_tester.get_shadow_test_summary()
assert summary[0]["total_predictions"] >= 100

# Lower minimum
config.shadow_min_predictions = 50

# Force promotion
await clm.manual_promote_shadow(ModelType.XGBOOST)
```

## Roadmap

### Future Enhancements

- [ ] Multi-GPU training support
- [ ] Distributed retraining (Ray, Dask)
- [ ] A/B testing framework
- [ ] Model explainability (SHAP, LIME)
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model compression (quantization, pruning)
- [ ] Real-time feature drift detection
- [ ] Automated feature engineering
- [ ] Ensemble model support
- [ ] Cloud deployment (AWS, Azure, GCP)

## License

Proprietary - Quantum Trader

## Support

For issues or questions:
- Email: support@quantum-trader.com
- Docs: https://docs.quantum-trader.com/ml-pipeline
- GitHub: https://github.com/quantum-trader/ml-pipeline

---

**Status**: ✅ Production Ready  
**Version**: 1.0.0  
**Last Updated**: 2024-01-20  
**Total Code**: ~9,300 lines across 15 files
