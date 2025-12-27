# ML/AI Pipeline Deployment Checklist

## ✅ Pre-Deployment Verification

### 1. Code Verification
- [x] **11 Python modules implemented** (~8,800 lines)
  - [x] data_pipeline.py
  - [x] model_training.py
  - [x] model_registry.py
  - [x] shadow_tester.py
  - [x] rl_meta_strategy.py
  - [x] rl_position_sizing.py
  - [x] drift_detector.py
  - [x] model_supervisor.py
  - [x] retraining.py
  - [x] clm.py
  - [x] api_endpoints.py

- [x] **Database schema** (schema.sql, 400 lines)
  - [x] 6 tables defined
  - [x] 4 views created
  - [x] 2 maintenance functions
  - [x] Indexes for performance

- [x] **Integration tests** (4 suites, ~1,000 lines)
  - [x] Full retraining workflow tests
  - [x] Drift detection workflow tests
  - [x] Shadow promotion workflow tests
  - [x] End-to-end pipeline tests

- [x] **API integration**
  - [x] Routes registered in main.py (line 1980-1982)
  - [x] CLM initialization in lifespan (line 391-425)
  - [x] 15+ REST endpoints available

- [x] **Documentation**
  - [x] README.md with full usage guide
  - [x] API documentation
  - [x] Configuration guide
  - [x] Troubleshooting section

## 🚀 Deployment Steps

### Step 1: Database Setup

```bash
# Connect to PostgreSQL
psql -U postgres -d quantum_trader

# Run schema creation
\i backend/domains/learning/schema.sql

# Verify tables created
\dt

# Expected output:
# - model_registry
# - shadow_test_results
# - rl_versions
# - drift_events
# - model_performance_logs
# - retraining_jobs

# Verify views
\dv

# Expected output:
# - active_models_summary
# - shadow_models_summary
# - recent_drift_events
# - model_performance_dashboard
```

**Verification:**
```sql
SELECT COUNT(*) FROM model_registry;  -- Should return 0 (empty table)
SELECT * FROM active_models_summary;  -- Should return empty result
```

### Step 2: Python Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.\.venv\Scripts\activate  # Windows

# Install ML/AI dependencies
pip install xgboost lightgbm torch torchmetrics stable-baselines3 scikit-learn pandas numpy sqlalchemy asyncpg redis

# Verify installations
python -c "import xgboost; print('XGBoost:', xgboost.__version__)"
python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)"
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
```

**Expected versions:**
- XGBoost: 2.0+
- LightGBM: 4.0+
- PyTorch: 2.0+
- Stable-Baselines3: 2.0+

### Step 3: Environment Configuration

Add to `.env` file:

```bash
# ML/AI Pipeline Configuration
QT_CLM_ENABLED=false  # Start disabled, enable after initial training
QT_CLM_RETRAIN_HOURS=168  # Weekly retraining
QT_CLM_DRIFT_HOURS=24  # Daily drift checks
QT_CLM_PERF_HOURS=6  # 6-hourly performance monitoring
QT_CLM_DRIFT_THRESHOLD=0.05  # KS-test p-value threshold
QT_CLM_SHADOW_MIN=100  # Minimum predictions before promotion
QT_CLM_AUTO_RETRAIN=true  # Auto-retrain on drift
QT_CLM_AUTO_PROMOTE=true  # Auto-promote better shadows

# Model file storage
QT_MODEL_STORAGE_PATH=./data/models  # Local path for model files

# Training configuration
QT_TRAINING_DATA_DAYS=90  # Days of historical data for training
QT_TRAINING_VALIDATION_SPLIT=0.2  # 20% validation set
```

**Create storage directory:**
```bash
mkdir -p data/models
chmod 755 data/models
```

### Step 4: Initial Training

**Start the backend:**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Trigger initial training:**
```bash
curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "retraining_type": "full",
    "trigger_reason": "initial_deployment",
    "days_of_data": 90
  }'

# Response: {"status": "success", "job_id": "retrain_20241202_100530"}
```

**Monitor training job:**
```bash
# Get job ID from previous response
export JOB_ID="retrain_20241202_100530"

# Check status
curl http://localhost:8000/api/v1/learning/retraining/$JOB_ID

# Expected progression:
# status: PENDING → RUNNING → COMPLETED
# models_succeeded: 4 (XGBoost, LightGBM, N-HiTS, PatchTST)
```

**Training duration (estimated):**
- XGBoost: 5-10 minutes
- LightGBM: 3-7 minutes
- N-HiTS: 15-30 minutes (GPU recommended)
- PatchTST: 20-40 minutes (GPU recommended)
- **Total: 45-90 minutes** (parallel training)

### Step 5: Verify Models

```bash
# List all models
curl http://localhost:8000/api/v1/learning/models

# Expected response:
[
  {
    "model_id": "xgboost_v1.0.0_20241202",
    "model_type": "xgboost",
    "version": "1.0.0",
    "status": "shadow",
    "metrics": {
      "accuracy": 0.75,
      "f1": 0.72,
      "train_loss": 0.35
    },
    "created_at": "2024-12-02T10:15:30Z",
    "feature_count": 52
  },
  ...
]

# Verify 4 shadow models created (one per model type)
curl http://localhost:8000/api/v1/learning/models?status=shadow | jq '. | length'
# Expected: 4
```

### Step 6: Promote Shadow Models to Active

```bash
# Promote each model type
for model_type in xgboost lightgbm nhits patchtst; do
  curl -X POST http://localhost:8000/api/v1/learning/models/promote \
    -H "Content-Type: application/json" \
    -d "{\"model_type\": \"$model_type\"}"
  echo "Promoted $model_type"
done

# Verify active models
curl http://localhost:8000/api/v1/learning/models?status=active | jq '. | length'
# Expected: 4
```

### Step 7: Enable CLM Automation

Update `.env`:
```bash
QT_CLM_ENABLED=true  # Enable continuous learning
```

Restart backend:
```bash
# Stop current process (Ctrl+C)
# Start again
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Verify CLM started:**
```bash
curl http://localhost:8000/api/v1/learning/status

# Expected response:
{
  "running": true,
  "last_retraining": null,  # No scheduled retraining yet
  "last_drift_check": null,
  "last_performance_check": null,
  "config": {
    "retraining_schedule_hours": 168,
    "drift_check_hours": 24,
    "performance_check_hours": 6,
    "auto_retraining_enabled": true,
    "auto_promotion_enabled": true
  },
  "active_models": {
    "xgboost": "xgboost_v1.0.0_20241202",
    "lightgbm": "lightgbm_v1.0.0_20241202",
    "nhits": "nhits_v1.0.0_20241202",
    "patchtst": "patchtst_v1.0.0_20241202"
  },
  "shadow_models": {}
}
```

## 📊 Post-Deployment Monitoring

### Day 1: Shadow Testing Phase

**Monitor predictions:**
```bash
# Check shadow test summary
curl "http://localhost:8000/api/v1/learning/shadow-testing/summary?days=1"

# Expected: Empty (no predictions yet)
# After trading starts, you'll see:
[
  {
    "model_id": "xgboost_v1.0.0_20241202",
    "model_type": "xgboost",
    "total_predictions": 45,
    "predictions_with_outcomes": 12,
    "avg_error": 0.03,
    "first_prediction": "2024-12-02T10:30:00Z",
    "last_prediction": "2024-12-02T18:45:00Z"
  }
]
```

**Database check:**
```sql
-- Check shadow predictions
SELECT 
    model_type,
    COUNT(*) as total_predictions,
    COUNT(actual_outcome) as with_outcomes,
    AVG(confidence) as avg_confidence
FROM shadow_test_results
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model_type;
```

### Day 2-7: Drift Monitoring

**Check drift events:**
```bash
curl "http://localhost:8000/api/v1/learning/drift/events?days=7"

# Expected: Empty initially
# After 24 hours (first drift check):
[
  {
    "id": 1,
    "drift_type": "FEATURE",
    "severity": "LOW",
    "drift_score": 0.02,
    "p_value": 0.15,
    "model_type": "xgboost",
    "feature_name": "rsi_14",
    "detection_time": "2024-12-03T10:00:00Z",
    "trigger_retraining": false
  }
]
```

**Manual drift check:**
```bash
# Trigger drift check for specific model
curl -X POST http://localhost:8000/api/v1/learning/drift/check/xgboost
```

### Week 1: First Scheduled Retraining

**After 7 days (168 hours), CLM will automatically trigger retraining:**

```bash
# Check retraining jobs
curl http://localhost:8000/api/v1/learning/retraining?status=COMPLETED

# Monitor new job
curl http://localhost:8000/api/v1/learning/retraining/{latest_job_id}
```

**Verify new shadow models created:**
```sql
SELECT 
    model_type,
    version,
    status,
    created_at,
    metrics
FROM model_registry
WHERE created_at > NOW() - INTERVAL '1 day'
ORDER BY created_at DESC;
```

### Ongoing: Performance Tracking

**Check model performance:**
```bash
# Get performance for active XGBoost model
export MODEL_ID=$(curl -s http://localhost:8000/api/v1/learning/models?model_type=xgboost\&status=active | jq -r '.[0].model_id')

curl "http://localhost:8000/api/v1/learning/performance/$MODEL_ID?days=30"

# Expected response:
{
  "model_id": "xgboost_v1.0.0_20241202",
  "history": [
    {
      "period_start": "2024-12-01T00:00:00Z",
      "period_end": "2024-12-02T00:00:00Z",
      "n_trades": 25,
      "winrate": 0.64,
      "winrate_ci_lower": 0.52,
      "winrate_ci_upper": 0.76,
      "sharpe_ratio": 1.85,
      "profit_factor": 1.92
    }
  ]
}
```

**Dashboard views:**
```sql
-- Active models summary
SELECT * FROM active_models_summary;

-- Recent drift events (last 7 days)
SELECT * FROM recent_drift_events;

-- Performance dashboard (last 30 days)
SELECT * FROM model_performance_dashboard;
```

## 🧪 Running Tests

**Unit tests:**
```bash
pytest backend/domains/learning/ -v
```

**Integration tests:**
```bash
# All integration tests
pytest tests/integration/ -v

# Specific test suite
pytest tests/integration/test_full_retraining_workflow.py -v
pytest tests/integration/test_drift_detection_workflow.py -v
pytest tests/integration/test_shadow_promotion_workflow.py -v
pytest tests/integration/test_end_to_end_pipeline.py -v
```

**Expected results:**
- All tests should pass
- Coverage > 80%
- No critical warnings

## 🛠️ Maintenance Tasks

### Weekly

**Check retraining jobs:**
```sql
SELECT 
    job_id,
    status,
    retraining_type,
    models_succeeded,
    models_failed,
    created_at,
    completed_at
FROM retraining_jobs
WHERE created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

**Review drift events:**
```sql
SELECT 
    drift_type,
    severity,
    COUNT(*) as count,
    AVG(drift_score) as avg_score
FROM drift_events
WHERE detection_time > NOW() - INTERVAL '7 days'
GROUP BY drift_type, severity
ORDER BY drift_type, severity;
```

### Monthly

**Run database cleanup:**
```sql
-- Clean old shadow test results (90+ days)
SELECT cleanup_old_shadow_results();

-- Archive retired models (180+ days)
SELECT archive_retired_models();

-- Vacuum tables
VACUUM ANALYZE model_registry;
VACUUM ANALYZE shadow_test_results;
VACUUM ANALYZE retraining_jobs;
```

**Review model performance:**
```bash
# Export performance data
curl "http://localhost:8000/api/v1/learning/models?status=active" | jq '.[] | {model_type, version, metrics}' > monthly_performance.json
```

### Quarterly

**Full system review:**
1. Review all active models
2. Analyze retraining frequency
3. Check drift detection accuracy
4. Evaluate shadow promotion decisions
5. Update configuration thresholds
6. Plan model architecture updates

## 🚨 Troubleshooting

### Issue: CLM Not Starting

**Symptom:**
```bash
curl http://localhost:8000/api/v1/learning/status
# Error: 503 Service Unavailable
```

**Solution:**
1. Check environment variables:
   ```bash
   env | grep QT_CLM
   ```
2. Check logs:
   ```bash
   grep -i "CLM" logs/backend.log
   ```
3. Verify dependencies:
   ```bash
   curl http://localhost:8000/api/v2/health
   # Check: redis, postgres, event_bus
   ```

### Issue: Training Job Failed

**Symptom:**
```bash
curl http://localhost:8000/api/v1/learning/retraining/{job_id}
# status: "FAILED"
```

**Solution:**
1. Check error message:
   ```sql
   SELECT error_message FROM retraining_jobs WHERE job_id = '{job_id}';
   ```
2. Common causes:
   - Insufficient historical data
   - GPU memory issues (for deep learning models)
   - Invalid training configuration
3. Retry with adjusted config:
   ```bash
   curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
     -d '{"retraining_type": "partial", "model_types": ["xgboost", "lightgbm"], "days_of_data": 60}'
   ```

### Issue: No Drift Detected

**Symptom:**
Drift events table is empty after several days

**Solution:**
1. Lower threshold:
   ```bash
   # Update .env
   QT_CLM_DRIFT_THRESHOLD=0.10  # Less sensitive
   ```
2. Manual drift check:
   ```bash
   curl -X POST http://localhost:8000/api/v1/learning/drift/check/xgboost
   ```
3. Verify data quality:
   ```sql
   SELECT COUNT(*) FROM ohlcv WHERE timestamp > NOW() - INTERVAL '7 days';
   ```

### Issue: Shadow Model Not Promoting

**Symptom:**
Shadow models with better metrics not promoted

**Solution:**
1. Check prediction count:
   ```bash
   curl http://localhost:8000/api/v1/learning/shadow-testing/summary
   # total_predictions must be >= 100
   ```
2. Lower minimum:
   ```bash
   # Update .env
   QT_CLM_SHADOW_MIN=50
   ```
3. Force promotion:
   ```bash
   curl -X POST http://localhost:8000/api/v1/learning/models/promote \
     -d '{"model_type": "xgboost"}'
   ```

## ✅ Deployment Validation Checklist

- [ ] **Database schema deployed**
  - [ ] 6 tables created
  - [ ] 4 views created
  - [ ] 2 functions created
  
- [ ] **Dependencies installed**
  - [ ] XGBoost, LightGBM
  - [ ] PyTorch, TorchMetrics
  - [ ] Stable-Baselines3
  - [ ] Redis, AsyncPG
  
- [ ] **Environment configured**
  - [ ] QT_CLM_* variables set
  - [ ] Model storage path created
  - [ ] Redis connection working
  
- [ ] **Initial training completed**
  - [ ] Job status: COMPLETED
  - [ ] 4 models trained (all types)
  - [ ] Models promoted to active
  
- [ ] **CLM started**
  - [ ] Running: true
  - [ ] Active models: 4
  - [ ] Auto-retraining enabled
  
- [ ] **API endpoints working**
  - [ ] /api/v1/learning/status
  - [ ] /api/v1/learning/models
  - [ ] /api/v1/learning/retraining/trigger
  - [ ] /api/v1/learning/drift/events
  
- [ ] **Tests passing**
  - [ ] Integration tests: 100%
  - [ ] Unit tests: 100%
  
- [ ] **Monitoring configured**
  - [ ] Dashboard views accessible
  - [ ] Logs configured
  - [ ] Alerts set up

## 📈 Success Metrics

**Week 1:**
- ✅ 4 active models deployed
- ✅ Shadow testing active (100+ predictions)
- ✅ First drift check completed
- ✅ No critical errors

**Month 1:**
- ✅ First scheduled retraining completed
- ✅ At least 1 shadow promotion occurred
- ✅ Drift events being tracked
- ✅ Performance improving or stable

**Quarter 1:**
- ✅ Multiple retraining cycles completed
- ✅ Multiple shadow promotions
- ✅ Clear drift detection patterns
- ✅ Measurable performance improvements

---

**Deployment Status:** ✅ Ready for Production  
**Last Updated:** December 2, 2024  
**Version:** 1.0.0
