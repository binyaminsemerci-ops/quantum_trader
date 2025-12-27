# ML/AI Pipeline Quick Reference

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Deploy database schema
psql -U postgres -d quantum_trader -f backend/domains/learning/schema.sql

# 2. Enable CLM in .env
echo "QT_CLM_ENABLED=false" >> .env  # Keep disabled until initial training

# 3. Start backend
python -m uvicorn backend.main:app --port 8000

# 4. Run initial training
curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{"retraining_type": "full", "trigger_reason": "initial_setup"}'

# 5. Wait 45-90 minutes, then promote models
for model_type in xgboost lightgbm nhits patchtst; do
  curl -X POST http://localhost:8000/api/v1/learning/models/promote \
    -d "{\"model_type\": \"$model_type\"}"
done

# 6. Enable CLM
# Update .env: QT_CLM_ENABLED=true
# Restart backend
```

## 📡 API Endpoints

### Status & Health
```bash
GET  /api/v1/learning/status        # CLM system status
GET  /api/v1/learning/health        # Health check
```

### Model Management
```bash
GET  /api/v1/learning/models                    # List all models
GET  /api/v1/learning/models/{model_id}         # Get model details
POST /api/v1/learning/models/promote            # Promote shadow to active
POST /api/v1/learning/models/{model_id}/retire  # Retire model
```

### Retraining
```bash
POST /api/v1/learning/retraining/trigger       # Trigger retraining
GET  /api/v1/learning/retraining/{job_id}      # Get job status
```

### Drift Monitoring
```bash
GET  /api/v1/learning/drift/events               # List drift events
POST /api/v1/learning/drift/check/{model_type}  # Manual drift check
```

### Shadow Testing
```bash
GET /api/v1/learning/shadow-testing/summary  # Shadow test results
```

### Performance
```bash
GET /api/v1/learning/performance/{model_id}  # Model performance history
```

## 🔧 Common Commands

### Trigger Full Retraining
```bash
curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
  -H "Content-Type: application/json" \
  -d '{
    "retraining_type": "full",
    "trigger_reason": "manual",
    "days_of_data": 90
  }'
```

### Check Job Status
```bash
JOB_ID="retrain_20241202_100530"
curl http://localhost:8000/api/v1/learning/retraining/$JOB_ID | jq
```

### List Active Models
```bash
curl http://localhost:8000/api/v1/learning/models?status=active | jq
```

### Promote Shadow Model
```bash
curl -X POST http://localhost:8000/api/v1/learning/models/promote \
  -H "Content-Type: application/json" \
  -d '{"model_type": "xgboost"}'
```

### Check Drift Events
```bash
curl "http://localhost:8000/api/v1/learning/drift/events?days=7" | jq
```

### Get CLM Status
```bash
curl http://localhost:8000/api/v1/learning/status | jq
```

## 🗄️ Database Queries

### Active Models
```sql
SELECT * FROM active_models_summary;
```

### Recent Drift
```sql
SELECT * FROM recent_drift_events;
```

### Performance Dashboard
```sql
SELECT * FROM model_performance_dashboard;
```

### Retraining Jobs
```sql
SELECT 
    job_id,
    status,
    models_succeeded,
    models_failed,
    created_at,
    completed_at
FROM retraining_jobs
ORDER BY created_at DESC
LIMIT 10;
```

### Shadow Test Results
```sql
SELECT 
    model_type,
    COUNT(*) as total_predictions,
    COUNT(actual_outcome) as with_outcomes,
    AVG(confidence) as avg_confidence
FROM shadow_test_results
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY model_type;
```

## ⚙️ Environment Variables

```bash
# Enable/Disable
QT_CLM_ENABLED=true

# Scheduling
QT_CLM_RETRAIN_HOURS=168    # Weekly retraining
QT_CLM_DRIFT_HOURS=24       # Daily drift checks
QT_CLM_PERF_HOURS=6         # 6-hourly performance checks

# Thresholds
QT_CLM_DRIFT_THRESHOLD=0.05    # KS-test p-value
QT_CLM_SHADOW_MIN=100          # Min predictions for promotion

# Automation
QT_CLM_AUTO_RETRAIN=true    # Auto-retrain on drift
QT_CLM_AUTO_PROMOTE=true    # Auto-promote shadows

# Storage
QT_MODEL_STORAGE_PATH=./data/models

# Training
QT_TRAINING_DATA_DAYS=90
QT_TRAINING_VALIDATION_SPLIT=0.2
```

## 🔥 Quick Fixes

### CLM Not Starting
```bash
# Check logs
grep -i "CLM" logs/backend.log

# Verify dependencies
curl http://localhost:8000/api/v2/health | jq '.dependencies'
```

### Training Failed
```sql
-- Check error
SELECT error_message FROM retraining_jobs WHERE status = 'FAILED' ORDER BY created_at DESC LIMIT 1;

-- Retry partial
curl -X POST http://localhost:8000/api/v1/learning/retraining/trigger \
  -d '{"retraining_type": "partial", "model_types": ["xgboost", "lightgbm"]}'
```

### Shadow Not Promoting
```bash
# Check prediction count
curl http://localhost:8000/api/v1/learning/shadow-testing/summary | jq '.[].total_predictions'

# Force promotion
curl -X POST http://localhost:8000/api/v1/learning/models/promote -d '{"model_type": "xgboost"}'
```

## 📊 Monitoring Dashboard

```bash
# Create monitoring script
cat > monitor_clm.sh << 'EOF'
#!/bin/bash
while true; do
  clear
  echo "=== CLM Status ==="
  curl -s http://localhost:8000/api/v1/learning/status | jq '{running, active_models, last_retraining}'
  
  echo -e "\n=== Active Models ==="
  curl -s http://localhost:8000/api/v1/learning/models?status=active | jq '.[] | {model_type, version, metrics}'
  
  echo -e "\n=== Recent Drift ==="
  curl -s "http://localhost:8000/api/v1/learning/drift/events?days=1" | jq '.[] | {drift_type, severity, model_type}'
  
  echo -e "\n=== Shadow Testing ==="
  curl -s http://localhost:8000/api/v1/learning/shadow-testing/summary | jq '.[] | {model_type, total_predictions, predictions_with_outcomes}'
  
  sleep 60
done
EOF

chmod +x monitor_clm.sh
./monitor_clm.sh
```

## 🎯 Performance Targets

| Metric | Target | Warning | Critical |
|--------|--------|---------|----------|
| Retraining Success Rate | >95% | <90% | <80% |
| Active Model Winrate | >60% | <55% | <50% |
| Drift Detection Rate | 1-3/week | 0/week or >5/week | >10/week |
| Shadow Promotion Rate | 1-2/month | 0/month | >5/month |
| Training Duration | <90 min | <120 min | >150 min |
| API Response Time | <200ms | <500ms | >1000ms |

## 📝 Logging Locations

```bash
# CLM logs
tail -f logs/backend.log | grep -i "CLM"

# Retraining logs
tail -f logs/backend.log | grep -i "retrain"

# Drift detection logs
tail -f logs/backend.log | grep -i "drift"

# Model promotion logs
tail -f logs/backend.log | grep -i "promote"
```

## 🧪 Testing

```bash
# Full integration test suite
pytest tests/integration/ -v

# Specific test
pytest tests/integration/test_full_retraining_workflow.py::test_full_retraining_workflow -v

# With coverage
pytest tests/integration/ --cov=backend.domains.learning --cov-report=html
```

## 📞 Support Contacts

- **Documentation**: `backend/domains/learning/README.md`
- **Deployment Guide**: `DEPLOYMENT_CHECKLIST_ML_PIPELINE.md`
- **Architecture**: `ML_PIPELINE_ARCHITECTURE.md`
- **Issues**: Create GitHub issue with `[ML-PIPELINE]` tag

---

**Quick Reference Version:** 1.0.0  
**Last Updated:** December 2, 2024  
**Total API Endpoints:** 15+  
**Total Code:** ~9,300 lines
