# Continuous Learning Manager (CLM) Integration Guide

## Overview

The CLM system is now fully integrated into Quantum Trader with production-ready concrete implementations. This guide covers setup, testing, and deployment.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CLM Core                          │
│  - Protocol definitions                             │
│  - Orchestration logic                              │
│  - Retraining triggers                              │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│            Concrete Implementations                  │
│  1. BinanceDataClient                               │
│  2. QuantumFeatureEngineer                          │
│  3. QuantumModelTrainer                             │
│  4. QuantumModelEvaluator                           │
│  5. QuantumShadowTester                             │
│  6. SQLModelRegistry                                │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│         Quantum Trader Infrastructure               │
│  - Binance API (external_data.py)                  │
│  - Feature Engineering (feature_engineer.py)        │
│  - Model Training (train_and_save.py)               │
│  - Database (database.py)                           │
└─────────────────────────────────────────────────────┘
```

## Files Created

### Core Implementations
- **`backend/services/clm_implementations.py`** (~1100 lines)
  - All 6 protocol concrete implementations
  - Real Binance API integration
  - XGBoost/LightGBM training
  - SQLite model registry with versioning

### Integration Tools
- **`backend/services/clm_integration_demo.py`** (~350 lines)
  - Complete working example
  - 7-step demonstration workflow
  - User confirmation prompts

- **`backend/routes/clm_routes.py`** (~370 lines)
  - FastAPI REST endpoints
  - 7 API routes for CLM management
  - Background job support

### Tests
- **`tests/test_clm_implementations.py`** (~500 lines)
  - Unit tests for all implementations
  - Integration tests
  - Mock data generators

## Quick Start

### 1. Install Dependencies

```bash
pip install xgboost lightgbm scipy sqlalchemy
```

### 2. Run Demo Script

```bash
cd c:\quantum_trader
python backend/services/clm_integration_demo.py
```

This will:
1. Initialize all CLM components
2. Check current model status
3. Check retraining triggers
4. Train a new XGBoost model (with confirmation)
5. Evaluate and shadow test
6. Promote if better
7. Show updated status

### 3. Enable REST API

Add to `backend/main.py`:

```python
from backend.routes import clm_routes

# After creating FastAPI app
app.include_router(clm_routes.router)
```

### 4. Test API Endpoints

Start backend:
```bash
python backend/main.py
```

Access Swagger docs: `http://localhost:8000/docs`

Test endpoints:
- `GET /api/clm/models` - Current model status
- `POST /api/clm/retrain` - Trigger retraining
- `GET /api/clm/status` - Last retraining report

## Configuration

### CLM Settings

In demo script or when creating CLM instance:

```python
from backend.services.continuous_learning_manager import ContinuousLearningManager
from backend.services.clm_implementations import (
    BinanceDataClient,
    QuantumFeatureEngineer,
    QuantumModelTrainer,
    QuantumModelEvaluator,
    QuantumShadowTester,
    SQLModelRegistry,
)

# Initialize components
data_client = BinanceDataClient(symbol="BTCUSDT", interval="1h")
feature_engineer = QuantumFeatureEngineer(use_advanced=True)
model_trainer = QuantumModelTrainer()
model_evaluator = QuantumModelEvaluator(feature_engineer)
shadow_tester = QuantumShadowTester(data_client, feature_engineer)
model_registry = SQLModelRegistry()

# Create CLM with configuration
clm = ContinuousLearningManager(
    data_client=data_client,
    feature_engineer=feature_engineer,
    model_trainer=model_trainer,
    model_evaluator=model_evaluator,
    shadow_tester=shadow_tester,
    model_registry=model_registry,
    retrain_interval_days=7,           # Retrain every 7 days
    shadow_test_hours=24,               # Test for 24 hours
    min_improvement_threshold=0.02,     # Require 2% improvement
    training_lookback_days=90,          # Use 90 days of data
)
```

### Data Client Settings

```python
# Different trading pair
data_client = BinanceDataClient(symbol="ETHUSDT", interval="1h")

# Different timeframe
data_client = BinanceDataClient(symbol="BTCUSDT", interval="4h")
```

### Model Registry Settings

```python
# Use PostgreSQL instead of SQLite
registry = SQLModelRegistry(
    db_url="postgresql://user:pass@localhost/quantum_trader",
    model_dir=Path("ai_engine/models/clm")
)
```

## REST API Usage

### Manual Retraining

```bash
curl -X POST http://localhost:8000/api/clm/retrain \
  -H "Content-Type: application/json" \
  -d '{"models": ["xgboost", "lightgbm"], "force": true}'
```

Response:
```json
{
  "status": "started",
  "message": "Retraining job started in background"
}
```

### Check Status

```bash
curl http://localhost:8000/api/clm/status
```

Response:
```json
{
  "status": "completed",
  "report": {
    "trigger": "manual",
    "triggered_at": "2025-01-30T12:00:00Z",
    "completed_at": "2025-01-30T12:05:00Z",
    "duration_seconds": 245.3,
    "models_trained": ["xgboost", "lightgbm"],
    "promoted_models": ["xgboost"],
    "evaluations": [...],
    "shadow_tests": [...],
    "failed_models": []
  }
}
```

### Get Model Status

```bash
curl http://localhost:8000/api/clm/models
```

Response:
```json
{
  "xgboost": {
    "active_version": "v20250130_120000",
    "trained_at": "2025-01-30T12:00:00Z",
    "metrics": {
      "rmse": 0.0123,
      "mae": 0.0098,
      "directional_accuracy": 0.587
    }
  },
  "lightgbm": {...}
}
```

### Check Triggers

```bash
curl http://localhost:8000/api/clm/triggers
```

Response:
```json
{
  "xgboost": "Time-based: Last trained 8 days ago",
  "lightgbm": null,
  "nhits": "Never trained",
  "patchtst": "Never trained"
}
```

### Version History

```bash
curl http://localhost:8000/api/clm/history/xgboost?limit=5
```

Response:
```json
{
  "model_type": "xgboost",
  "versions": [
    {
      "version": "v20250130_120000",
      "status": "ACTIVE",
      "trained_at": "2025-01-30T12:00:00Z",
      "metrics": {...},
      "promoted_at": "2025-01-30T12:05:00Z"
    },
    ...
  ]
}
```

## Scheduled Retraining

### Option 1: APScheduler (Recommended)

Add to `backend/main.py`:

```python
from apscheduler.schedulers.background import BackgroundScheduler
from backend.routes.clm_routes import get_clm_instance

def scheduled_retrain():
    """Background job for scheduled retraining."""
    clm = get_clm_instance()
    report = clm.run_full_cycle(force=False)
    print(f"Scheduled retrain completed: {len(report.promoted_models)} models promoted")

# Create scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    scheduled_retrain,
    trigger="interval",
    days=1,  # Run daily
    id="clm_retrain",
    replace_existing=True
)
scheduler.start()
```

### Option 2: Cron Job

Create `scripts/clm_retrain.sh`:

```bash
#!/bin/bash
curl -X POST http://localhost:8000/api/clm/retrain \
  -H "Content-Type: application/json" \
  -d '{"force": false}'
```

Add to crontab:
```
0 2 * * * /path/to/scripts/clm_retrain.sh
```

### Option 3: Frontend Button

Add button in React frontend:

```javascript
async function triggerRetrain() {
  const response = await fetch('/api/clm/retrain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ models: ['xgboost', 'lightgbm'], force: false })
  });
  
  const data = await response.json();
  console.log('Retraining started:', data);
}
```

## Database Schema

### clm_model_versions Table

```sql
CREATE TABLE clm_model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(100) NOT NULL UNIQUE,
    status VARCHAR(20) NOT NULL,
    trained_at TIMESTAMP NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    metrics_json TEXT,
    training_samples INTEGER,
    promoted_at TIMESTAMP,
    retired_at TIMESTAMP,
    
    UNIQUE (model_type, version)
);

CREATE INDEX ix_model_type ON clm_model_versions(model_type);
CREATE INDEX ix_status ON clm_model_versions(status);
CREATE INDEX ix_version ON clm_model_versions(version);
```

### File Storage Structure

```
ai_engine/models/clm/
├── xgboost_v20250130_120000.pkl
├── xgboost_v20250129_120000.pkl
├── lightgbm_v20250130_120000.pkl
└── ...
```

## Monitoring & Alerts

### Logging

All CLM operations are logged:

```python
import logging

# Configure logging in main.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Watch logs:
```bash
tail -f logs/clm.log | grep "CLM"
```

### Key Metrics to Monitor

1. **Retraining Duration**: Should be < 10 minutes
2. **Model Performance**: Track RMSE, MAE, directional accuracy
3. **Promotion Rate**: How often models get promoted
4. **Shadow Test Results**: Candidate vs active performance
5. **Data Fetch Errors**: Binance API failures

### Alerting (Optional)

Add to scheduled retrain job:

```python
def scheduled_retrain_with_alerts():
    try:
        clm = get_clm_instance()
        report = clm.run_full_cycle(force=False)
        
        # Send success notification
        if len(report.promoted_models) > 0:
            send_slack_message(f"✅ CLM: Promoted {len(report.promoted_models)} models")
        
        # Alert on failures
        if len(report.failed_models) > 0:
            send_slack_message(f"⚠️ CLM: {len(report.failed_models)} models failed training")
    
    except Exception as e:
        send_slack_message(f"❌ CLM: Retraining job failed: {str(e)}")
        raise
```

## Troubleshooting

### Issue: Binance API Connection Fails

**Symptoms**: Empty DataFrames from data_client

**Solutions**:
1. Check internet connection
2. Verify Binance API is accessible
3. Check if symbol/interval are valid
4. Review rate limits

```python
# Test connection
data_client = BinanceDataClient()
df = data_client.load_recent_data(days=1)
print(f"Fetched {len(df)} rows")
```

### Issue: Model Training Takes Too Long

**Symptoms**: Retraining job timeout

**Solutions**:
1. Reduce training data lookback:
   ```python
   clm = ContinuousLearningManager(..., training_lookback_days=30)
   ```

2. Reduce model complexity:
   ```python
   model_trainer.train_xgboost(df, params={"n_estimators": 100, "max_depth": 5})
   ```

3. Use faster interval:
   ```python
   data_client = BinanceDataClient(interval="4h")  # Instead of 1h
   ```

### Issue: Models Not Promoting

**Symptoms**: Shadow tests pass but no promotions

**Solutions**:
1. Lower improvement threshold:
   ```python
   clm = ContinuousLearningManager(..., min_improvement_threshold=0.01)
   ```

2. Check shadow test duration (may need more data):
   ```python
   clm = ContinuousLearningManager(..., shadow_test_hours=48)
   ```

3. Review evaluation metrics:
   ```python
   curl http://localhost:8000/api/clm/status | jq '.report.shadow_tests'
   ```

### Issue: Database Lock Errors

**Symptoms**: SQLite database is locked

**Solutions**:
1. Use PostgreSQL for production:
   ```python
   registry = SQLModelRegistry(db_url="postgresql://...")
   ```

2. Reduce concurrent access
3. Check file permissions on SQLite file

### Issue: Out of Memory During Training

**Symptoms**: Process killed during model training

**Solutions**:
1. Reduce training data size
2. Use sampling:
   ```python
   df_sample = df.sample(frac=0.5, random_state=42)
   model = model_trainer.train_xgboost(df_sample)
   ```

3. Train models sequentially instead of parallel
4. Increase system RAM

## Performance Tuning

### Data Caching

Enable caching in BinanceDataClient:

```python
data_client = BinanceDataClient(
    symbol="BTCUSDT",
    interval="1h",
    cache_enabled=True,  # Cache fetched data
    cache_ttl=3600       # Cache for 1 hour
)
```

### Model Hyperparameters

Optimize for your use case:

```python
# Fast training (development)
params = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05
}

# Accurate (production)
params = {
    "n_estimators": 1000,
    "max_depth": 10,
    "learning_rate": 0.01
}
```

### Parallel Training

Train multiple models in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def train_model(model_type):
    # Train single model
    pass

with ThreadPoolExecutor(max_workers=2) as executor:
    executor.map(train_model, [ModelType.XGBOOST, ModelType.LIGHTGBM])
```

## Production Deployment Checklist

- [ ] Install all dependencies
- [ ] Run demo script successfully
- [ ] Configure database (PostgreSQL recommended)
- [ ] Set up persistent model storage
- [ ] Add CLM routes to main.py
- [ ] Test all API endpoints
- [ ] Configure logging
- [ ] Set up monitoring/alerting
- [ ] Schedule periodic retraining
- [ ] Document configuration in .env
- [ ] Run integration tests
- [ ] Deploy to production environment
- [ ] Monitor first retraining cycle
- [ ] Update README with CLM instructions

## Next Steps

1. **Add More Models**: Implement N-HiTS and PatchTST trainers
2. **Multi-Asset Support**: Extend to trade multiple pairs
3. **A/B Testing**: Deploy multiple models simultaneously
4. **Rollback**: Add model rollback functionality
5. **Explainability**: Add SHAP values for model interpretability
6. **Ensemble**: Combine multiple model predictions

## Support

- **Documentation**: `AI_CONTINUOUS_LEARNING_SUMMARY.md`
- **Core Module**: `backend/services/continuous_learning_manager.py`
- **Implementations**: `backend/services/clm_implementations.py`
- **Tests**: `tests/test_clm_implementations.py`

## License

Part of Quantum Trader - AI-powered crypto trading system
