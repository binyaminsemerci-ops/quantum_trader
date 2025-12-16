# Continuous Learning Manager (CLM)

## Overview

The **Continuous Learning Manager** is Quantum Trader's adaptive intelligence layer that ensures all ML models remain effective over time. It automatically detects when retraining is needed, trains new model versions, evaluates them rigorously, runs shadow tests in live conditions, and promotes superior candidates to production.

**Key Benefit:** Prevents model drift and keeps predictions aligned with evolving market dynamics.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Continuous Learning Manager                  │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Trigger System │→ │ Training Engine │→ │  Evaluator   │ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
│          ↓                    ↓                     ↓        │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Shadow Tester  │→ │ Promotion Logic │→ │   Registry   │ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Responsibilities

### 1. **Trigger Detection**
Determines when retraining is needed based on:
- **Time-based**: Model age exceeds threshold (e.g., 7 days)
- **Data volume**: Sufficient new data available (e.g., 10,000 new bars)
- **Performance decay**: Recent accuracy drops below baseline
- **Regime shift**: Market conditions change significantly
- **Manual**: Explicit user/admin request

### 2. **Training Pipeline**
Trains new versions of all ML models:
- XGBoost (gradient boosting trees)
- LightGBM (fast gradient boosting)
- N-HiTS (PyTorch time-series neural network)
- PatchTST (transformer for time-series forecasting)

Each model is trained on:
- Historical data (90 days by default)
- Freshly engineered features
- Consistent validation splits

### 3. **Evaluation**
Validates candidate models on held-out data:

**Regression Metrics:**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Error standard deviation

**Classification Metrics:**
- Directional accuracy (% of correct direction predictions)
- Hit rate (adjusted for class imbalance)
- Correlation with target

**Comparison:**
- Delta vs. active model
- Regime-segmented performance
- Statistical significance tests

### 4. **Shadow Testing**
Runs candidates in **live mode** parallel to active models:
- Collects real-time predictions
- Compares error distributions (KS test)
- Tracks directional accuracy over time
- Evaluates stability across regimes

**Shadow Duration:** 24 hours by default (configurable)

### 5. **Promotion Logic**
Upgrades candidates to production if:
- Evaluation shows improvement > 2% (configurable)
- Shadow test recommends promotion
- No critical issues detected

**Rollback Safety:** Old models marked as RETIRED (not deleted)

---

## Data Structures

### ModelArtifact
```python
@dataclass
class ModelArtifact:
    model_type: ModelType
    version: str                    # e.g., "v20250130_143052"
    trained_at: datetime
    metrics: dict[str, float]       # rmse, mae, directional_accuracy, etc.
    model_object: Any               # serialized model weights
    status: ModelStatus             # CANDIDATE, SHADOW, ACTIVE, RETIRED
    training_range: tuple[datetime, datetime]
    feature_config: dict            # feature engineering settings
    training_params: dict           # hyperparameters
    data_points: int                # training dataset size
```

### EvaluationResult
```python
@dataclass
class EvaluationResult:
    model_type: ModelType
    version: str
    
    # Metrics
    rmse: float
    mae: float
    error_std: float
    directional_accuracy: float
    hit_rate: float
    
    # Comparison
    vs_active_rmse_delta: float     # negative = better
    vs_active_direction_delta: float # positive = better
    
    # Statistical
    correlation_with_target: float
    prediction_bias: float
    regime_accuracy: dict[str, float]
    
    evaluated_at: datetime
```

### ShadowTestResult
```python
@dataclass
class ShadowTestResult:
    model_type: ModelType
    candidate_version: str
    active_version: str
    
    # Live performance
    live_predictions: int
    candidate_mae: float
    active_mae: float
    candidate_direction_acc: float
    active_direction_acc: float
    
    # Distribution comparison
    error_ks_statistic: float       # Kolmogorov-Smirnov test
    error_mean_diff: float
    error_std_diff: float
    
    # Recommendation
    recommend_promotion: bool
    reason: str
    
    tested_from: datetime
    tested_hours: float
```

---

## Usage

### Basic Initialization

```python
from backend.services.continuous_learning_manager import (
    ContinuousLearningManager,
    ModelType,
)

clm = ContinuousLearningManager(
    data_client=your_data_client,
    feature_engineer=your_feature_engineer,
    trainer=your_trainer,
    evaluator=your_evaluator,
    shadow_tester=your_shadow_tester,
    registry=your_model_registry,
    retrain_interval_days=7,
    shadow_test_hours=24,
    min_improvement_threshold=0.02,  # 2%
    training_lookback_days=90,
)
```

### Check Retraining Triggers

```python
triggers = clm.check_if_retrain_needed()

for model_type, trigger in triggers.items():
    if trigger:
        print(f"{model_type.value} needs retraining: {trigger.value}")
```

### Run Full Retraining Cycle

```python
report = clm.run_full_cycle(force=True)

print(f"Promoted: {len(report.promoted_models)}")
print(f"Failed: {len(report.failed_models)}")
print(f"Duration: {report.total_duration_seconds:.1f}s")

for model_type in report.promoted_models:
    print(f"  ✅ {model_type.value} promoted")
```

### Manual Model Training

```python
# Train specific models only
artifacts = clm.retrain_all(models=[ModelType.XGBOOST, ModelType.NHITS])

# Evaluate
results = clm.evaluate_models(artifacts)

# Shadow test
shadows = clm.run_shadow_tests(artifacts)

# Promote
promoted = clm.promote_if_better(artifacts, results, shadows)
```

### Get Model Status

```python
status = clm.get_model_status_summary()

for model_type, info in status.items():
    print(f"{model_type}: {info['active_version']}")
    print(f"  Trained: {info['active_trained_at']}")
    print(f"  Metrics: {info['active_metrics']}")
```

---

## Interfaces (Dependency Injection)

CLM depends on these protocols:

### DataClient
```python
class DataClient(Protocol):
    def load_training_data(self, start: datetime, end: datetime) -> pd.DataFrame: ...
    def load_recent_data(self, days: int) -> pd.DataFrame: ...
    def load_validation_data(self, days: int) -> pd.DataFrame: ...
```

### FeatureEngineer
```python
class FeatureEngineer(Protocol):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_names(self) -> list[str]: ...
```

### ModelTrainer
```python
class ModelTrainer(Protocol):
    def train_xgboost(self, df: pd.DataFrame, params: dict) -> Any: ...
    def train_lightgbm(self, df: pd.DataFrame, params: dict) -> Any: ...
    def train_nhits(self, df: pd.DataFrame, params: dict) -> Any: ...
    def train_patchtst(self, df: pd.DataFrame, params: dict) -> Any: ...
```

### ModelEvaluator
```python
class ModelEvaluator(Protocol):
    def evaluate(
        self, 
        model: Any, 
        df: pd.DataFrame,
        model_type: ModelType
    ) -> EvaluationResult: ...
    
    def compare_to_active(
        self,
        candidate_result: EvaluationResult,
        active_result: EvaluationResult
    ) -> EvaluationResult: ...
```

### ShadowTester
```python
class ShadowTester(Protocol):
    def run_shadow_test(
        self,
        model_type: ModelType,
        candidate_model: Any,
        active_model: Any,
        hours: int = 24
    ) -> ShadowTestResult: ...
```

### ModelRegistry
```python
class ModelRegistry(Protocol):
    def get_active(self, model_type: ModelType) -> ModelArtifact | None: ...
    def save_model(self, artifact: ModelArtifact) -> None: ...
    def promote(self, model_type: ModelType, new_version: str) -> None: ...
    def retire(self, model_type: ModelType, version: str) -> None: ...
    def get_model_history(
        self, 
        model_type: ModelType, 
        limit: int = 10
    ) -> list[ModelArtifact]: ...
```

---

## Integration with Quantum Trader

### 1. Scheduled Retraining (Background Task)

```python
import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

def retrain_job():
    report = clm.run_full_cycle()
    logger.info(f"CLM cycle complete: {report.summary()}")

# Run every 7 days at 2 AM
scheduler.add_job(
    retrain_job,
    'cron',
    day='*/7',
    hour=2,
)

scheduler.start()
```

### 2. Performance-Triggered Retraining

```python
# In position monitor or performance tracker
if current_model_accuracy < threshold:
    logger.warning("Model accuracy below threshold, triggering retraining")
    report = clm.run_full_cycle(force=True)
```

### 3. FastAPI Endpoints

```python
from fastapi import APIRouter

router = APIRouter(prefix="/api/clm")

@router.post("/retrain")
async def trigger_retraining(models: list[str] = None):
    """Manually trigger retraining"""
    model_types = [ModelType(m) for m in models] if models else None
    report = clm.run_full_cycle(models=model_types, force=True)
    return {
        "promoted": [m.value for m in report.promoted_models],
        "failed": [m.value for m in report.failed_models],
        "duration": report.total_duration_seconds,
    }

@router.get("/status")
async def get_clm_status():
    """Get current model status"""
    return clm.get_model_status_summary()

@router.get("/triggers")
async def check_triggers():
    """Check retraining triggers"""
    triggers = clm.check_if_retrain_needed()
    return {
        model_type.value: trigger.value if trigger else None
        for model_type, trigger in triggers.items()
    }
```

---

## Configuration

### Environment Variables

```bash
# Retraining
CLM_RETRAIN_INTERVAL_DAYS=7
CLM_TRAINING_LOOKBACK_DAYS=90

# Shadow Testing
CLM_SHADOW_TEST_HOURS=24
CLM_MIN_IMPROVEMENT_THRESHOLD=0.02

# Data Volume Trigger
CLM_DATA_THRESHOLD_POINTS=10000

# Performance Decay Trigger
CLM_PERFORMANCE_DECAY_THRESHOLD=0.05
```

### Config File (config/clm_config.yaml)

```yaml
continuous_learning:
  retrain_interval_days: 7
  shadow_test_hours: 24
  min_improvement_threshold: 0.02
  training_lookback_days: 90
  
  triggers:
    time_threshold_days: 7
    data_threshold_points: 10000
    performance_decay_threshold: 0.05
  
  models:
    - xgboost
    - lightgbm
    - nhits
    - patchtst
  
  training_params:
    xgboost:
      n_estimators: 500
      max_depth: 7
      learning_rate: 0.01
      subsample: 0.8
    
    lightgbm:
      n_estimators: 500
      max_depth: 7
      learning_rate: 0.01
      subsample: 0.8
    
    nhits:
      input_size: 120
      h: 24
      n_blocks: 3
    
    patchtst:
      input_size: 120
      h: 24
      patch_len: 16
```

---

## Performance Considerations

### Training Parallelization
```python
# Train models in parallel
from concurrent.futures import ThreadPoolExecutor

def train_parallel(clm, models):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(clm._train_single_model, model_type, df): model_type
            for model_type in models
        }
        
        artifacts = {}
        for future in futures:
            model_type = futures[future]
            artifacts[model_type] = future.result()
        
        return artifacts
```

### Shadow Test Resource Management
```python
# Limit concurrent shadow tests
MAX_CONCURRENT_SHADOWS = 2

shadow_queue = []
active_shadows = []

for model_type, artifact in artifacts.items():
    if len(active_shadows) >= MAX_CONCURRENT_SHADOWS:
        # Wait for one to complete
        wait_for_shadow_completion()
    
    shadow = start_shadow_test(artifact)
    active_shadows.append(shadow)
```

---

## Monitoring & Alerts

### Metrics to Track
- **Retraining frequency**: How often cycles run
- **Promotion rate**: % of candidates promoted
- **Training duration**: Time per model type
- **Shadow test duration**: Time to completion
- **Evaluation metrics**: RMSE, MAE, directional accuracy trends

### Alerts
```python
# Prometheus metrics example
from prometheus_client import Counter, Histogram, Gauge

clm_retrains_total = Counter('clm_retrains_total', 'Total retraining cycles')
clm_promotions_total = Counter('clm_promotions_total', 'Total model promotions')
clm_failures_total = Counter('clm_failures_total', 'Total training failures')
clm_training_duration = Histogram('clm_training_duration_seconds', 'Training duration')
clm_active_model_age_days = Gauge('clm_active_model_age_days', 'Active model age')

# In CLM
clm_retrains_total.inc()
clm_training_duration.observe(report.total_duration_seconds)
clm_promotions_total.inc(len(report.promoted_models))
```

---

## Testing

### Run Unit Tests
```bash
python -m pytest tests/test_continuous_learning_manager.py -v
```

### Run Example
```bash
python backend/services/continuous_learning_manager_example.py
```

**Expected Output:**
```
================================================================================
CONTINUOUS LEARNING MANAGER - DEMO
================================================================================

CLM Initialized ✓

Current Model Status:
--------------------------------------------------------------------------------
  xgboost      | v20250101_000000     | Trained: 2025-01-20
  lightgbm     | v20250101_000000     | Trained: 2025-01-20
  nhits        | v20250101_000000     | Trained: 2025-01-20
  patchtst     | v20250101_000000     | Trained: 2025-01-20

Running Full Retraining Cycle:
================================================================================
...
RETRAINING REPORT
================================================================================
CLM Report [2025-01-30 14:30]
Trigger: manual
Trained: 4 models
Promoted: 2 models
Failed: 2 models
Duration: 45.3s
```

---

## FAQ

### Q: How long does a full cycle take?
**A:** Typically 10-60 minutes depending on:
- Training data size (90 days default)
- Model complexity (N-HiTS/PatchTST slower than XGBoost)
- Shadow test duration (24 hours by default, but can be reduced)

### Q: Can I retrain specific models only?
**A:** Yes:
```python
report = clm.run_full_cycle(models=[ModelType.XGBOOST], force=True)
```

### Q: What happens if training fails?
**A:** Failed models are logged but don't block the cycle. The old active model remains in production.

### Q: Can I skip shadow testing?
**A:** Yes, set `shadow_test_hours=0` or modify `promote_if_better()` to rely only on validation evaluation.

### Q: How do I rollback a promoted model?
**A:** Manually promote the previous version:
```python
old_version = "v20250120_120000"
clm.registry.promote(ModelType.XGBOOST, old_version)
```

---

## Roadmap

### Planned Features
- [ ] **Multi-stage shadow testing** (canary → 10% → 50% → 100%)
- [ ] **A/B testing framework** (split traffic between models)
- [ ] **Automatic rollback** (if promoted model underperforms)
- [ ] **Ensemble model training** (meta-learners on top of base models)
- [ ] **Transfer learning** (warm-start from previous model)
- [ ] **Distributed training** (Dask/Ray for large datasets)
- [ ] **AutoML integration** (hyperparameter tuning with Optuna)
- [ ] **Explainability reports** (SHAP values, feature importance)

---

## License

MIT License - Part of Quantum Trader AI Trading System
