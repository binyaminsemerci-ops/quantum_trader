# ML/AI Pipeline Architecture - Continuous Learning System

**System**: Quantum Trader  
**Component**: Continuous Learning & Retraining Pipeline  
**Version**: 1.0  
**Date**: December 2, 2025

---

## 🎯 Executive Summary

This document describes a **production-ready, fully automated ML/AI pipeline** for continuous learning, model retraining, shadow testing, and RL agent maintenance in Quantum Trader.

**Key Capabilities:**
- **Automated retraining** for 4 supervised models (XGBoost, LightGBM, N-HiTS, PatchTST)
- **RL agent maintenance** for Meta Strategy Controller + Position Sizing Agent
- **Drift detection** with statistical tests (KS-test, PSI)
- **Shadow testing** with automatic promotion based on performance
- **Model registry** with versioning, metrics tracking, status management
- **Event-driven** integration with EventBus for real-time coordination
- **Policy-driven** configuration via PolicyStore for all training parameters

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CONTINUOUS LEARNING MANAGER (CLM)               │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐              │
│  │   Scheduler  │   │ Drift Watch  │   │  Performance │              │
│  │  (Time/Event)│   │  (KS-test)   │   │   Monitor    │              │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘              │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             │                                           │
│                     ┌───────▼────────┐                                 │
│                     │   RETRAINING   │                                 │
│                     │  ORCHESTRATOR  │                                 │
│                     └───────┬────────┘                                 │
│                             │                                           │
└─────────────────────────────┼───────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐    ┌───────▼────────┐    ┌──────▼────────┐
│  DATA PIPELINE│    │ MODEL TRAINING │    │  RL PIPELINE  │
│               │    │                │    │               │
│ ·DataFetcher  │───▶│ ·XGBoost       │    │ ·Meta Strat   │
│ ·Features     │    │ ·LightGBM      │    │ ·Pos Sizing   │
│ ·Labels       │    │ ·N-HiTS        │    │ ·Q-tables     │
│               │    │ ·PatchTST      │    │ ·Versioning   │
└───────────────┘    └───────┬────────┘    └──────┬────────┘
                             │                     │
                     ┌───────▼─────────────────────▼────────┐
                     │        MODEL REGISTRY                │
                     │                                       │
                     │  ·Versions  ·Metrics  ·Status        │
                     │  ·Active/Shadow/Retired              │
                     │  ·Postgres + File Storage            │
                     └───────┬───────────────────────────────┘
                             │
                     ┌───────▼────────┐
                     │ SHADOW TESTER  │
                     │                │
                     │ ·Parallel eval │
                     │ ·PnL tracking  │
                     │ ·Auto-promote  │
                     └───────┬────────┘
                             │
                     ┌───────▼────────┐
                     │  AI TRADING    │
                     │    ENGINE      │
                     │  (Production)  │
                     └────────────────┘
```

---

## 📊 Data Flow

### 1. **Normal Operation** (No Retraining)
```
Market Data → AI Trading Engine → Predictions → Risk → Execution
                      │
                      ├─→ EventBus: "trade_closed" → Model Supervisor
                      │
                      └─→ Drift Detector (continuous monitoring)
```

### 2. **Retraining Trigger** (Time/Drift/Performance)
```
Trigger Event → CLM → Retraining Orchestrator
                           │
                           ├─→ Data Pipeline (fetch historical data)
                           │        │
                           │        └─→ Feature Engineer (indicators + labels)
                           │
                           ├─→ Model Training (XGB/LGBM/NHITS/PATCHTST)
                           │        │
                           │        └─→ Evaluation (val metrics)
                           │
                           ├─→ Model Registry (save as SHADOW)
                           │
                           └─→ Shadow Tester (parallel evaluation)
                                    │
                                    └─→ Promotion (if better) → ACTIVE
```

### 3. **RL Maintenance** (Periodic/Regime Change)
```
Trade Outcomes → RL Agents (Q-table updates)
                      │
                      ├─→ Versioned snapshots (every N trades)
                      │
                      └─→ Regime change → Q-table reset/re-init
```

---

## 🧩 Module Details

### 1. `data_pipeline.py`

**Purpose:** Fetch historical market data and engineer features/labels.

**Classes:**
- `HistoricalDataFetcher`: Fetch OHLCV data from Binance + database
- `FeatureEngineer`: Compute technical indicators (RSI, MACD, BB, ATR, etc.)

**Key Functions:**
```python
async def fetch_historical_data(
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "5m"
) -> pd.DataFrame

def engineer_features(
    df: pd.DataFrame,
    config: FeatureConfig
) -> pd.DataFrame  # Adds 50+ technical indicators

def generate_labels(
    df: pd.DataFrame,
    label_type: str = "future_return",  # or "direction", "regime"
    horizon: int = 12  # 12 candles ahead
) -> pd.DataFrame
```

**Integration:**
- Called by `ModelTraining` during retraining
- Configurable via `PolicyStore` (lookback period, timeframes, indicators)
- Publishes `data_pipeline.fetch_completed` event via EventBus

---

### 2. `model_training.py`

**Purpose:** Train supervised ML models with proper train/val/test splits.

**Functions:**
```python
async def train_xgboost(
    data: pd.DataFrame,
    config: XGBConfig
) -> Tuple[ModelArtifact, EvaluationResult]

async def train_lightgbm(...)
async def train_nhits(...)  # Time series transformer
async def train_patchtst(...)  # Patch-based transformer

# Evaluation
def evaluate_model(
    model: Any,
    test_data: pd.DataFrame
) -> EvaluationResult:
    """
    Returns:
        - RMSE, MAE, R² (regression)
        - Accuracy, F1, Precision, Recall (classification)
        - Directional accuracy (% correct direction predictions)
        - Sharpe ratio (simulated PnL)
        - Calibration error
    """
```

**Integration:**
- Called by `RetrainingOrchestrator`
- Hyperparameters from `PolicyStore.training_config`
- Publishes `model.training.started`, `model.training.completed` events
- Saves artifacts to `ModelRegistry`

---

### 3. `model_registry.py`

**Purpose:** Centralized storage for all model versions with metadata.

**Database Schema:**
```sql
CREATE TABLE model_registry (
    model_id VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,  -- xgb, lgbm, nhits, patchtst, rl_meta, rl_sizing
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- TRAINING, SHADOW, ACTIVE, RETIRED
    metrics JSONB,  -- {rmse: 0.02, accuracy: 0.78, sharpe: 1.5, ...}
    training_config JSONB,
    training_data_range JSONB,  -- {start: "2024-01-01", end: "2024-12-01"}
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP,
    retired_at TIMESTAMP,
    file_path TEXT,  -- /models/xgb_v7.pkl
    notes TEXT
);

CREATE INDEX idx_model_type_status ON model_registry(model_type, status);
CREATE INDEX idx_created_at ON model_registry(created_at DESC);
```

**API:**
```python
async def register_model(model_info: ModelArtifact) -> str  # Returns model_id
async def get_active_model(model_type: str) -> ModelArtifact
async def get_shadow_model(model_type: str) -> ModelArtifact
async def promote_shadow_to_active(model_type: str) -> None
async def retire_model(model_id: str) -> None
async def list_models(
    model_type: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50
) -> List[ModelArtifact]
```

**Integration:**
- Used by all training/testing modules
- Publishes `model.registered`, `model.promoted`, `model.retired` events

---

### 4. `shadow_tester.py`

**Purpose:** Run new models in parallel with active models to compare performance.

**Process:**
1. Receive `signal_generated` event from AI Trading Engine
2. Generate predictions from **both** ACTIVE and SHADOW models
3. Store predictions in `shadow_test_results` table
4. Track:
   - Hit rate (% correct predictions)
   - Average confidence
   - PnL if trades were executed
   - Sharpe ratio over test period
5. After N samples (e.g., 1000 trades), evaluate:
   - If SHADOW > ACTIVE on key metrics → trigger promotion
   - If SHADOW < ACTIVE → mark as REJECTED

**Database Schema:**
```sql
CREATE TABLE shadow_test_results (
    id SERIAL PRIMARY KEY,
    shadow_model_id VARCHAR(255) NOT NULL,
    active_model_id VARCHAR(255) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    active_prediction JSONB,  -- {action: "BUY", confidence: 0.82, ...}
    shadow_prediction JSONB,
    actual_outcome JSONB,  -- {pnl_pct: 0.03, duration_hours: 2.5, ...}
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_shadow_model_timestamp ON shadow_test_results(shadow_model_id, timestamp);
```

**API:**
```python
async def record_shadow_prediction(
    shadow_model_id: str,
    active_model_id: str,
    symbol: str,
    active_pred: dict,
    shadow_pred: dict
) -> None

async def record_outcome(
    test_id: int,
    outcome: dict  # {pnl_pct, duration_hours, ...}
) -> None

async def evaluate_shadow_model(
    shadow_model_id: str,
    min_samples: int = 1000
) -> ShadowEvaluationResult:
    """
    Returns:
        - total_samples
        - active_metrics: {hit_rate, avg_pnl, sharpe, ...}
        - shadow_metrics: {hit_rate, avg_pnl, sharpe, ...}
        - recommendation: "PROMOTE" | "REJECT" | "CONTINUE_TESTING"
    """

async def promote_if_ready(shadow_model_id: str) -> bool:
    """Auto-promote if shadow is better than active"""
```

**Integration:**
- Subscribes to `ai.signal.generated` events
- Subscribes to `execution.trade.closed` events for outcomes
- Publishes `shadow.promotion.recommended`, `shadow.promoted` events

---

### 5. `rl_meta_strategy.py`

**Purpose:** Refactored Meta Strategy Controller with versioned Q-tables.

**State Space:**
```python
@dataclass
class MetaStrategyState:
    market_regime: str  # TRENDING, RANGING, CHOPPY, VOLATILE
    volatility_level: str  # LOW, MEDIUM, HIGH, EXTREME
    liquidity_level: str  # LOW, MEDIUM, HIGH
    recent_performance: str  # GOOD (>5% last 7d), NEUTRAL, POOR (<-3%)
```

**Action Space:**
```python
class StrategyType(Enum):
    TREND_FOLLOW = "trend_follow"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    RANGE_BOUND = "range_bound"
    DEFENSIVE = "defensive"  # Low risk, high confidence only
```

**Reward Function:**
```python
def calculate_reward(
    pnl_pct: float,
    time_in_trade_hours: float,
    max_drawdown_pct: float,
    winrate: float
) -> float:
    """
    Reward = PnL% + time_bonus - drawdown_penalty + winrate_bonus
    
    Example:
        +3% PnL in 2 hours with 1% max DD, 70% winrate
        → reward = 3.0 + (24/2)*0.1 - 1.0*2 + (0.7-0.5)*10
        → reward = 3.0 + 1.2 - 2.0 + 2.0 = 4.2
    """
    time_bonus = (24 / max(time_in_trade_hours, 0.1)) * 0.1
    drawdown_penalty = abs(max_drawdown_pct) * 2.0
    winrate_bonus = (winrate - 0.5) * 10.0
    
    return pnl_pct + time_bonus - drawdown_penalty + winrate_bonus
```

**Versioned Storage:**
```python
# Q-table saved as:
# data/rl_meta_strategy_v{version}.json
{
    "version": 7,
    "created_at": "2025-12-02T10:00:00",
    "q_table": {
        "TRENDING_HIGH_MED_GOOD": {
            "trend_follow": 4.5,
            "mean_reversion": -1.2,
            ...
        },
        ...
    },
    "metadata": {
        "total_updates": 12450,
        "epsilon": 0.05,
        "learning_rate": 0.1
    }
}
```

**API:**
```python
async def select_strategy(state: MetaStrategyState) -> StrategyType
async def update_q_table(
    state: MetaStrategyState,
    action: StrategyType,
    reward: float,
    next_state: MetaStrategyState
) -> None
async def save_version(version: int) -> None
async def load_version(version: int) -> None
async def reset_on_regime_shift() -> None
```

**Integration:**
- Loads config from `PolicyStore` (epsilon, learning_rate)
- Publishes `rl.meta.strategy_selected`, `rl.meta.updated` events
- Triggered by `execution.trade.closed` events for reward updates

---

### 6. `rl_position_sizing.py`

**Purpose:** Refactored RL Position Sizing Agent with enhanced state/action spaces.

**State Space:**
```python
@dataclass
class SizingState:
    market_regime: str  # From RegimeDetector
    signal_confidence: float  # 0.0-1.0
    portfolio_exposure: float  # 0.0-1.0 (% of capital in positions)
    recent_winrate: float  # Last 20 trades
    volatility: float  # ATR-based
```

**Action Space:**
```python
@dataclass
class SizingAction:
    size_multiplier: float  # [0.3, 0.5, 0.7, 1.0, 1.5]
    leverage: int  # [1, 2, 3, 5, 7, 10]
    tp_strategy: str  # "conservative", "balanced", "aggressive"
    sl_strategy: str  # "tight", "medium", "wide"
```

**Reward Engineering:**
```python
def calculate_reward(
    pnl_pct: float,
    time_hours: float,
    max_dd_pct: float,
    hit_tp: bool,
    hit_sl: bool
) -> float:
    """
    Comprehensive reward function:
    - PnL% (main component)
    - Time penalty (longer = worse)
    - Drawdown penalty
    - TP bonus (reward hitting take profit)
    - SL penalty (punish stop loss)
    """
    reward = pnl_pct
    
    # Time penalty: -0.05 per hour
    reward -= time_hours * 0.05
    
    # Drawdown penalty
    reward -= abs(max_dd_pct) * 3.0
    
    # TP/SL bonuses
    if hit_tp:
        reward += 1.0  # Bonus for clean exit
    if hit_sl:
        reward -= 0.5  # Penalty for stop out
    
    return reward
```

**Safety Integration:**
```python
async def get_sizing_decision(
    state: SizingState,
    policy_store: PolicyStore
) -> SizingDecision:
    """
    1. RL agent selects action (size, leverage, TP/SL)
    2. Safety checks via PolicyStore:
       - Enforce max_risk_pct_per_trade
       - Enforce max_leverage
       - Enforce max_positions
    3. Return final decision (possibly clamped)
    """
    rl_action = self._q_learning_select(state)
    
    # Safety clamp
    policy = await policy_store.get_policy()
    mode = policy.active_mode
    
    final_size = min(
        rl_action.size_usd,
        policy.modes[mode].max_risk_pct_per_trade * balance
    )
    final_leverage = min(
        rl_action.leverage,
        policy.modes[mode].max_leverage
    )
    
    return SizingDecision(
        size_usd=final_size,
        leverage=final_leverage,
        ...
    )
```

**Integration:**
- Called by AI Trading Engine for every signal
- Updates Q-table on `execution.trade.closed` events
- Versioned snapshots every 500 trades
- Publishes `rl.sizing.decision`, `rl.sizing.updated` events

---

### 7. `clm.py` (Continuous Learning Manager)

**Purpose:** Central orchestrator for all learning activities.

**Responsibilities:**
1. **Scheduling:**
   - Time-based: Every 7 days
   - Event-based: On drift detection, performance drop
   - Manual: Via API trigger

2. **Supervised Retraining:**
   - Coordinate data fetch → training → evaluation → shadow test
   - Track retraining history
   - Monitor shadow test progress

3. **RL Maintenance:**
   - Periodic Q-table snapshots
   - Prune old states
   - Reset on regime shift

4. **EventBus Integration:**
   - Subscribe to:
     - `trade.closed`
     - `drift.detected`
     - `performance.alert`
     - `regime.changed`
   - Publish:
     - `clm.retraining.started`
     - `clm.retraining.completed`
     - `clm.model.promoted`
     - `clm.rl.reset`

**API:**
```python
class ContinuousLearningManager:
    async def start_scheduler(self) -> None
    async def stop_scheduler(self) -> None
    
    async def trigger_full_retraining(
        self,
        reason: str = "manual"
    ) -> str:  # Returns job_id
    
    async def trigger_partial_retraining(
        self,
        model_types: List[str]
    ) -> str:
    
    async def handle_drift_event(self, event: dict) -> None
    async def handle_performance_alert(self, event: dict) -> None
    async def handle_regime_change(self, event: dict) -> None
    
    async def get_status(self) -> CLMStatus:
        """
        Returns:
            - current_jobs: [...]
            - last_retraining: datetime
            - active_models: {xgb: v7, lgbm: v5, ...}
            - shadow_models: {xgb: v8_testing, ...}
            - rl_versions: {meta: v12, sizing: v34}
        """
    
    async def get_retraining_history(
        self,
        limit: int = 20
    ) -> List[RetrainingJob]
```

**Configuration (via PolicyStore):**
```json
{
  "clm_config": {
    "enabled": true,
    "schedule_interval_days": 7,
    "auto_promote_threshold": {
      "min_samples": 1000,
      "min_improvement_pct": 5.0,
      "max_drawdown_pct": 3.0
    },
    "drift_detection": {
      "enabled": true,
      "check_interval_hours": 6,
      "ks_test_threshold": 0.05
    },
    "rl_maintenance": {
      "snapshot_interval_trades": 500,
      "reset_on_regime_change": true,
      "prune_old_states_threshold": 10000
    }
  }
}
```

---

### 8. `drift_detector.py`

**Purpose:** Monitor for concept drift in features, predictions, and outcomes.

**Detection Methods:**

1. **Feature Distribution Drift (KS-test)**
```python
from scipy.stats import ks_2samp

def detect_feature_drift(
    historical: pd.DataFrame,
    recent: pd.DataFrame,
    feature: str,
    threshold: float = 0.05
) -> Tuple[bool, float]:
    """
    Kolmogorov-Smirnov test for distribution shift.
    
    Returns:
        (is_drifting, p_value)
    """
    stat, p_value = ks_2samp(
        historical[feature].values,
        recent[feature].values
    )
    return p_value < threshold, p_value
```

2. **Prediction Distribution Drift**
```python
# Compare distribution of predictions (e.g., confidence scores)
# over last 30 days vs previous 30 days
```

3. **Performance Drift**
```python
# Track rolling winrate, avg PnL
# Alert if drops below threshold
```

**Database Schema:**
```sql
CREATE TABLE drift_events (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMP NOT NULL,
    drift_type VARCHAR(50) NOT NULL,  -- feature, prediction, performance
    feature_name VARCHAR(100),
    p_value FLOAT,
    threshold FLOAT,
    severity VARCHAR(20),  -- low, medium, high, critical
    metadata JSONB,
    action_taken VARCHAR(50),  -- retraining_triggered, alert_sent, none
    created_at TIMESTAMP DEFAULT NOW()
);
```

**API:**
```python
async def check_feature_drift(
    features: List[str],
    lookback_days: int = 60
) -> List[DriftResult]

async def check_prediction_drift(
    model_type: str,
    lookback_days: int = 30
) -> DriftResult

async def check_performance_drift(
    lookback_days: int = 7
) -> DriftResult
```

**Integration:**
- Runs on schedule (every 6 hours)
- Publishes `drift.detected` event if threshold exceeded
- CLM subscribes and triggers retraining

---

### 9. `model_supervisor.py`

**Purpose:** Monitor model health and performance in production.

**Monitored Metrics:**

1. **Winrate:** % of profitable trades
2. **Calibration:** Are 80% confidence predictions actually right 80% of the time?
3. **Bias:** Long/short imbalance
4. **Latency:** Model inference time
5. **Coverage:** % of symbols with confident predictions

**Database Schema:**
```sql
CREATE TABLE model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    winrate FLOAT,
    avg_confidence FLOAT,
    calibration_error FLOAT,
    long_short_ratio FLOAT,
    avg_latency_ms FLOAT,
    total_predictions INT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_timestamp ON model_performance_logs(model_id, timestamp DESC);
```

**API:**
```python
async def log_prediction(
    model_id: str,
    prediction: dict,
    latency_ms: float
) -> None

async def log_outcome(
    prediction_id: int,
    outcome: dict  # {pnl_pct, hit_tp, hit_sl, ...}
) -> None

async def get_model_health(
    model_id: str,
    lookback_hours: int = 24
) -> ModelHealth:
    """
    Returns:
        - winrate
        - calibration_error
        - long_short_ratio
        - avg_latency_ms
        - recommendation: "HEALTHY" | "DEGRADED" | "CRITICAL"
    """

async def check_all_models(self) -> List[ModelHealth]:
    """Periodic check of all active models"""
```

**Alerts:**
```python
# Trigger alerts if:
- Winrate < 45% (threshold from policy)
- Calibration error > 0.15
- Latency > 500ms
- Long/short ratio > 3.0 or < 0.33 (severe bias)
```

**Integration:**
- Logs every prediction from AI Trading Engine
- Publishes `model.performance.degraded` event if thresholds exceeded
- CLM subscribes and may trigger retraining

---

### 10. `retraining.py` (Retraining Orchestrator)

**Purpose:** Workflow engine that executes the full retraining pipeline.

**Main Workflows:**

#### A. Full Supervised Retraining
```python
async def run_full_supervised_retraining(
    symbols: List[str],
    lookback_days: int = 180,
    config: RetrainingConfig
) -> RetrainingResult:
    """
    Steps:
    1. Fetch historical data (DataPipeline)
    2. Engineer features (DataPipeline)
    3. Train all models (XGB, LGBM, NHITS, PATCHTST)
    4. Evaluate on validation set
    5. Register as SHADOW in ModelRegistry
    6. Start shadow testing
    7. Return job summary
    """
    job_id = generate_job_id()
    
    try:
        # Step 1: Data
        logger.info(f"[{job_id}] Fetching historical data...")
        data = await data_pipeline.fetch_historical_data(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=lookback_days),
            end_date=datetime.now(),
            timeframe="5m"
        )
        
        # Step 2: Features
        logger.info(f"[{job_id}] Engineering features...")
        features = data_pipeline.engineer_features(data, config.feature_config)
        labels = data_pipeline.generate_labels(features, horizon=12)
        
        # Step 3-5: Train each model
        results = {}
        for model_type in ["xgboost", "lightgbm", "nhits", "patchtst"]:
            logger.info(f"[{job_id}] Training {model_type}...")
            
            model_artifact, eval_result = await model_training.train_model(
                model_type=model_type,
                data=labels,
                config=config.training_configs[model_type]
            )
            
            # Register as SHADOW
            model_id = await model_registry.register_model(model_artifact)
            logger.info(f"[{job_id}] Registered {model_type} as {model_id} (SHADOW)")
            
            results[model_type] = {
                "model_id": model_id,
                "metrics": eval_result.metrics
            }
        
        # Step 6: Start shadow testing
        for model_id in results.values():
            await shadow_tester.start_testing(model_id)
        
        return RetrainingResult(
            job_id=job_id,
            status="completed",
            models_trained=results,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"[{job_id}] Retraining failed: {e}", exc_info=True)
        return RetrainingResult(
            job_id=job_id,
            status="failed",
            error=str(e),
            timestamp=datetime.now()
        )
```

#### B. Partial Retraining (Single Model)
```python
async def run_partial_retraining(
    model_type: str,
    reason: str
) -> RetrainingResult:
    """Retrain only one model type"""
    # Similar to full, but only for specified model_type
```

#### C. RL Maintenance
```python
async def run_rl_maintenance() -> None:
    """
    Periodic RL maintenance:
    1. Save Q-table snapshots
    2. Prune very old states
    3. Update version numbers
    """
    # Meta Strategy
    await rl_meta_strategy.save_version(
        version=get_next_version("meta")
    )
    
    # Position Sizing
    await rl_position_sizing.save_version(
        version=get_next_version("sizing")
    )
    
    logger.info("RL maintenance completed")
```

#### D. Drift Response
```python
async def handle_drift_event(event: dict) -> None:
    """
    Triggered by drift detector.
    
    Action depends on severity:
    - LOW: Log alert, continue monitoring
    - MEDIUM: Trigger partial retraining for affected models
    - HIGH: Trigger full retraining
    - CRITICAL: Stop trading, trigger emergency retraining
    """
    severity = event["severity"]
    
    if severity == "LOW":
        logger.warning(f"Low drift detected: {event}")
        return
    
    if severity == "MEDIUM":
        logger.warning(f"Medium drift detected, retraining affected models...")
        await run_partial_retraining(
            model_type=event.get("model_type", "xgboost"),
            reason="drift_detected"
        )
    
    if severity in ["HIGH", "CRITICAL"]:
        logger.error(f"{severity} drift detected! Full retraining...")
        await run_full_supervised_retraining(
            symbols=get_active_symbols(),
            reason=f"drift_{severity.lower()}"
        )
```

#### E. Performance Response
```python
async def handle_performance_drop(event: dict) -> None:
    """
    Triggered by model supervisor when winrate/metrics drop.
    """
    model_id = event["model_id"]
    model_type = event["model_type"]
    
    logger.warning(f"Performance drop for {model_id}, retraining {model_type}...")
    
    await run_partial_retraining(
        model_type=model_type,
        reason="performance_degradation"
    )
```

**Integration:**
- Called by CLM
- Uses all other modules (data, training, registry, shadow)
- Publishes detailed progress events via EventBus

---

## 🔄 Complete Scenario Examples

### Scenario 1: Drift Detected → Retraining → Promotion

**Timeline:**

```
T+0h:   Drift Detector runs scheduled check
        → Detects HIGH drift in RSI feature distribution (p_value=0.02)
        → Publishes "drift.detected" event

T+1m:   CLM receives event
        → Logs alert in dashboard
        → Calls RetrainingOrchestrator.handle_drift_event()

T+2m:   Retraining Orchestrator starts
        → Job ID: RTN-20251202-001
        → Publishes "retraining.started" event
        → Fetches 180 days of data for 50 symbols

T+15m:  Data fetched, features engineered
        → 1.2M data points, 64 features
        → Labels generated (12-step ahead return)

T+30m:  XGBoost training starts
        → 80/10/10 train/val/test split
        → 5-fold CV
        → Hyperparams from PolicyStore

T+45m:  XGBoost trained
        → Val RMSE: 0.018 (vs 0.022 for current)
        → Directional accuracy: 64% (vs 58%)
        → Registered as xgb_v8 (SHADOW)

T+50m:  LightGBM trained
        → Similar improvements
        → Registered as lgbm_v6 (SHADOW)

T+1h:   N-HiTS training starts (slower, DL model)
        → 500 epochs
        → Early stopping on val loss

T+2h:   PatchTST training completes
        → All 4 models registered as SHADOW

T+2h:   Shadow Tester activated
        → Starts parallel evaluation
        → Target: 1000 samples before promotion

T+3d:   Shadow testing continues
        → 850 samples collected
        → xgb_v8: 62% winrate vs 57% for xgb_v7
        → Sharpe: 1.8 vs 1.4

T+5d:   Shadow testing reaches 1000 samples
        → xgb_v8: 63% winrate (7% improvement) ✅
        → lgbm_v6: 61% winrate (5% improvement) ✅
        → Shadow Tester recommends PROMOTION

T+5d+1h: Promotion executed
         → xgb_v7 status: ACTIVE → RETIRED
         → xgb_v8 status: SHADOW → ACTIVE
         → Same for lgbm
         → Publishes "model.promoted" events
         → AI Trading Engine automatically uses new models

T+5d+2h: Drift Detector next check
         → Drift resolved (p_value=0.12) ✅
         → System stabilized
```

---

### Scenario 2: RL Reset on Regime Shift

**Timeline:**

```
T+0h:   RegimeDetector detects major shift
        → Market: TRENDING_LOW_VOL → CHOPPY_HIGH_VOL
        → Publishes "regime.changed" event

T+1m:   CLM receives event
        → Checks PolicyStore: reset_on_regime_change = true
        → Calls rl_meta_strategy.reset_on_regime_shift()
        → Calls rl_position_sizing.reset_on_regime_shift()

T+2m:   RL Meta Strategy reset
        → Old Q-table saved as meta_v12_archived.json
        → New Q-table initialized (empty, epsilon=0.5)
        → Version incremented: v12 → v13
        → Publishes "rl.meta.reset" event

T+3m:   RL Position Sizing reset
        → Old Q-table saved as sizing_v34_archived.json
        → New Q-table initialized
        → Version incremented: v34 → v35
        → Publishes "rl.sizing.reset" event

T+1h:   First trades in new regime
        → RL agents explore with high epsilon
        → Learning from scratch for new conditions

T+24h:  50 trades completed
        → Q-tables filling up
        → Epsilon decay: 0.5 → 0.4

T+7d:   500 trades completed
        → Q-tables mature
        → Epsilon decay: 0.4 → 0.1
        → Performance stabilized in new regime

T+14d:  Next regime check
        → Still CHOPPY_HIGH_VOL
        → No reset needed
        → RL continues learning
```

---

## 🔌 EventBus Integration Points

### Events Published

| Module | Event | Payload |
|--------|-------|---------|
| DataPipeline | `data.fetch.started` | `{symbols, start_date, end_date}` |
| DataPipeline | `data.fetch.completed` | `{rows, features, duration_sec}` |
| ModelTraining | `model.training.started` | `{model_type, job_id}` |
| ModelTraining | `model.training.completed` | `{model_id, metrics, duration_sec}` |
| ModelRegistry | `model.registered` | `{model_id, model_type, version}` |
| ModelRegistry | `model.promoted` | `{model_id, old_status, new_status}` |
| ModelRegistry | `model.retired` | `{model_id, reason}` |
| ShadowTester | `shadow.testing.started` | `{model_id, target_samples}` |
| ShadowTester | `shadow.promotion.recommended` | `{model_id, metrics_comparison}` |
| ShadowTester | `shadow.promoted` | `{model_id, final_metrics}` |
| RLMetaStrategy | `rl.meta.strategy_selected` | `{state, strategy, q_value}` |
| RLMetaStrategy | `rl.meta.updated` | `{state, action, reward, version}` |
| RLMetaStrategy | `rl.meta.reset` | `{old_version, new_version, reason}` |
| RLPositionSizing | `rl.sizing.decision` | `{state, action, size_usd, leverage}` |
| RLPositionSizing | `rl.sizing.updated` | `{state, action, reward, version}` |
| RLPositionSizing | `rl.sizing.reset` | `{old_version, new_version, reason}` |
| DriftDetector | `drift.detected` | `{drift_type, feature, p_value, severity}` |
| ModelSupervisor | `model.performance.degraded` | `{model_id, metrics, recommendation}` |
| CLM | `clm.retraining.started` | `{job_id, trigger_reason}` |
| CLM | `clm.retraining.completed` | `{job_id, models_trained, duration_sec}` |

### Events Subscribed

| Module | Subscribes To | Purpose |
|--------|---------------|---------|
| ShadowTester | `ai.signal.generated` | Get predictions for comparison |
| ShadowTester | `execution.trade.closed` | Record actual outcomes |
| RLMetaStrategy | `execution.trade.closed` | Update Q-table with rewards |
| RLPositionSizing | `execution.trade.closed` | Update Q-table with rewards |
| DriftDetector | `ai.prediction.logged` | Monitor prediction distribution |
| ModelSupervisor | `ai.prediction.logged` | Track model performance |
| ModelSupervisor | `execution.trade.closed` | Calculate winrate, calibration |
| CLM | `drift.detected` | Trigger retraining |
| CLM | `model.performance.degraded` | Trigger retraining |
| CLM | `regime.changed` | Reset RL agents |
| CLM | `shadow.promotion.recommended` | Execute promotion |

---

## 🗄️ Database Schemas (Complete)

```sql
-- ============================================================================
-- MODEL REGISTRY
-- ============================================================================

CREATE TABLE model_registry (
    model_id VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    version VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- TRAINING, SHADOW, ACTIVE, RETIRED
    metrics JSONB NOT NULL,
    training_config JSONB,
    training_data_range JSONB,
    feature_count INT,
    created_at TIMESTAMP DEFAULT NOW(),
    promoted_at TIMESTAMP,
    retired_at TIMESTAMP,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    notes TEXT,
    UNIQUE(model_type, version)
);

CREATE INDEX idx_model_type_status ON model_registry(model_type, status);
CREATE INDEX idx_created_at_desc ON model_registry(created_at DESC);


-- ============================================================================
-- SHADOW TESTING
-- ============================================================================

CREATE TABLE shadow_test_results (
    id SERIAL PRIMARY KEY,
    shadow_model_id VARCHAR(255) NOT NULL REFERENCES model_registry(model_id),
    active_model_id VARCHAR(255) NOT NULL REFERENCES model_registry(model_id),
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    active_prediction JSONB NOT NULL,
    shadow_prediction JSONB NOT NULL,
    actual_outcome JSONB,  -- NULL until trade closes
    outcome_recorded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_shadow_model ON shadow_test_results(shadow_model_id, timestamp DESC);
CREATE INDEX idx_outcome_null ON shadow_test_results(shadow_model_id) 
    WHERE actual_outcome IS NULL;


-- ============================================================================
-- RL VERSIONS
-- ============================================================================

CREATE TABLE rl_versions (
    id SERIAL PRIMARY KEY,
    rl_type VARCHAR(50) NOT NULL,  -- meta_strategy, position_sizing
    version INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    archived_at TIMESTAMP,
    q_table_file TEXT NOT NULL,
    total_states INT,
    total_updates INT,
    epsilon FLOAT,
    learning_rate FLOAT,
    metadata JSONB,
    UNIQUE(rl_type, version)
);

CREATE INDEX idx_rl_type_version ON rl_versions(rl_type, version DESC);


-- ============================================================================
-- DRIFT EVENTS
-- ============================================================================

CREATE TABLE drift_events (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMP NOT NULL,
    drift_type VARCHAR(50) NOT NULL,  -- feature, prediction, performance
    feature_name VARCHAR(100),
    p_value FLOAT,
    threshold FLOAT,
    severity VARCHAR(20) NOT NULL,  -- low, medium, high, critical
    metadata JSONB,
    action_taken VARCHAR(50),  -- retraining_triggered, alert_sent, none
    retraining_job_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_detected_at ON drift_events(detected_at DESC);
CREATE INDEX idx_severity ON drift_events(severity) WHERE severity IN ('high', 'critical');


-- ============================================================================
-- MODEL PERFORMANCE LOGS
-- ============================================================================

CREATE TABLE model_performance_logs (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) NOT NULL REFERENCES model_registry(model_id),
    timestamp TIMESTAMP NOT NULL,
    winrate FLOAT,
    total_trades INT,
    avg_confidence FLOAT,
    calibration_error FLOAT,
    long_short_ratio FLOAT,
    avg_latency_ms FLOAT,
    total_predictions INT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_timestamp ON model_performance_logs(model_id, timestamp DESC);


-- ============================================================================
-- RETRAINING JOBS
-- ============================================================================

CREATE TABLE retraining_jobs (
    job_id VARCHAR(255) PRIMARY KEY,
    job_type VARCHAR(50) NOT NULL,  -- full, partial, rl_maintenance
    trigger_reason VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,  -- running, completed, failed
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    duration_seconds INT,
    models_trained JSONB,  -- {xgb: "model_id", lgbm: "model_id", ...}
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_started_at ON retraining_jobs(started_at DESC);
CREATE INDEX idx_status ON retraining_jobs(status);
```

---

## 🧪 Testing Strategy

### 1. Unit Tests

Test each module in isolation:

```python
# test_data_pipeline.py
async def test_fetch_historical_data():
    fetcher = HistoricalDataFetcher()
    data = await fetcher.fetch_historical_data(
        symbols=["BTCUSDT"],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 7),
        timeframe="1h"
    )
    assert len(data) == 168  # 7 days * 24 hours

# test_model_training.py
async def test_train_xgboost():
    # Use dummy data
    data = generate_dummy_features()
    model, metrics = await train_xgboost(data, config)
    assert metrics.accuracy > 0.5
    assert model is not None
```

### 2. Integration Tests

Test module interactions:

```python
# test_retraining_workflow.py
async def test_full_retraining_workflow():
    """
    End-to-end test:
    1. Fetch data
    2. Train models
    3. Register in registry
    4. Verify SHADOW status
    """
    orchestrator = RetrainingOrchestrator()
    result = await orchestrator.run_full_supervised_retraining(
        symbols=["BTCUSDT"],
        lookback_days=30
    )
    
    assert result.status == "completed"
    assert "xgboost" in result.models_trained
    
    # Verify registry
    model = await model_registry.get_shadow_model("xgboost")
    assert model.status == ModelStatus.SHADOW
```

### 3. Shadow Testing Simulation

Test with historical data:

```python
# test_shadow_tester.py
async def test_shadow_promotion():
    """
    Simulate 1000 trades with ACTIVE and SHADOW models.
    Verify promotion logic.
    """
    shadow_tester = ShadowTester()
    
    # Create fake models
    active_model = MockModel(winrate=0.55)
    shadow_model = MockModel(winrate=0.62)
    
    # Simulate 1000 predictions
    for i in range(1000):
        symbol = random.choice(["BTCUSDT", "ETHUSDT"])
        
        active_pred = active_model.predict(symbol)
        shadow_pred = shadow_model.predict(symbol)
        
        await shadow_tester.record_shadow_prediction(
            shadow_model_id="shadow_1",
            active_model_id="active_1",
            symbol=symbol,
            active_pred=active_pred,
            shadow_pred=shadow_pred
        )
        
        # Simulate outcome
        outcome = simulate_trade_outcome(active_pred)
        await shadow_tester.record_outcome(i, outcome)
    
    # Check promotion
    evaluation = await shadow_tester.evaluate_shadow_model("shadow_1")
    assert evaluation.recommendation == "PROMOTE"
```

### 4. Replay Testing

Use historical trades to test RL updates:

```python
# test_rl_replay.py
async def test_rl_position_sizing_replay():
    """
    Replay 3 months of historical trades.
    Verify Q-table learns optimal sizing.
    """
    agent = RLPositionSizingAgent()
    
    # Load historical trades from DB
    trades = load_historical_trades(
        start_date=datetime(2024, 9, 1),
        end_date=datetime(2024, 12, 1)
    )
    
    # Replay each trade
    for trade in trades:
        state = extract_state(trade)
        action = agent.select_action(state)
        
        # Update with actual outcome
        reward = calculate_reward(trade.pnl_pct, trade.duration_hours)
        next_state = extract_next_state(trade)
        
        agent.update_q_table(state, action, reward, next_state)
    
    # Verify learning
    assert agent.epsilon < 0.1  # Exploration reduced
    assert len(agent.q_table) > 100  # States populated
    
    # Test on validation set
    val_trades = load_historical_trades(
        start_date=datetime(2024, 12, 1),
        end_date=datetime(2024, 12, 15)
    )
    
    val_pnl = simulate_rl_performance(agent, val_trades)
    assert val_pnl > 0  # Net positive
```

### 5. Observability Tests

Verify logging and events:

```python
# test_event_integration.py
async def test_clm_event_flow():
    """
    Verify EventBus integration.
    """
    event_log = []
    
    # Subscribe to all CLM events
    event_bus.subscribe("clm.*", lambda e: event_log.append(e))
    
    # Trigger retraining
    clm = ContinuousLearningManager()
    await clm.trigger_full_retraining(reason="test")
    
    # Wait for completion
    await asyncio.sleep(60)
    
    # Verify events
    assert any(e["type"] == "clm.retraining.started" for e in event_log)
    assert any(e["type"] == "clm.retraining.completed" for e in event_log)
```

---

## 📈 Observability & Monitoring

### 1. Metrics to Track

**Retraining Metrics:**
- Retraining frequency (jobs per week)
- Job success rate
- Average duration per job
- Models trained per job

**Model Performance:**
- Active model versions
- Model winrate over time
- Calibration error over time
- Prediction latency (p50, p95, p99)

**Shadow Testing:**
- Shadow models currently testing
- Samples collected per shadow
- Promotion rate (% promoted)
- Time to promotion (avg days)

**RL Metrics:**
- Q-table size (states)
- Q-table updates per day
- Epsilon decay over time
- RL winrate vs random baseline

**Drift Metrics:**
- Drift events per week
- P-values distribution
- Time to retraining after drift

### 2. Dashboards

**CLM Dashboard** (Grafana):
```
┌─────────────────────────────────────────────┐
│  CONTINUOUS LEARNING MANAGER DASHBOARD      │
├─────────────────────────────────────────────┤
│  Retraining Status:  [IDLE / RUNNING]       │
│  Last Retraining:    2 days ago             │
│  Next Scheduled:     in 5 days              │
├─────────────────────────────────────────────┤
│  Active Models:                             │
│    XGBoost:     v8  (promoted 2d ago)       │
│    LightGBM:    v6  (promoted 2d ago)       │
│    N-HiTS:      v4  (promoted 7d ago)       │
│    PatchTST:    v3  (promoted 7d ago)       │
├─────────────────────────────────────────────┤
│  Shadow Models:                             │
│    XGBoost v9:  Testing (850/1000 samples)  │
│    Metrics:     62% winrate (+5% vs active) │
├─────────────────────────────────────────────┤
│  RL Agents:                                 │
│    Meta Strat:  v13 (1250 states)           │
│    Pos Sizing:  v35 (3400 states)           │
├─────────────────────────────────────────────┤
│  Drift Status:  [✓ NO DRIFT DETECTED]       │
│  Last Check:    12 minutes ago              │
├─────────────────────────────────────────────┤
│  Recent Jobs:                               │
│    RTN-20251202-001:  COMPLETED (45min)     │
│    RTN-20251129-012:  COMPLETED (52min)     │
│    RTN-20251125-008:  FAILED (timeout)      │
└─────────────────────────────────────────────┘
```

### 3. Alerts

**Critical Alerts:**
- Retraining job failed
- Drift severity CRITICAL detected
- All models performance DEGRADED
- Shadow testing stuck (no progress >24h)

**Warning Alerts:**
- Drift severity HIGH detected
- Single model performance degraded
- RL Q-table reset triggered
- Shadow promotion recommended (manual review)

---

## 🚀 Deployment & Rollout

### Phase 1: Setup (Week 1)
1. Create database schemas
2. Deploy all modules to `backend/domains/learning/`
3. Configure PolicyStore with CLM settings
4. Add API endpoints to `main.py`

### Phase 2: Testing (Week 2)
1. Run integration tests with dummy data
2. Replay testing with 6 months historical data
3. Verify EventBus integration
4. Load testing (simulate 1000 retraining jobs)

### Phase 3: Shadow Deployment (Week 3)
1. Deploy CLM with scheduler DISABLED
2. Manual trigger for first retraining
3. Monitor shadow testing for 1 week
4. No production impact (shadow only)

### Phase 4: Production Rollout (Week 4)
1. Enable CLM scheduler (7-day interval)
2. Enable drift detector (6-hour checks)
3. Enable auto-promotion (with thresholds)
4. Monitor closely for 2 weeks

### Phase 5: Optimization (Week 5-6)
1. Tune hyperparameters based on results
2. Adjust promotion thresholds
3. Optimize training speed (GPU, parallel)
4. Add more features to models

---

## 📚 Summary

This ML/AI pipeline architecture provides:

✅ **Complete automation** - No manual intervention needed  
✅ **Continuous learning** - Models adapt to changing markets  
✅ **Safety first** - Shadow testing before promotion  
✅ **RL maintenance** - Q-tables stay fresh and relevant  
✅ **Drift detection** - Proactive model updates  
✅ **Production-ready** - Full observability, error handling, versioning  
✅ **Extensible** - Easy to add new models or features  
✅ **Event-driven** - Decoupled, reactive architecture  
✅ **Policy-driven** - All parameters configurable via PolicyStore  

**Next:** Implementation of all 10 code modules begins now.
