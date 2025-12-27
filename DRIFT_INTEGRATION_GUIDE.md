# DRIFT DETECTION INTEGRATION GUIDE
## Module 3: Connecting Drift Detection to Quantum Trader

**Document Purpose:** Complete integration instructions for drift detection system

---

## INTEGRATION OVERVIEW

**What Gets Connected:**
- **DriftDetectionManager** monitors all 4 AI models (XGBoost, LightGBM, N-HiTS, PatchTST)
- **Feature distributions** tracked continuously from market data
- **Performance metrics** computed after each trade
- **Automatic retraining** triggered when drift detected
- **Integration with Module 2 (RL):** RL handles short-term weight shifts, Drift handles long-term degradation

**Files to Modify:** 6 files

---

## FILE 1: `backend/services/ai/ai_trading_engine.py`

### Changes Required

**1. Import DriftDetectionManager**
```python
# Add to imports at top of file
from services.ai.drift_detection_manager import (
    DriftDetectionManager,
    DriftAlert,
    DriftContext,
    DriftSeverity,
    RetrainingUrgency
)
```

**2. Initialize Drift Manager in `__init__`**
```python
class AITradingEngine:
    def __init__(self, ...):
        # Existing initializations
        self.memory_state_manager = MemoryStateManager(...)
        self.reinforcement_manager = ReinforcementSignalManager(...)
        
        # NEW: Initialize drift detection
        self.drift_manager = DriftDetectionManager(
            psi_severe_threshold=0.25,
            ks_p_value_threshold=0.01,
            win_rate_drop_threshold=0.05,
            performance_window_size=100,
            checkpoint_path="/app/data/drift_checkpoints/drift_state.json"
        )
        
        # Establish baselines on first run (after initial training)
        self._establish_drift_baselines_if_needed()
        
        logger.info("DriftDetectionManager initialized")
```

**3. Add Baseline Establishment Method**
```python
def _establish_drift_baselines_if_needed(self) -> None:
    """
    Establish drift detection baselines for all models.
    
    Called on startup if no existing baselines found.
    """
    # Check if baselines already exist
    if len(self.drift_manager.baseline_distributions) > 0:
        logger.info("Drift baselines already established")
        return
    
    logger.info("Establishing drift baselines for all models...")
    
    # Load historical data for baseline (last 500 trades)
    historical_trades = self._load_historical_trades(limit=500)
    
    if len(historical_trades) < 100:
        logger.warning("Insufficient historical data for drift baselines")
        return
    
    # Prepare data per model
    for model_name in ['xgboost', 'lightgbm', 'nhits', 'patchtst']:
        try:
            # Extract features, predictions, outcomes
            feature_values = self._extract_features_from_trades(historical_trades)
            predictions = np.array([t[f'{model_name}_prediction'] for t in historical_trades])
            outcomes = np.array([1 if t['pnl'] > 0 else 0 for t in historical_trades])
            confidences = np.array([t[f'{model_name}_confidence'] for t in historical_trades])
            
            # Establish baseline
            self.drift_manager.establish_baseline(
                model_name=model_name,
                feature_values=feature_values,
                predictions=predictions,
                actual_outcomes=outcomes,
                confidences=confidences
            )
            
            logger.info(f"Baseline established for {model_name}")
        
        except Exception as e:
            logger.error(f"Failed to establish baseline for {model_name}: {e}")
    
    # Save checkpoint
    self.drift_manager.checkpoint()
```

**4. Add Drift Detection to Signal Generation**
```python
async def generate_signal(self, symbol: str, market_data: Dict) -> Optional[TradingSignal]:
    """
    Generate trading signal with drift monitoring.
    """
    # Existing signal generation
    memory_context = self.memory_state_manager.get_memory_context(symbol)
    rl_context = self.reinforcement_manager.get_reinforcement_context()
    
    # Get model predictions
    model_predictions = await self.ensemble_manager.predict_with_rl(
        symbol=symbol,
        market_data=market_data,
        memory_context=memory_context,
        rl_context=rl_context
    )
    
    # NEW: Check drift status before trading
    drift_context = self.drift_manager.get_drift_context(model_name='ensemble')
    
    # If critical drift detected, reduce confidence or skip trade
    if drift_context.active_alerts:
        for alert in drift_context.active_alerts:
            if alert.severity in ['severe', 'critical']:
                logger.warning(
                    f"Critical drift detected for ensemble: {alert.drift_types}. "
                    f"Reducing confidence by 30%"
                )
                # Reduce confidence to account for model uncertainty
                model_predictions['confidence'] *= 0.70
    
    # Rest of signal generation logic
    # ...
    
    return signal
```

**5. Add Drift Tracking After Trade Execution**
```python
def process_trade_result(
    self,
    trade_result: Dict,
    model_predictions: Dict,
    feature_values: Dict
) -> None:
    """
    Process trade result for RL and drift detection.
    """
    # Existing RL processing
    self.reinforcement_manager.process_trade_outcome(...)
    
    # NEW: Update drift detection for each model
    for model_name in ['xgboost', 'lightgbm', 'nhits', 'patchtst']:
        self.drift_manager.process_trade_outcome(
            model_name=model_name,
            prediction=model_predictions[model_name]['prediction'],
            confidence=model_predictions[model_name]['confidence'],
            actual_outcome=1 if trade_result['pnl'] > 0 else 0,
            feature_values=feature_values
        )
    
    # Periodic drift detection (every 100 trades)
    if self.drift_manager.trades_since_baseline.get('xgboost', 0) % 100 == 0:
        self._check_drift_all_models(feature_values)
```

**6. Add Periodic Drift Check Method**
```python
def _check_drift_all_models(self, recent_feature_values: Dict) -> None:
    """
    Check for drift across all models.
    
    Called periodically (every 100 trades) or on-demand.
    """
    logger.info("Running drift detection check across all models...")
    
    # Collect recent data (last 100 trades)
    recent_trades = self._get_recent_trades(limit=100)
    
    if len(recent_trades) < 50:
        logger.debug("Insufficient recent data for drift detection")
        return
    
    # Check each model
    for model_name in ['xgboost', 'lightgbm', 'nhits', 'patchtst']:
        try:
            # Prepare data
            feature_values = self._extract_features_from_trades(recent_trades)
            predictions = np.array([t[f'{model_name}_prediction'] for t in recent_trades])
            outcomes = np.array([1 if t['pnl'] > 0 else 0 for t in recent_trades])
            confidences = np.array([t[f'{model_name}_confidence'] for t in recent_trades])
            
            # Detect drift
            alert = self.drift_manager.detect_drift(
                model_name=model_name,
                feature_values=feature_values,
                predictions=predictions,
                actual_outcomes=outcomes,
                confidences=confidences
            )
            
            # Handle alert if drift detected
            if alert:
                self._handle_drift_alert(alert)
        
        except Exception as e:
            logger.error(f"Drift detection failed for {model_name}: {e}")
```

**7. Add Drift Alert Handler**
```python
def _handle_drift_alert(self, alert: DriftAlert) -> None:
    """
    Handle drift alert and trigger retraining if needed.
    """
    logger.warning(
        f"DRIFT ALERT: {alert.model_name} - "
        f"Severity={alert.severity}, Urgency={alert.urgency}, "
        f"Types={alert.drift_types}"
    )
    
    # Trigger retraining for urgent/immediate alerts
    if alert.urgency in ['urgent', 'immediate']:
        retrain_job = self.drift_manager.trigger_retraining(
            model_name=alert.model_name,
            alert=alert
        )
        
        logger.warning(f"Retraining job queued: {retrain_job['job_id']}")
        
        # Send notification (email, Slack, etc.)
        self._send_drift_notification(alert, retrain_job)
    
    # For scheduled retraining, add to queue
    elif alert.urgency == 'scheduled':
        logger.info(f"Scheduled retraining for {alert.model_name} within 72 hours")
        # Add to retraining scheduler
    
    # For monitoring, just log
    else:
        logger.info(f"Monitoring {alert.model_name} for further drift")
```

---

## FILE 2: `backend/services/ai/ensemble_manager.py`

### Changes Required

**1. Add Drift-Aware Prediction Method**
```python
async def predict_with_drift_awareness(
    self,
    symbol: str,
    market_data: Dict,
    memory_context: MemoryContext,
    rl_context: ReinforcementContext,
    drift_context: DriftContext  # NEW parameter
) -> Dict:
    """
    Generate ensemble prediction with drift awareness.
    
    Reduces weight of models with active drift alerts.
    """
    # Get individual model predictions
    model_predictions = await self._get_all_model_predictions(symbol, market_data)
    
    # Apply RL weights
    rl_weights = rl_context.model_weights.to_dict()
    
    # NEW: Adjust weights based on drift status
    if drift_context.active_alerts:
        logger.warning(f"Adjusting weights due to {len(drift_context.active_alerts)} drift alerts")
        
        for alert in drift_context.active_alerts:
            model_name = alert.model_name
            
            # Reduce weight based on severity
            if alert.severity == 'critical':
                rl_weights[model_name] *= 0.30  # Reduce to 30%
            elif alert.severity == 'severe':
                rl_weights[model_name] *= 0.60  # Reduce to 60%
            elif alert.severity == 'moderate':
                rl_weights[model_name] *= 0.85  # Reduce to 85%
        
        # Renormalize weights
        total = sum(rl_weights.values())
        rl_weights = {k: v / total for k, v in rl_weights.items()}
    
    # Compute weighted ensemble prediction
    ensemble_prediction = sum(
        model_predictions[model]['prediction'] * rl_weights[model]
        for model in rl_weights.keys()
    )
    
    ensemble_confidence = sum(
        model_predictions[model]['confidence'] * rl_weights[model]
        for model in rl_weights.keys()
    )
    
    return {
        'prediction': ensemble_prediction,
        'confidence': ensemble_confidence,
        'model_predictions': model_predictions,
        'applied_weights': rl_weights,
        'drift_adjusted': len(drift_context.active_alerts) > 0
    }
```

---

## FILE 3: `backend/services/ai/event_driven_executor.py`

### Changes Required

**1. Extract Features for Drift Tracking**
```python
async def execute_trade(self, signal: TradingSignal) -> Dict:
    """
    Execute trade and track features for drift detection.
    """
    # Execute trade (existing logic)
    result = await self._execute_binance_order(signal)
    
    # NEW: Extract feature values for drift tracking
    feature_values = self._extract_current_features(signal.symbol)
    
    # Process trade result for RL
    self.ai_engine.process_trade_result(
        trade_result=result,
        model_predictions=signal.model_predictions,
        feature_values=feature_values  # NEW parameter
    )
    
    return result

def _extract_current_features(self, symbol: str) -> Dict[str, float]:
    """
    Extract current feature values for drift detection.
    
    Returns dictionary of feature_name → value.
    """
    # Get latest market data
    market_data = self.market_data_service.get_latest(symbol)
    
    # Calculate technical indicators (same as used in training)
    features = {
        'rsi_14': market_data['rsi_14'],
        'macd': market_data['macd'],
        'bb_position': market_data['bb_position'],
        'volume_sma_ratio': market_data['volume'] / market_data['volume_sma_20'],
        'atr_14': market_data['atr_14'],
        'volatility_30d': market_data['volatility_30d'],
        # Add all features used in model training
    }
    
    return features
```

---

## FILE 4: `backend/services/ai/ai_hedgefund_os.py`

### Changes Required

**1. Add Drift Monitoring to Health Check**
```python
async def system_health_check(self) -> Dict:
    """
    Comprehensive health check including drift status.
    """
    health = {
        'timestamp': datetime.now().isoformat(),
        'memory_state': self.memory_manager.get_diagnostics(),
        'reinforcement_learning': self.rl_manager.get_diagnostics(),
        'drift_detection': self.drift_manager.get_diagnostics(),  # NEW
    }
    
    # Check for critical drift alerts
    drift_diag = health['drift_detection']
    if drift_diag['active_alerts_count'] > 0:
        health['status'] = 'WARNING'
        health['warnings'] = [
            f"Active drift alerts: {drift_diag['active_alerts_count']}"
        ]
    
    # Check for degraded models
    for model_name, status in drift_diag['models_status'].items():
        if status['status'] == 'DEGRADED':
            health['warnings'].append(
                f"{model_name} performance degraded: WR delta={status['win_rate_delta']:.3f}"
            )
    
    return health
```

---

## FILE 5: `backend/main.py`

### Changes Required

**1. Add Drift Checkpoint Management**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan with drift checkpoint management"""
    logger.info("Starting AI-HFOS with drift detection...")
    
    # Startup
    ai_engine = get_ai_engine()
    
    # Establish drift baselines if needed
    ai_engine._establish_drift_baselines_if_needed()
    
    # Load drift checkpoint
    ai_engine.drift_manager._load_checkpoint()
    
    yield
    
    # Shutdown - save drift state
    logger.info("Shutting down, saving drift checkpoints...")
    ai_engine.drift_manager.checkpoint()
    
    logger.info("Shutdown complete")
```

---

## FILE 6: `backend/routes/ai.py`

### Changes Required

**1. Add Drift Detection Endpoints**

```python
@router.get("/drift/diagnostics")
async def get_drift_diagnostics():
    """
    Get drift detection diagnostics for all models.
    
    Returns:
        - Models tracked
        - Active alerts count
        - PSI scores per feature
        - Performance deltas
    """
    ai_engine = get_ai_engine()
    diagnostics = ai_engine.drift_manager.get_diagnostics()
    
    return {
        'status': 'success',
        'data': diagnostics
    }

@router.get("/drift/context/{model_name}")
async def get_drift_context(model_name: str):
    """
    Get drift context for specific model.
    
    Args:
        model_name: 'xgboost', 'lightgbm', 'nhits', or 'patchtst'
    """
    ai_engine = get_ai_engine()
    
    try:
        context = ai_engine.drift_manager.get_drift_context(model_name)
        
        return {
            'status': 'success',
            'data': asdict(context)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift/alerts/{model_name}")
async def get_drift_alerts(model_name: str):
    """Get active drift alerts for model"""
    ai_engine = get_ai_engine()
    
    alerts = ai_engine.drift_manager.active_alerts.get(model_name, [])
    
    return {
        'status': 'success',
        'data': {
            'model_name': model_name,
            'active_alerts_count': len(alerts),
            'alerts': [asdict(alert) for alert in alerts]
        }
    }

@router.post("/drift/baseline/reset/{model_name}")
async def reset_drift_baseline(model_name: str):
    """
    Reset drift baseline for model after retraining.
    
    Call this endpoint after successfully retraining a model.
    """
    ai_engine = get_ai_engine()
    
    # Load new training data
    new_trades = ai_engine._load_historical_trades(limit=500)
    
    # Extract data
    feature_values = ai_engine._extract_features_from_trades(new_trades)
    predictions = np.array([t[f'{model_name}_prediction'] for t in new_trades])
    outcomes = np.array([1 if t['pnl'] > 0 else 0 for t in new_trades])
    confidences = np.array([t[f'{model_name}_confidence'] for t in new_trades])
    
    # Reset baseline
    ai_engine.drift_manager.reset_baseline_after_retrain(
        model_name=model_name,
        new_feature_values=feature_values,
        new_predictions=predictions,
        new_actual_outcomes=outcomes,
        new_confidences=confidences
    )
    
    return {
        'status': 'success',
        'message': f'Baseline reset for {model_name}',
        'new_baseline_trades': len(predictions)
    }

@router.post("/drift/check/{model_name}")
async def trigger_drift_check(model_name: str):
    """
    Manually trigger drift check for a model.
    
    Useful for on-demand analysis.
    """
    ai_engine = get_ai_engine()
    
    # Get recent trades
    recent_trades = ai_engine._get_recent_trades(limit=100)
    
    # Extract data
    feature_values = ai_engine._extract_features_from_trades(recent_trades)
    predictions = np.array([t[f'{model_name}_prediction'] for t in recent_trades])
    outcomes = np.array([1 if t['pnl'] > 0 else 0 for t in recent_trades])
    confidences = np.array([t[f'{model_name}_confidence'] for t in recent_trades])
    
    # Detect drift
    alert = ai_engine.drift_manager.detect_drift(
        model_name=model_name,
        feature_values=feature_values,
        predictions=predictions,
        actual_outcomes=outcomes,
        confidences=confidences
    )
    
    if alert:
        return {
            'status': 'drift_detected',
            'alert': asdict(alert)
        }
    else:
        return {
            'status': 'no_drift',
            'message': f'No drift detected for {model_name}'
        }
```

---

## INTEGRATION FLOW DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                     DRIFT DETECTION FLOW                        │
└─────────────────────────────────────────────────────────────────┘

1. INITIALIZATION (Startup)
   ↓
   [main.py] → Load checkpoints
   ↓
   [ai_trading_engine.py] → Establish baselines (if needed)
   ↓
   500 historical trades → Compute feature distributions + performance
   ↓
   Baseline stored in DriftDetectionManager

2. SIGNAL GENERATION (Per Trade)
   ↓
   [ai_trading_engine.py] → Get drift_context
   ↓
   Check active alerts → Adjust confidence if drift detected
   ↓
   [ensemble_manager.py] → Apply drift-adjusted weights
   ↓
   Generate signal with reduced confidence for degraded models

3. TRADE EXECUTION
   ↓
   [event_driven_executor.py] → Extract feature values
   ↓
   Execute trade → Get PnL result

4. POST-TRADE PROCESSING
   ↓
   [ai_trading_engine.py] → process_trade_result()
   ↓
   [drift_manager] → process_trade_outcome() for each model
   ↓
   Store: prediction, confidence, outcome, features
   ↓
   Every 100 trades → Trigger drift detection

5. DRIFT DETECTION (Every 100 Trades)
   ↓
   [drift_manager] → detect_drift()
   ↓
   ┌─────────────────────────────────────────────────┐
   │ Stage 1: PSI (Feature Distribution Drift)      │
   │ Stage 2: KS-test (Prediction Distribution)     │
   │ Stage 3: Performance Degradation               │
   │ Stage 4: Determine Severity + Urgency          │
   └─────────────────────────────────────────────────┘
   ↓
   IF drift detected → Create DriftAlert

6. ALERT HANDLING
   ↓
   [ai_trading_engine.py] → _handle_drift_alert()
   ↓
   IF urgency = IMMEDIATE/URGENT:
      ↓
      trigger_retraining() → Queue retrain job
      ↓
      Send notifications (email/Slack)
   ↓
   IF urgency = SCHEDULED:
      ↓
      Schedule retraining within 72h
   ↓
   Update drift context

7. RETRAINING (When Triggered)
   ↓
   External retraining pipeline executes
   ↓
   New model trained on recent data (last 3 months)
   ↓
   Deploy new model weights

8. POST-RETRAINING
   ↓
   Call API: POST /drift/baseline/reset/{model_name}
   ↓
   [drift_manager] → reset_baseline_after_retrain()
   ↓
   Clear alerts, reset counters, establish new baseline
   ↓
   Resume normal operation

9. MONITORING (Continuous)
   ↓
   [ai_hedgefund_os.py] → system_health_check()
   ↓
   Include drift diagnostics in health status
   ↓
   Dashboard displays: PSI scores, WR deltas, active alerts
```

---

## CONFIGURATION VARIABLES

Add to `.env`:

```bash
# Drift Detection Parameters
DRIFT_PSI_SEVERE_THRESHOLD=0.25
DRIFT_PSI_MODERATE_THRESHOLD=0.15
DRIFT_PSI_MINOR_THRESHOLD=0.10
DRIFT_PSI_BINS=10

DRIFT_KS_P_VALUE_THRESHOLD=0.01

DRIFT_WR_DROP_THRESHOLD=0.05        # 5 pp
DRIFT_F1_DROP_THRESHOLD=0.10        # 10%
DRIFT_CALIBRATION_THRESHOLD=0.08    # 8 pp

DRIFT_PERFORMANCE_WINDOW_SIZE=100   # Trades per window
DRIFT_MIN_TRADES_DETECTION=50
DRIFT_CONSECUTIVE_WINDOWS=2

DRIFT_URGENT_RETRAIN_TRADES=200
DRIFT_SCHEDULED_RETRAIN_HOURS=72

DRIFT_CHECKPOINT_PATH=/app/data/drift_checkpoints/drift_state.json
DRIFT_CHECKPOINT_INTERVAL=300       # 5 minutes
```

---

## INTEGRATION TEST SCRIPT

```python
# test_drift_integration.py

import asyncio
from backend.services.ai.ai_trading_engine import AITradingEngine

async def test_drift_integration():
    """Test drift detection integration"""
    
    # Initialize engine
    engine = AITradingEngine()
    
    print("1. Testing baseline establishment...")
    engine._establish_drift_baselines_if_needed()
    assert len(engine.drift_manager.baseline_distributions) > 0
    print("✓ Baselines established")
    
    print("\n2. Testing drift detection...")
    # Simulate 100 trades with degraded performance
    for i in range(100):
        engine.drift_manager.process_trade_outcome(
            model_name='xgboost',
            prediction=0.55,
            confidence=0.60,
            actual_outcome=0 if i % 3 == 0 else 1,  # 33% WR (degraded)
            feature_values={'rsi_14': 50.0 + i * 0.5}
        )
    
    # Trigger drift check
    engine._check_drift_all_models(recent_feature_values={})
    
    # Check for alerts
    context = engine.drift_manager.get_drift_context('xgboost')
    assert len(context.active_alerts) > 0
    print(f"✓ Drift detected: {len(context.active_alerts)} alerts")
    
    print("\n3. Testing drift-aware prediction...")
    signal = await engine.generate_signal(
        symbol='BTCUSDT',
        market_data={}
    )
    # Confidence should be reduced due to drift
    assert signal.confidence < 0.70
    print(f"✓ Confidence reduced to {signal.confidence:.2f}")
    
    print("\n4. Testing baseline reset...")
    # Simulate retraining
    new_trades = engine._load_historical_trades(limit=500)
    feature_values = engine._extract_features_from_trades(new_trades)
    predictions = [0.65] * 500
    outcomes = [1] * 300 + [0] * 200  # 60% WR
    confidences = [0.65] * 500
    
    engine.drift_manager.reset_baseline_after_retrain(
        model_name='xgboost',
        new_feature_values=feature_values,
        new_predictions=np.array(predictions),
        new_actual_outcomes=np.array(outcomes),
        new_confidences=np.array(confidences)
    )
    
    # Alerts should be cleared
    context = engine.drift_manager.get_drift_context('xgboost')
    assert len(context.active_alerts) == 0
    print("✓ Baseline reset successful")
    
    print("\n✅ All integration tests passed!")

if __name__ == "__main__":
    asyncio.run(test_drift_integration())
```

---

## VERIFICATION CHECKLIST

After integration, verify:

- [ ] **Baselines Established**: Check `/drift/diagnostics` shows 4 models tracked
- [ ] **Trade Tracking**: Verify `trades_since_baseline` increments with each trade
- [ ] **Feature Extraction**: Confirm feature values logged in drift manager
- [ ] **PSI Calculation**: Check PSI scores computed for all features
- [ ] **KS Test**: Verify KS test runs on prediction distributions
- [ ] **Performance Metrics**: Confirm win rate, F1, calibration tracked per window
- [ ] **Drift Detection**: Trigger manual check via `/drift/check/{model}` endpoint
- [ ] **Alert Generation**: Verify alerts created when drift detected
- [ ] **Retraining Trigger**: Confirm retraining job queued for urgent alerts
- [ ] **Drift-Aware Weights**: Check ensemble weights adjusted during drift
- [ ] **Confidence Reduction**: Verify signal confidence reduced for degraded models
- [ ] **Checkpoint Persistence**: Confirm drift state saved every 5 minutes
- [ ] **Checkpoint Restore**: Restart service, verify state restored from checkpoint
- [ ] **Baseline Reset**: Test `/drift/baseline/reset/{model}` after simulated retrain
- [ ] **Health Monitoring**: Check `system_health_check()` includes drift status
- [ ] **API Endpoints**: Test all 5 new drift endpoints
- [ ] **Notifications**: Verify drift alerts sent to monitoring channels
- [ ] **Dashboard**: Confirm drift metrics visible in UI

---

## EXPECTED OUTPUT

**Healthy System (No Drift):**
```json
{
  "timestamp": "2025-11-26T14:30:00",
  "models_tracked": ["xgboost", "lightgbm", "nhits", "patchtst"],
  "active_alerts_count": 0,
  "models_status": {
    "xgboost": {
      "trades_since_baseline": 850,
      "win_rate_delta": 0.02,
      "consecutive_poor_windows": 0,
      "features_with_drift_count": 0,
      "max_psi_score": 0.08,
      "max_psi_feature": "rsi_14",
      "active_alerts": 0,
      "status": "HEALTHY"
    }
  }
}
```

**Degraded System (Drift Detected):**
```json
{
  "timestamp": "2025-11-26T18:45:00",
  "models_tracked": ["xgboost", "lightgbm", "nhits", "patchtst"],
  "active_alerts_count": 2,
  "models_status": {
    "xgboost": {
      "trades_since_baseline": 1250,
      "win_rate_delta": -0.07,
      "consecutive_poor_windows": 3,
      "features_with_drift_count": 4,
      "max_psi_score": 0.31,
      "max_psi_feature": "volatility_30d",
      "active_alerts": 1,
      "status": "DEGRADED"
    },
    "lightgbm": {
      "status": "DEGRADED",
      "active_alerts": 1
    }
  }
}
```

---

## TROUBLESHOOTING

### Issue: "No baseline for model X"
**Solution:** Call `_establish_drift_baselines_if_needed()` on startup or manually via API

### Issue: PSI scores always 0.0
**Solution:** Check feature extraction - ensure features match training data exactly

### Issue: Too many false positive alerts
**Solution:** Increase `psi_severe_threshold` from 0.25 to 0.30 or increase `consecutive_windows_threshold`

### Issue: Drift detection too slow
**Solution:** Reduce `performance_window_size` from 100 to 50 trades

### Issue: Checkpoint file not found
**Solution:** Ensure `/app/data/drift_checkpoints/` directory exists and has write permissions

---

**Integration Status:** Ready for implementation  
**Next Steps:** Risk analysis (Section 5)
