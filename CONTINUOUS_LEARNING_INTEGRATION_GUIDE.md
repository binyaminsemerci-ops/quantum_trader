# CONTINUOUS LEARNING: INTEGRATION GUIDE

**Module 6 Section 4: Integration Guide**

This guide shows how to integrate Continuous Learning with the existing system.

---

## PART 1: ENSEMBLE MANAGER INTEGRATION

### Add Import

```python
# In ai_engine/ensemble_manager.py

# Add after shadow model imports (line ~30):
try:
    from backend.services.ai.continuous_learning_manager import (
        ContinuousLearningManager,
        RetrainingTrigger,
        PerformanceMetrics
    )
    CONTINUOUS_LEARNING_AVAILABLE = True
except ImportError:
    CONTINUOUS_LEARNING_AVAILABLE = False
    logger.warning("[CL] continuous_learning_manager not available")
```

### Initialize in __init__

```python
# In EnsembleManager.__init__ (after shadow manager init, line ~145):

# Continuous Learning Manager (optional)
self.cl_manager = None
self.cl_enabled = False

if CONTINUOUS_LEARNING_AVAILABLE:
    self.cl_enabled = os.getenv('ENABLE_CONTINUOUS_LEARNING', 'false').lower() == 'true'
    
    if self.cl_enabled:
        try:
            self.cl_manager = ContinuousLearningManager(
                storage_path='data/continuous_learning',
                checkpoint_path='data/continuous_learning_checkpoint.json',
                ewma_alpha=float(os.getenv('CL_EWMA_ALPHA', '0.1')),
                ewma_threshold=float(os.getenv('CL_EWMA_THRESHOLD', '3.0')),
                feature_drift_threshold=float(os.getenv('CL_FEATURE_DRIFT_THRESHOLD', '0.3')),
                online_learning_rate=float(os.getenv('CL_ONLINE_LR', '0.01')),
                enable_online_learning=os.getenv('CL_ENABLE_ONLINE', 'true').lower() == 'true'
            )
            
            # Initialize baseline (if champion exists)
            if self.shadow_manager:
                champion_metrics = self.shadow_manager.get_champion_metrics()
                if champion_metrics:
                    self.cl_manager.initialize_baseline(
                        win_rate=champion_metrics['win_rate'],
                        feature_importance={'placeholder': 1.0}  # TODO: Get real importance
                    )
            
            logger.info("[CL] Continuous learning manager ENABLED")
            
        except Exception as e:
            logger.error(f"[CL] Failed to initialize: {e}")
            self.cl_enabled = False
    else:
        logger.info("[CL] Continuous learning DISABLED")
```

### Add Method: record_trade_outcome_for_cl

```python
# Add to EnsembleManager class (after record_trade_outcome_for_shadow):

def record_trade_outcome_for_cl(
    self,
    symbol: str,
    outcome: float,  # 1.0 = win, 0.0 = loss
    features: Dict[str, float],
    pnl: float
) -> Optional[Dict[str, Any]]:
    """
    Record trade outcome for continuous learning monitoring.
    
    Args:
        symbol: Trading pair
        outcome: 1.0 (win) or 0.0 (loss)
        features: Feature dict used for prediction
        pnl: Trade PnL
    
    Returns:
        CL status dict with trigger recommendations
    """
    if not self.cl_enabled or self.cl_manager is None:
        return None
    
    try:
        # Record trade and check triggers
        cl_status = self.cl_manager.record_trade(
            outcome=outcome,
            features=features,
            shap_values=None,  # TODO: Compute SHAP values
            feature_vector=None  # TODO: Convert features to vector
        )
        
        # Log if retraining triggered
        if cl_status['should_retrain']:
            triggers = cl_status['triggers']
            urgency = cl_status['urgency_score']
            
            logger.warning(
                f"[CL] 🔥 Retraining recommended: {triggers} "
                f"(urgency={urgency:.1f}/100)"
            )
            
            # Create retraining event
            if urgency >= 10:  # CRITICAL threshold
                metrics = PerformanceMetrics(
                    win_rate=cl_status['performance']['ewma_wr'],
                    sharpe_ratio=0.0,  # TODO: Compute
                    sortino_ratio=0.0,
                    max_drawdown=0.0,
                    calmar_ratio=0.0,
                    n_trades=cl_status['trade_count']
                )
                
                event_id = self.cl_manager.trigger_retraining(
                    trigger=RetrainingTrigger[triggers[0]],
                    urgency_score=urgency,
                    metrics_before=metrics,
                    reason=f"Urgency={urgency:.1f}, triggers={triggers}"
                )
                
                logger.warning(f"[CL] Retraining event created: {event_id}")
        
        return cl_status
        
    except Exception as e:
        logger.error(f"[CL] Error recording trade: {e}")
        return None
```

### Add Method: get_cl_status

```python
# Add to EnsembleManager class:

def get_cl_status(self) -> Optional[Dict[str, Any]]:
    """Get continuous learning status"""
    if not self.cl_enabled or self.cl_manager is None:
        return None
    
    try:
        return self.cl_manager.get_status()
    except Exception as e:
        logger.error(f"[CL] Error getting status: {e}")
        return None
```

---

## PART 2: API ENDPOINTS

### Add to backend/routes/ai.py

```python
# Add after shadow model endpoints (line ~1500):

# ============================================================================
# CONTINUOUS LEARNING ENDPOINTS
# ============================================================================

@router.get("/continuous-learning/status")
async def get_continuous_learning_status():
    """Get continuous learning status"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        
        # Get ensemble (from global or create)
        ensemble = get_ensemble_instance()
        
        status = ensemble.get_cl_status()
        
        if status is None:
            return {
                "enabled": False,
                "message": "Continuous learning not enabled"
            }
        
        return {
            "enabled": True,
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting CL status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/continuous-learning/history")
async def get_retraining_history(limit: int = 10):
    """Get retraining event history"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        
        ensemble = get_ensemble_instance()
        
        if ensemble.cl_manager is None:
            raise HTTPException(status_code=400, detail="CL not enabled")
        
        history = ensemble.cl_manager.get_retraining_history(limit=limit)
        
        return {
            "events": history,
            "count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting CL history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/continuous-learning/trigger")
async def trigger_manual_retraining(
    reason: str = "Manual trigger",
    urgency: float = 50.0
):
    """Manually trigger retraining"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        from backend.services.ai.continuous_learning_manager import (
            RetrainingTrigger,
            PerformanceMetrics
        )
        
        ensemble = get_ensemble_instance()
        
        if ensemble.cl_manager is None:
            raise HTTPException(status_code=400, detail="CL not enabled")
        
        # Get current metrics
        cl_status = ensemble.cl_manager.get_status()
        
        metrics = PerformanceMetrics(
            win_rate=cl_status.get('performance_ewma', 0.58),
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            n_trades=cl_status['trade_count']
        )
        
        # Trigger
        event_id = ensemble.cl_manager.trigger_retraining(
            trigger=RetrainingTrigger.MANUAL,
            urgency_score=urgency,
            metrics_before=metrics,
            reason=reason
        )
        
        return {
            "success": True,
            "event_id": event_id,
            "message": f"Retraining triggered: {reason}"
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/continuous-learning/performance")
async def get_performance_monitoring():
    """Get performance monitoring metrics"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        
        ensemble = get_ensemble_instance()
        
        if ensemble.cl_manager is None:
            raise HTTPException(status_code=400, detail="CL not enabled")
        
        monitor = ensemble.cl_manager.performance_monitor
        
        return {
            "baseline_wr": monitor.baseline_wr,
            "ewma_wr": monitor.ewma_wr,
            "ewma_decay": (monitor.baseline_wr - monitor.ewma_wr) * 100 if monitor.ewma_wr else 0,
            "cusum_positive": monitor.cusum_positive,
            "cusum_negative": monitor.cusum_negative,
            "history_length": len(monitor.performance_history)
        }
        
    except Exception as e:
        logger.error(f"Error getting performance monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/continuous-learning/features")
async def get_feature_importance():
    """Get current feature importance"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        
        ensemble = get_ensemble_instance()
        
        if ensemble.cl_manager is None:
            raise HTTPException(status_code=400, detail="CL not enabled")
        
        tracker = ensemble.cl_manager.feature_tracker
        
        return {
            "baseline": tracker.baseline_importance,
            "current": tracker.current_importance,
            "top_features": tracker._get_top_features(10)
        }
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/continuous-learning/versions")
async def get_model_versions():
    """Get model version history"""
    try:
        from ai_engine.ensemble_manager import EnsembleManager
        
        ensemble = get_ensemble_instance()
        
        if ensemble.cl_manager is None:
            raise HTTPException(status_code=400, detail="CL not enabled")
        
        versioner = ensemble.cl_manager.versioner
        
        return {
            "current_version": versioner.current_version,
            "history": versioner.get_version_history()
        }
        
    except Exception as e:
        logger.error(f"Error getting versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

---

## PART 3: ENVIRONMENT VARIABLES

### Add to .env

```bash
# Continuous Learning Configuration
ENABLE_CONTINUOUS_LEARNING=true

# Performance Monitoring
CL_EWMA_ALPHA=0.1                  # EWMA smoothing (0.1 = 10% weight to new)
CL_EWMA_THRESHOLD=3.0              # Decay threshold (percentage points)
CL_CUSUM_K=0.5                     # CUSUM slack parameter
CL_CUSUM_H=5.0                     # CUSUM threshold

# Feature Tracking
CL_FEATURE_DRIFT_THRESHOLD=0.3     # JS divergence threshold

# Online Learning
CL_ENABLE_ONLINE=true              # Enable online learning
CL_ONLINE_LR=0.01                  # Learning rate
CL_ONLINE_UPDATE_FREQ=10           # Update every N trades

# Retraining
CL_SCHEDULED_RETRAIN_DAYS=30       # Days between scheduled retrains
```

---

## PART 4: INTEGRATION WITH OTHER MODULES

### Module 1: Memory States

```python
# In record_trade_outcome_for_cl:

# Get state context from memory manager
if self.memory_manager:
    state_context = self.memory_manager.get_state_context(symbol, regime)
    features['memory_confidence'] = state_context['confidence_multiplier']
```

### Module 2: Reinforcement Signals

```python
# In record_trade_outcome_for_cl:

# Compute reward signal
if self.reinforcement_manager:
    reward = self.reinforcement_manager.compute_reward({
        'win': outcome == 1.0,
        'pnl': pnl,
        'sharpe': 0.0  # TODO: Compute
    })
    features['reward_signal'] = reward
```

### Module 3: Drift Detection

```python
# In record_trade_outcome_for_cl:

# Check drift status
if self.drift_detector:
    drift_status = self.drift_detector.check_drift(features, prediction)
    if drift_status['drift_detected']:
        # Trigger CL retraining immediately
        logger.warning("[CL] Drift detected - triggering retraining")
        urgency = 100.0  # Maximum urgency
```

### Module 4: Covariate Shift

```python
# In record_trade_outcome_for_cl:

# Compute feature shift
if self.covariate_manager:
    shift_weights = self.covariate_manager.compute_importance_weights(features)
    # Use shift weights to adjust feature importance in CL
```

### Module 5: Shadow Models

```python
# After successful retraining:

# Deploy retrained model as challenger
new_model_version = "ensemble_v2.0"
self.shadow_manager.register_model(
    model_name=new_model_version,
    model_type='ensemble',
    version='2.0',
    role=ModelRole.CHALLENGER,
    description='Retrained model via continuous learning'
)

# Shadow test for 500 trades
# If better → promote
# If worse → rollback CL version
```

---

## PART 5: USAGE EXAMPLES

### Example 1: Record Trade Outcome

```python
# After trade execution:

outcome = 1.0 if trade_won else 0.0
features = {
    'rsi': 65.0,
    'macd': -0.02,
    'volume': 1_200_000,
    'order_book_imbalance': 0.15
}

cl_status = ensemble.record_trade_outcome_for_cl(
    symbol='BTCUSDT',
    outcome=outcome,
    features=features,
    pnl=150.0
)

if cl_status and cl_status['should_retrain']:
    print(f"🔥 Retraining recommended: {cl_status['triggers']}")
    print(f"Urgency: {cl_status['urgency_score']:.1f}/100")
```

### Example 2: Monitor Performance

```python
# Check continuous learning status
cl_status = ensemble.get_cl_status()

print(f"Trade count: {cl_status['trade_count']}")
print(f"Current version: {cl_status['current_version']}")
print(f"Days since retrain: {cl_status['days_since_retrain']}")
print(f"EWMA WR: {cl_status['performance_ewma']:.2%}")
print(f"Baseline WR: {cl_status['performance_baseline']:.2%}")
```

### Example 3: Trigger Manual Retraining

```python
# Manual trigger (e.g., after major market event)
event_id = ensemble.cl_manager.trigger_retraining(
    trigger=RetrainingTrigger.MANUAL,
    urgency_score=80.0,
    metrics_before=PerformanceMetrics(
        win_rate=0.55,
        sharpe_ratio=1.5,
        sortino_ratio=1.8,
        max_drawdown=0.15,
        calmar_ratio=10.0,
        n_trades=1000
    ),
    reason="Bitcoin halving - major market regime shift"
)

print(f"Retraining event ID: {event_id}")
```

---

## PART 6: TESTING

### Unit Tests

```python
# test_continuous_learning_integration.py

def test_cl_initialization():
    """Test CL manager initializes correctly"""
    ensemble = EnsembleManager()
    assert ensemble.cl_manager is not None
    assert ensemble.cl_enabled == True

def test_record_trade_outcome():
    """Test recording trade outcome"""
    ensemble = EnsembleManager()
    ensemble.cl_manager.initialize_baseline(
        win_rate=0.58,
        feature_importance={'rsi': 0.3, 'macd': 0.2}
    )
    
    cl_status = ensemble.record_trade_outcome_for_cl(
        symbol='BTCUSDT',
        outcome=1.0,
        features={'rsi': 65.0, 'macd': -0.02},
        pnl=100.0
    )
    
    assert cl_status is not None
    assert 'should_retrain' in cl_status

def test_retraining_trigger():
    """Test retraining trigger logic"""
    # Simulate performance decay
    for i in range(100):
        outcome = 0.0 if i < 60 else 1.0  # 40% WR
        cl_status = ensemble.record_trade_outcome_for_cl(
            symbol='BTCUSDT',
            outcome=outcome,
            features={'rsi': 50.0},
            pnl=-50.0 if outcome == 0.0 else 100.0
        )
    
    # Should trigger retraining
    assert cl_status['should_retrain'] == True
    assert RetrainingTrigger.EWMA_DECAY.value in cl_status['triggers']
```

---

## PART 7: MONITORING DASHBOARD

```python
# scripts/continuous_learning_dashboard.py

import requests
import time
from datetime import datetime

def show_cl_dashboard():
    """Display continuous learning monitoring dashboard"""
    
    while True:
        # Get status
        status = requests.get('http://localhost:8000/continuous-learning/status').json()
        performance = requests.get('http://localhost:8000/continuous-learning/performance').json()
        history = requests.get('http://localhost:8000/continuous-learning/history?limit=5').json()
        
        # Clear screen
        print('\033[2J\033[H')
        
        # Header
        print("=" * 80)
        print("CONTINUOUS LEARNING DASHBOARD".center(80))
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(80))
        print("=" * 80)
        
        # Status
        print(f"\n📊 STATUS:")
        print(f"  Version: {status['status']['current_version']}")
        print(f"  Trades: {status['status']['trade_count']}")
        print(f"  Days since retrain: {status['status']['days_since_retrain'] or 'N/A'}")
        print(f"  Online learning: {'✅ ACTIVE' if status['status']['online_learning'] else '❌ DISABLED'}")
        
        # Performance
        print(f"\n📈 PERFORMANCE:")
        baseline = performance['baseline_wr'] * 100
        ewma = performance['ewma_wr'] * 100
        decay = performance['ewma_decay']
        
        decay_indicator = "🟢" if decay < 1.0 else "🟡" if decay < 3.0 else "🔴"
        
        print(f"  Baseline WR: {baseline:.2f}%")
        print(f"  Current EWMA: {ewma:.2f}%")
        print(f"  Decay: {decay:.2f}pp {decay_indicator}")
        print(f"  CUSUM⁺: {performance['cusum_positive']:.2f}")
        print(f"  CUSUM⁻: {performance['cusum_negative']:.2f}")
        
        # Recent events
        print(f"\n📜 RECENT RETRAINING EVENTS:")
        for event in history['events']:
            timestamp = event['timestamp'][:19]
            trigger = event['trigger']
            urgency = event['urgency_score']
            print(f"  {timestamp} | {trigger:20s} | Urgency: {urgency:5.1f}/100")
        
        # Wait
        time.sleep(10)

if __name__ == '__main__':
    show_cl_dashboard()
```

---

## DEPLOYMENT CHECKLIST

- [ ] Add imports to ensemble_manager.py
- [ ] Initialize CL manager in __init__
- [ ] Add record_trade_outcome_for_cl method
- [ ] Add get_cl_status method
- [ ] Add 6 API endpoints in routes/ai.py
- [ ] Configure environment variables in .env
- [ ] Create continuous_learning_dashboard.py
- [ ] Run unit tests
- [ ] Test API endpoints
- [ ] Monitor for 24 hours
- [ ] Verify retraining triggers work
- [ ] Check online learning updates
- [ ] Validate model versioning

---

**Module 6 Section 4: Integration Guide - COMPLETE ✅**

Next: Risk Analysis
