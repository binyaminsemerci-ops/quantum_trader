# SHADOW MODELS: INTEGRATION GUIDE

**Module 5: Shadow Models - Section 4**

## Overview

This guide explains how to integrate the Shadow Model Manager into the existing AI trading system, enabling zero-risk parallel testing of challenger models with automatic promotion.

---

## FILE MODIFICATIONS

### 1. backend/services/ai/ai_trading_engine.py

**Purpose:** Integrate shadow model predictions alongside champion predictions

**Modifications:**

```python
# Add import
from backend.services.ai.shadow_model_manager import (
    ShadowModelManager, ModelRole, ModelMetadata, PromotionStatus
)

class AITradingEngine:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Initialize Shadow Model Manager
        self.shadow_manager = ShadowModelManager(
            min_trades_for_promotion=int(os.getenv('SHADOW_MIN_TRADES', 500)),
            mdd_tolerance=float(os.getenv('SHADOW_MDD_TOLERANCE', 1.20)),
            alpha=float(os.getenv('SHADOW_ALPHA', 0.05)),
            n_bootstrap=int(os.getenv('SHADOW_N_BOOTSTRAP', 10000)),
            checkpoint_path='data/shadow_models_checkpoint.json'
        )
        
        # Register champion model
        if self.ensemble_manager.model_weights:
            champion_name = max(self.ensemble_manager.model_weights, 
                               key=self.ensemble_manager.model_weights.get)
            self.shadow_manager.register_model(
                model_name=champion_name,
                model_type='ensemble',
                version='1.0',
                role=ModelRole.CHAMPION,
                description='Current production ensemble'
            )
        
        # Track trades for shadow testing
        self.trades_since_shadow_check = 0
        self.shadow_check_interval = int(os.getenv('SHADOW_CHECK_INTERVAL', 100))
    
    def generate_signal(self, symbol: str, features: Dict) -> Dict:
        """Generate trading signal (modified to include shadow predictions)"""
        
        # Get champion prediction (existing logic)
        champion_name = self.shadow_manager.get_champion()
        champion_prediction = self.ensemble_manager.predict(features)
        
        # Get challenger predictions (shadow mode)
        challengers = self.shadow_manager.get_challengers()
        shadow_predictions = {}
        
        for challenger_name in challengers:
            # Get challenger model
            challenger_model = self._get_model_by_name(challenger_name)
            if challenger_model:
                # Generate shadow prediction
                shadow_pred = challenger_model.predict(features)
                shadow_predictions[challenger_name] = shadow_pred
        
        # Return champion prediction for execution
        return {
            'signal': champion_prediction['signal'],
            'confidence': champion_prediction['confidence'],
            'champion_model': champion_name,
            'shadow_predictions': shadow_predictions  # For tracking only
        }
    
    def record_trade_outcome(self, symbol: str, prediction: Dict, actual_outcome: int, pnl: float):
        """Record trade outcome for all models (champion + shadows)"""
        
        # Record champion outcome
        champion_name = prediction['champion_model']
        self.shadow_manager.record_prediction(
            model_name=champion_name,
            prediction=prediction['signal'],
            actual_outcome=actual_outcome,
            pnl=pnl,
            confidence=prediction['confidence'],
            executed=True  # Champion was actually executed
        )
        
        # Record shadow outcomes
        for shadow_name, shadow_pred in prediction.get('shadow_predictions', {}).items():
            self.shadow_manager.record_prediction(
                model_name=shadow_name,
                prediction=shadow_pred['signal'],
                actual_outcome=actual_outcome,
                pnl=pnl,  # Same outcome, different prediction
                confidence=shadow_pred['confidence'],
                executed=False  # Shadow was NOT executed
            )
        
        # Increment counter
        self.trades_since_shadow_check += 1
        
        # Periodic shadow testing (every 100 trades)
        if self.trades_since_shadow_check >= self.shadow_check_interval:
            self._check_shadow_promotions()
            self.trades_since_shadow_check = 0
    
    def _check_shadow_promotions(self):
        """Check if any challenger is ready for promotion"""
        challengers = self.shadow_manager.get_challengers()
        
        for challenger_name in challengers:
            trade_count = self.shadow_manager.get_trade_count(challenger_name)
            
            # Check if minimum trades reached
            if trade_count < 500:
                logger.info(f"[Shadow] {challenger_name}: {trade_count}/500 trades")
                continue
            
            # Run promotion check
            decision = self.shadow_manager.check_promotion_criteria(challenger_name)
            
            if decision is None:
                continue
            
            logger.info(
                f"[Shadow] {challenger_name} promotion check: "
                f"Status={decision.status.value}, Score={decision.promotion_score:.1f}/100"
            )
            
            # Auto-promote if approved
            if decision.status == PromotionStatus.APPROVED:
                logger.info(f"[Shadow] Auto-promoting {challenger_name} → Champion")
                
                success = self.shadow_manager.promote_challenger(challenger_name)
                
                if success:
                    # Update ensemble weights to reflect new champion
                    self._update_ensemble_after_promotion(challenger_name)
                    
                    # Alert team
                    self._send_promotion_alert(challenger_name, decision)
            
            elif decision.status == PromotionStatus.PENDING:
                # Manual review needed
                logger.warning(
                    f"[Shadow] {challenger_name} needs manual review: {decision.reason}"
                )
                self._send_manual_review_alert(challenger_name, decision)
            
            else:
                # Rejected
                logger.info(
                    f"[Shadow] {challenger_name} rejected: {decision.reason}"
                )
    
    def _update_ensemble_after_promotion(self, new_champion: str):
        """Update ensemble weights after promotion"""
        # Set new champion to 100% weight in ensemble
        for model_name in self.ensemble_manager.model_weights:
            if model_name == new_champion:
                self.ensemble_manager.model_weights[model_name] = 1.0
            else:
                self.ensemble_manager.model_weights[model_name] = 0.0
        
        logger.info(f"Ensemble weights updated: {new_champion} = 100%")
    
    def _send_promotion_alert(self, challenger_name: str, decision: PromotionDecision):
        """Send alert when challenger promoted"""
        improvement = decision.challenger_metrics['win_rate'] - decision.champion_metrics['win_rate']
        
        message = (
            f"🎉 SHADOW MODEL PROMOTION\n"
            f"Model: {challenger_name}\n"
            f"Score: {decision.promotion_score:.1f}/100\n"
            f"WR Improvement: +{improvement:.2%}\n"
            f"Sharpe: {decision.challenger_metrics['sharpe_ratio']:.2f}\n"
            f"Reason: {decision.reason}"
        )
        
        # Send via notification system
        self.notification_service.send_alert(message, priority='HIGH')
    
    def _send_manual_review_alert(self, challenger_name: str, decision: PromotionDecision):
        """Send alert when manual review needed"""
        message = (
            f"⚠️ SHADOW MODEL NEEDS REVIEW\n"
            f"Model: {challenger_name}\n"
            f"Score: {decision.promotion_score:.1f}/100\n"
            f"Reason: {decision.reason}\n"
            f"Action: Review dashboard and approve/reject manually"
        )
        
        self.notification_service.send_alert(message, priority='MEDIUM')
    
    def _get_model_by_name(self, model_name: str):
        """Get model instance by name"""
        # Map model names to actual model instances
        model_map = {
            'xgboost': self.ensemble_manager.xgboost_model,
            'lightgbm': self.ensemble_manager.lightgbm_model,
            'catboost': self.ensemble_manager.catboost_model,
            'neural_network': self.ensemble_manager.nn_model
        }
        
        for key, model in model_map.items():
            if key in model_name.lower():
                return model
        
        return None
    
    def deploy_challenger_model(self, model_name: str, model_type: str, model_instance, description: str = ""):
        """Deploy a new challenger model for shadow testing"""
        
        # Register with shadow manager
        self.shadow_manager.register_model(
            model_name=model_name,
            model_type=model_type,
            version='1.0',
            role=ModelRole.CHALLENGER,
            description=description
        )
        
        # Add to model map (for predictions)
        self.challenger_models[model_name] = model_instance
        
        logger.info(f"[Shadow] Deployed challenger: {model_name} ({model_type})")
        
        return {
            'status': 'deployed',
            'model_name': model_name,
            'role': 'challenger',
            'allocation': '0% (shadow mode)'
        }
```

---

### 2. backend/services/ai/ensemble_manager.py

**Purpose:** Add shadow model support to ensemble predictions

**Modifications:**

```python
class EnsembleManager:
    def __init__(self, ...):
        # ... existing initialization ...
        
        # Track shadow models
        self.shadow_models: Dict[str, Any] = {}  # challenger_name → model_instance
    
    def register_shadow_model(self, model_name: str, model_instance):
        """Register a shadow model for parallel testing"""
        self.shadow_models[model_name] = model_instance
        logger.info(f"[Ensemble] Registered shadow model: {model_name}")
    
    def predict_with_shadows(self, features: Dict) -> Dict:
        """Generate predictions for champion + all shadows"""
        
        # Champion prediction (existing logic)
        champion_pred = self.predict(features)
        
        # Shadow predictions
        shadow_preds = {}
        for shadow_name, shadow_model in self.shadow_models.items():
            try:
                shadow_pred = shadow_model.predict(features)
                shadow_preds[shadow_name] = shadow_pred
            except Exception as e:
                logger.error(f"[Shadow] Prediction failed for {shadow_name}: {e}")
        
        return {
            'champion': champion_pred,
            'shadows': shadow_preds
        }
    
    def remove_shadow_model(self, model_name: str):
        """Remove a shadow model (after promotion or rejection)"""
        if model_name in self.shadow_models:
            del self.shadow_models[model_name]
            logger.info(f"[Ensemble] Removed shadow model: {model_name}")
```

---

### 3. backend/routes/ai.py

**Purpose:** Add API endpoints for shadow model management

**Modifications:**

```python
from backend.services.ai.shadow_model_manager import PromotionStatus

# ============================================================================
# SHADOW MODEL ENDPOINTS
# ============================================================================

@router.get("/shadow/status")
async def get_shadow_status():
    """Get status of all shadow models"""
    try:
        shadow_manager = ai_trading_engine.shadow_manager
        
        champion = shadow_manager.get_champion()
        challengers = shadow_manager.get_challengers()
        
        status = {
            'champion': {
                'model_name': champion,
                'metrics': shadow_manager.get_metrics(champion).to_dict() if champion else None,
                'trade_count': shadow_manager.get_trade_count(champion)
            },
            'challengers': []
        }
        
        for challenger in challengers:
            metrics = shadow_manager.get_metrics(challenger)
            decision = shadow_manager.get_pending_decision(challenger)
            
            challenger_info = {
                'model_name': challenger,
                'metrics': metrics.to_dict() if metrics else None,
                'trade_count': shadow_manager.get_trade_count(challenger),
                'promotion_status': decision.status.value if decision else 'pending',
                'promotion_score': decision.promotion_score if decision else 0,
                'reason': decision.reason if decision else ''
            }
            
            status['challengers'].append(challenger_info)
        
        return {
            'status': 'success',
            'data': status
        }
    
    except Exception as e:
        logger.error(f"Failed to get shadow status: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/comparison/{challenger_name}")
async def get_shadow_comparison(challenger_name: str):
    """Get detailed comparison between champion and challenger"""
    try:
        shadow_manager = ai_trading_engine.shadow_manager
        
        champion = shadow_manager.get_champion()
        
        champion_metrics = shadow_manager.get_metrics(champion)
        challenger_metrics = shadow_manager.get_metrics(challenger_name)
        
        if not champion_metrics or not challenger_metrics:
            return {'status': 'error', 'message': 'Insufficient data'}
        
        # Get test results
        test_results_history = shadow_manager.get_test_results_history(challenger_name, n=1)
        latest_test = test_results_history[0] if test_results_history else None
        
        comparison = {
            'champion': {
                'model_name': champion,
                'win_rate': champion_metrics.win_rate,
                'sharpe_ratio': champion_metrics.sharpe_ratio,
                'mean_pnl': champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown,
                'total_pnl': champion_metrics.total_pnl,
                'n_trades': champion_metrics.n_trades
            },
            'challenger': {
                'model_name': challenger_name,
                'win_rate': challenger_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl,
                'max_drawdown': challenger_metrics.max_drawdown,
                'total_pnl': challenger_metrics.total_pnl,
                'n_trades': challenger_metrics.n_trades
            },
            'difference': {
                'win_rate': challenger_metrics.win_rate - champion_metrics.win_rate,
                'sharpe_ratio': challenger_metrics.sharpe_ratio - champion_metrics.sharpe_ratio,
                'mean_pnl': challenger_metrics.mean_pnl - champion_metrics.mean_pnl,
                'max_drawdown': champion_metrics.max_drawdown - challenger_metrics.max_drawdown  # Lower is better
            },
            'statistical_tests': latest_test.to_dict() if latest_test else None
        }
        
        return {
            'status': 'success',
            'data': comparison
        }
    
    except Exception as e:
        logger.error(f"Failed to get comparison: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/deploy")
async def deploy_shadow_model(request: dict):
    """Deploy a new challenger model for shadow testing"""
    try:
        model_name = request.get('model_name')
        model_type = request.get('model_type')
        description = request.get('description', '')
        
        if not model_name or not model_type:
            return {'status': 'error', 'message': 'model_name and model_type required'}
        
        # Load model from disk or training pipeline
        model_instance = ai_trading_engine._load_model(model_name, model_type)
        
        # Deploy as challenger
        result = ai_trading_engine.deploy_challenger_model(
            model_name=model_name,
            model_type=model_type,
            model_instance=model_instance,
            description=description
        )
        
        return {
            'status': 'success',
            'data': result
        }
    
    except Exception as e:
        logger.error(f"Failed to deploy shadow model: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/promote/{challenger_name}")
async def promote_shadow_model(challenger_name: str, force: bool = False):
    """Manually promote a challenger to champion"""
    try:
        shadow_manager = ai_trading_engine.shadow_manager
        
        success = shadow_manager.promote_challenger(challenger_name, force=force)
        
        if success:
            # Update ensemble
            ai_trading_engine._update_ensemble_after_promotion(challenger_name)
            
            return {
                'status': 'success',
                'message': f'{challenger_name} promoted to champion'
            }
        else:
            return {
                'status': 'error',
                'message': 'Promotion failed (check criteria)'
            }
    
    except Exception as e:
        logger.error(f"Failed to promote: {e}")
        return {'status': 'error', 'message': str(e)}


@router.post("/shadow/rollback")
async def rollback_champion(reason: str = "Manual rollback"):
    """Rollback to previous champion"""
    try:
        shadow_manager = ai_trading_engine.shadow_manager
        
        success = shadow_manager.rollback_to_previous_champion(reason=reason)
        
        if success:
            # Update ensemble
            champion = shadow_manager.get_champion()
            ai_trading_engine._update_ensemble_after_promotion(champion)
            
            return {
                'status': 'success',
                'message': f'Rolled back to {champion}'
            }
        else:
            return {
                'status': 'error',
                'message': 'Rollback failed (no history)'
            }
    
    except Exception as e:
        logger.error(f"Failed to rollback: {e}")
        return {'status': 'error', 'message': str(e)}


@router.get("/shadow/history")
async def get_promotion_history(n: int = 10):
    """Get promotion history"""
    try:
        shadow_manager = ai_trading_engine.shadow_manager
        
        history = shadow_manager.get_promotion_history(n=n)
        
        return {
            'status': 'success',
            'data': [event.to_dict() for event in history]
        }
    
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {'status': 'error', 'message': str(e)}
```

---

## CONFIGURATION (.env)

Add the following environment variables:

```bash
# Shadow Model Configuration
SHADOW_MIN_TRADES=500                  # Minimum trades before promotion check
SHADOW_MDD_TOLERANCE=1.20             # Max drawdown tolerance (1.20 = 20% worse allowed)
SHADOW_ALPHA=0.05                     # Significance level (5%)
SHADOW_N_BOOTSTRAP=10000              # Bootstrap iterations
SHADOW_CHECK_INTERVAL=100             # Check every N trades
```

---

## INTEGRATION FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                    SHADOW MODEL INTEGRATION                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INITIALIZATION (System Startup)                            │
│     ┌──────────────────────────────────────┐                   │
│     │ AITradingEngine.__init__()           │                   │
│     │  └─ Initialize ShadowModelManager    │                   │
│     │  └─ Register champion model          │                   │
│     │  └─ Load checkpoint (if exists)      │                   │
│     └──────────────────────────────────────┘                   │
│                        │                                        │
│  2. SIGNAL GENERATION (Per Trade)                              │
│     ┌──────────────────────────────────────┐                   │
│     │ generate_signal(symbol, features)    │                   │
│     │  └─ Champion prediction (executed)   │                   │
│     │  └─ Challenger predictions (shadow)  │                   │
│     │  └─ Return champion signal only      │                   │
│     └──────────────────────────────────────┘                   │
│                        │                                        │
│  3. OUTCOME RECORDING (After Trade Closes)                     │
│     ┌──────────────────────────────────────┐                   │
│     │ record_trade_outcome(...)            │                   │
│     │  └─ Record champion outcome (PnL)    │                   │
│     │  └─ Record shadow outcomes (PnL)     │                   │
│     │  └─ Increment trade counter          │                   │
│     └──────────────────────────────────────┘                   │
│                        │                                        │
│  4. PERIODIC TESTING (Every 100 Trades)                        │
│     ┌──────────────────────────────────────┐                   │
│     │ _check_shadow_promotions()           │                   │
│     │  └─ For each challenger:             │                   │
│     │      ├─ Check trade count ≥500       │                   │
│     │      ├─ Run statistical tests        │                   │
│     │      ├─ Check promotion criteria     │                   │
│     │      └─ Auto-promote if approved     │                   │
│     └──────────────────────────────────────┘                   │
│                        │                                        │
│  5. PROMOTION (If Criteria Met)                                │
│     ┌──────────────────────────────────────┐                   │
│     │ promote_challenger(challenger_name)  │                   │
│     │  └─ Archive old champion             │                   │
│     │  └─ Promote challenger → champion    │                   │
│     │  └─ Update ensemble weights          │                   │
│     │  └─ Send alert to team               │                   │
│     │  └─ Reset monitoring (100 trades)    │                   │
│     └──────────────────────────────────────┘                   │
│                        │                                        │
│  6. MONITORING (First 100 Trades Post-Promotion)               │
│     ┌──────────────────────────────────────┐                   │
│     │ _check_post_promotion_health()       │                   │
│     │  └─ Check WR vs baseline             │                   │
│     │  └─ Alert if WR drops >3pp           │                   │
│     │  └─ Rollback if WR drops >5pp        │                   │
│     └──────────────────────────────────────┘                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## TESTING SCRIPT

Create `scripts/test_shadow_integration.py`:

```python
"""Test shadow model integration"""

import sys
sys.path.append('.')

from backend.services.ai.ai_trading_engine import AITradingEngine
from backend.services.ai.shadow_model_manager import ModelRole
import numpy as np

def test_shadow_integration():
    """Test complete shadow model workflow"""
    
    print("="*60)
    print("SHADOW MODEL INTEGRATION TEST")
    print("="*60)
    
    # 1. Initialize engine
    print("\n[1] Initializing AITradingEngine...")
    engine = AITradingEngine()
    
    # 2. Verify champion registered
    champion = engine.shadow_manager.get_champion()
    print(f"[2] Champion model: {champion}")
    assert champion is not None, "No champion model found"
    
    # 3. Deploy challenger
    print("\n[3] Deploying challenger model...")
    result = engine.deploy_challenger_model(
        model_name='lightgbm_test',
        model_type='lightgbm',
        model_instance=None,  # Mock model
        description='Test challenger'
    )
    print(f"    Status: {result['status']}")
    print(f"    Allocation: {result['allocation']}")
    
    # 4. Simulate 500 trades
    print("\n[4] Simulating 500 trades...")
    np.random.seed(42)
    
    for i in range(500):
        # Generate signal
        features = {'rsi': 50, 'macd': 0.1, 'volume': 1000}
        signal = engine.generate_signal('BTCUSDT', features)
        
        # Simulate outcome
        outcome = 1 if np.random.rand() < 0.56 else 0
        pnl = np.random.normal(50, 120)
        
        # Record outcome
        engine.record_trade_outcome('BTCUSDT', signal, outcome, pnl)
        
        if (i + 1) % 100 == 0:
            print(f"    Trades: {i+1}/500")
    
    # 5. Check promotion decision
    print("\n[5] Checking promotion criteria...")
    decision = engine.shadow_manager.check_promotion_criteria('lightgbm_test')
    
    if decision:
        print(f"    Status: {decision.status.value}")
        print(f"    Score: {decision.promotion_score:.1f}/100")
        print(f"    Reason: {decision.reason}")
        
        print(f"\n    Champion WR: {decision.champion_metrics['win_rate']:.2%}")
        print(f"    Challenger WR: {decision.challenger_metrics['win_rate']:.2%}")
    
    # 6. Promotion history
    print("\n[6] Promotion history:")
    history = engine.shadow_manager.get_promotion_history(n=5)
    print(f"    Total promotions: {len(history)}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE ✅")
    print("="*60)

if __name__ == '__main__':
    test_shadow_integration()
```

Run with:
```bash
python scripts/test_shadow_integration.py
```

---

## VERIFICATION CHECKLIST

- [ ] **Shadow Manager initialized** in AITradingEngine
- [ ] **Champion registered** on system startup
- [ ] **Shadow predictions generated** for challengers (not executed)
- [ ] **Trade outcomes recorded** for champion + shadows
- [ ] **Periodic testing triggered** every 100 trades
- [ ] **Statistical tests run** (t-test, bootstrap, Sharpe, WR)
- [ ] **Promotion criteria checked** (5 primary + 3 secondary)
- [ ] **Auto-promotion occurs** when score ≥70
- [ ] **Ensemble weights updated** after promotion
- [ ] **Alerts sent** for promotions and manual reviews
- [ ] **Post-promotion monitoring** (first 100 trades)
- [ ] **Rollback works** if new champion degrades >5pp
- [ ] **Checkpoint saves/restores** state correctly
- [ ] **API endpoints functional** (status, comparison, deploy, promote, rollback, history)
- [ ] **Dashboard displays** shadow model status
- [ ] **Logs include** shadow model events
- [ ] **No performance impact** on champion predictions (<5ms overhead)

---

## TROUBLESHOOTING

### Issue 1: Challenger not getting predictions

**Symptom:** Challenger trade count stays at 0

**Diagnosis:**
```python
# Check if challenger registered
challengers = engine.shadow_manager.get_challengers()
print(f"Challengers: {challengers}")

# Check shadow predictions
signal = engine.generate_signal('BTCUSDT', features)
print(f"Shadow predictions: {signal.get('shadow_predictions', {})}")
```

**Solution:**
- Verify challenger registered with `ModelRole.CHALLENGER`
- Check `_get_model_by_name()` returns correct model instance
- Ensure `record_trade_outcome()` called with shadow predictions

### Issue 2: Promotion never triggers

**Symptom:** Challenger has 500+ trades but no promotion check

**Diagnosis:**
```python
# Check trade count
count = engine.shadow_manager.get_trade_count('lightgbm_test')
print(f"Trade count: {count}")

# Check if periodic testing running
print(f"Trades since check: {engine.trades_since_shadow_check}")
```

**Solution:**
- Verify `shadow_check_interval` is 100 (not 1000)
- Check `_check_shadow_promotions()` is called in `record_trade_outcome()`
- Review logs for promotion check messages

### Issue 3: False rejections (good model rejected)

**Symptom:** Challenger WR 58% but promotion rejected

**Diagnosis:**
```python
# Get promotion decision
decision = engine.shadow_manager.check_promotion_criteria('lightgbm_test')
print(f"Status: {decision.status.value}")
print(f"Reason: {decision.reason}")

# Check criteria individually
print(f"Statistical significance: {decision.statistical_significance}")
print(f"Sharpe criterion: {decision.sharpe_criterion}")
print(f"Sample size: {decision.sample_size_criterion}")
```

**Solution:**
- Check if sample size <500 (increase `min_trades`)
- Verify statistical tests: p-value may be 0.06 (just above threshold)
- Review MDD: challenger MDD may exceed champion * 1.20
- Consider lowering `alpha` to 0.10 for less stringent testing

### Issue 4: Post-promotion rollback

**Symptom:** New champion promoted then immediately rolled back

**Diagnosis:**
```python
# Check promotion baseline
baseline = engine.shadow_manager.promotion_baseline_wr
current_metrics = engine.shadow_manager.get_metrics(champion)
print(f"Baseline WR: {baseline:.2%}")
print(f"Current WR: {current_metrics.win_rate:.2%}")
print(f"Drop: {baseline - current_metrics.win_rate:.2%}")
```

**Solution:**
- Short-term variance: 100 trades may not be enough for stable WR
- Increase rollback threshold from 3pp to 5pp
- Extend monitoring window from 100 to 200 trades
- Check if market conditions changed post-promotion

### Issue 5: High computational overhead

**Symptom:** Latency increased from 50ms to 150ms

**Diagnosis:**
```python
import time

# Time champion prediction
start = time.time()
champion_pred = engine.generate_signal('BTCUSDT', features)
champion_time = time.time() - start

print(f"Champion prediction: {champion_time*1000:.1f}ms")

# Time shadow predictions
start = time.time()
shadows = engine.shadow_manager.get_challengers()
for shadow in shadows:
    shadow_pred = engine._get_model_by_name(shadow).predict(features)
shadow_time = time.time() - start

print(f"Shadow predictions ({len(shadows)}): {shadow_time*1000:.1f}ms")
```

**Solution:**
- Limit challengers to max 2 at a time
- Use async/await for shadow predictions (non-blocking)
- Cache shadow model instances (avoid repeated loading)
- Profile slow model predictions, optimize if needed

---

## EXPECTED OUTPUT

### Healthy System (No Promotions Needed)

```
[2025-11-26 10:00:00] [Shadow] xgboost_v1: 100/500 trades
[2025-11-26 10:15:00] [Shadow] xgboost_v1: 200/500 trades
[2025-11-26 10:30:00] [Shadow] xgboost_v1: 300/500 trades
[2025-11-26 10:45:00] [Shadow] xgboost_v1: 400/500 trades
[2025-11-26 11:00:00] [Shadow] xgboost_v1: 500/500 trades
[2025-11-26 11:00:05] [Shadow] xgboost_v1 promotion check: Status=rejected, Score=42.3/100
[2025-11-26 11:00:05] [Shadow] xgboost_v1 rejected: Statistical significance not achieved (p=0.23)
```

### Promotion Event

```
[2025-11-26 15:00:00] [Shadow] lightgbm_v2: 500/500 trades
[2025-11-26 15:00:05] [Shadow] lightgbm_v2 promotion check: Status=approved, Score=78.5/100
[2025-11-26 15:00:05] [Shadow] Auto-promoting lightgbm_v2 → Champion
[2025-11-26 15:00:06] 🎉 PROMOTION: lightgbm_v2 promoted to champion (replacing xgboost_v1) | Score: 78.5/100 | Improvement: WR +2.10%, Sharpe +0.35
[2025-11-26 15:00:06] Ensemble weights updated: lightgbm_v2 = 100%
[2025-11-26 15:00:07] [Alert] HIGH: Shadow model lightgbm_v2 promoted (WR: 58.1%, Sharpe: 2.15)
```

### Post-Promotion Monitoring

```
[2025-11-26 15:01:00] [Shadow] lightgbm_v2: Post-promotion monitoring (10/100 trades)
[2025-11-26 15:02:00] [Shadow] lightgbm_v2: Post-promotion monitoring (20/100 trades)
...
[2025-11-26 15:10:00] [Shadow] lightgbm_v2: Post-promotion monitoring (100/100 trades)
[2025-11-26 15:10:01] [Shadow] lightgbm_v2: Promotion stable (WR: 57.8%, baseline: 58.1%)
```

---

**Module 5 Section 4: Integration Guide - COMPLETE ✅**

Next: Risk Analysis (Section 5)
