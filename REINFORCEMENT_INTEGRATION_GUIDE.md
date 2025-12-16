# REINFORCEMENT SIGNALS - SYSTEM INTEGRATION GUIDE

## Overview

This guide shows how to integrate ReinforcementSignalManager into Quantum Trader's existing architecture to enable real-time model weight adaptation and confidence calibration.

---

## Files That Must Be Updated

### 1. **backend/services/ai_trading_engine.py**

Add Reinforcement Signal Manager initialization and signal processing:

```python
# At top of file
from backend.services.ai.reinforcement_signal_manager import (
    ReinforcementSignalManager,
    ReinforcementContext,
    ModelType
)

class AITradingEngine:
    def __init__(self, ...):
        # ... existing code ...
        
        # Initialize Memory State Manager (from Module 1)
        self.memory_manager = MemoryStateManager(...)
        
        # [NEW] Initialize Reinforcement Signal Manager
        self.reinforcement_manager = ReinforcementSignalManager(
            learning_rate=0.05,
            discount_factor=0.95,
            initial_exploration_rate=0.20,
            checkpoint_path="/app/data/reinforcement_state.json"
        )
        logger.info("[AI ENGINE] Reinforcement Signal Manager initialized")
    
    def generate_signals(self) -> List[Signal]:
        """Generate AI signals with reinforcement learning"""
        signals = []
        
        # Get memory context (from Module 1)
        memory_context = self.memory_manager.get_memory_context()
        
        # [NEW] Get reinforcement context
        rl_context = self.reinforcement_manager.get_reinforcement_context()
        
        logger.info(
            f"[AI ENGINE] RL Context: "
            f"Weights=[XGB:{rl_context.model_weights.xgboost:.3f}, "
            f"LGB:{rl_context.model_weights.lightgbm:.3f}, "
            f"N-HiTS:{rl_context.model_weights.nhits:.3f}, "
            f"PatchTST:{rl_context.model_weights.patchtst:.3f}], "
            f"ε={rl_context.exploration_rate:.3f}"
        )
        
        # Emergency stop check (from memory)
        if not memory_context.allow_new_entries:
            logger.error("[AI ENGINE] Memory blocked new entries")
            return []
        
        for symbol in self.symbols:
            # Skip blacklisted symbols
            if symbol in memory_context.symbol_blacklist:
                logger.debug(f"[AI ENGINE] Skipping blacklisted symbol: {symbol}")
                continue
            
            # Fetch market data
            ohlcv = self._fetch_market_data(symbol)
            features = self.feature_engineer.compute_features(ohlcv)
            
            # [MODIFIED] Get ensemble prediction with RL weights
            action, confidence, meta = self.ensemble.predict_with_rl(
                symbol=symbol,
                features=features,
                rl_context=rl_context
            )
            
            # Store model votes for reinforcement learning
            model_votes = meta.get('model_votes', {})
            
            # Apply memory adjustments
            memory_adjusted_confidence = confidence * (1.0 + memory_context.confidence_adjustment * 0.5)
            memory_adjusted_confidence = max(0.0, min(1.0, memory_adjusted_confidence))
            
            # Detect regime
            regime = self.regime_detector.detect(ohlcv)
            
            # Update memory regime tracking
            self.memory_manager.update_regime(...)
            
            # Calculate dynamic TP/SL
            tp_pct, sl_pct = self._calculate_dynamic_tpsl(memory_adjusted_confidence, regime)
            
            # Create setup hash (for pattern memory)
            setup_hash = MemoryStateManager.hash_market_setup(...)
            
            signals.append(Signal(
                symbol=symbol,
                action=action,
                confidence=memory_adjusted_confidence,
                original_confidence=confidence,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                regime=regime,
                setup_hash=setup_hash,
                memory_context=memory_context,
                model_votes=model_votes,  # [NEW] Store for RL
                rl_context=rl_context  # [NEW] Store for RL
            ))
        
        # Auto-checkpoint checks
        self.memory_manager.auto_checkpoint_check()
        self.reinforcement_manager.auto_checkpoint_check()
        
        return signals
```

---

### 2. **backend/services/ensemble_manager.py**

Modify ensemble voting to use RL weights:

```python
from backend.services.ai.reinforcement_signal_manager import ReinforcementContext, ModelType

class EnsembleManager:
    def predict_with_rl(
        self,
        symbol: str,
        features: Dict,
        rl_context: ReinforcementContext
    ) -> Tuple[str, float, Dict]:
        """
        Generate ensemble prediction with reinforcement learning
        
        Args:
            symbol: Trading symbol
            features: Market features
            rl_context: Reinforcement learning context
            
        Returns:
            Tuple of (action, confidence, meta)
        """
        # Get individual model predictions
        model_predictions = {}
        
        # XGBoost
        xgb_action, xgb_conf = self.xgboost_model.predict(features)
        model_predictions[ModelType.XGBOOST.value] = {
            'action': xgb_action,
            'confidence': xgb_conf
        }
        
        # LightGBM
        lgb_action, lgb_conf = self.lightgbm_model.predict(features)
        model_predictions[ModelType.LIGHTGBM.value] = {
            'action': lgb_action,
            'confidence': lgb_conf
        }
        
        # N-HiTS
        nhits_action, nhits_conf = self.nhits_model.predict(features)
        model_predictions[ModelType.NHITS.value] = {
            'action': nhits_action,
            'confidence': nhits_conf
        }
        
        # PatchTST
        patchtst_action, patchtst_conf = self.patchtst_model.predict(features)
        model_predictions[ModelType.PATCHTST.value] = {
            'action': patchtst_action,
            'confidence': patchtst_conf
        }
        
        # [NEW] Apply RL weights (explore or exploit)
        from backend.services.ai.reinforcement_signal_manager import ReinforcementSignalManager
        rl_weights, is_exploring = self.ai_engine.reinforcement_manager.apply_reinforcement_to_signal(
            model_predictions=model_predictions,
            use_exploration=True
        )
        
        # [NEW] Apply confidence scaling from calibration
        scaled_predictions = {}
        for model_str, pred in model_predictions.items():
            scaler = rl_context.confidence_scalers.get(model_str, 1.0)
            scaled_conf = pred['confidence'] * scaler
            scaled_conf = max(0.0, min(1.0, scaled_conf))  # Clip to [0, 1]
            
            scaled_predictions[model_str] = {
                'action': pred['action'],
                'confidence': scaled_conf,
                'original_confidence': pred['confidence'],
                'scaler': scaler
            }
        
        # Weighted voting
        long_score = 0.0
        short_score = 0.0
        hold_score = 0.0
        
        for model_str, pred in scaled_predictions.items():
            weight = rl_weights.get(model_str, 0.25)
            conf = pred['confidence']
            
            if pred['action'] == 'LONG':
                long_score += weight * conf
            elif pred['action'] == 'SHORT':
                short_score += weight * conf
            else:
                hold_score += weight * conf
        
        # Determine final action
        if long_score > short_score and long_score > hold_score:
            final_action = 'LONG'
            final_confidence = long_score
        elif short_score > long_score and short_score > hold_score:
            final_action = 'SHORT'
            final_confidence = short_score
        else:
            final_action = 'HOLD'
            final_confidence = hold_score
        
        # Metadata
        meta = {
            'model_votes': model_predictions,  # Original votes
            'scaled_predictions': scaled_predictions,  # After calibration
            'rl_weights': rl_weights,
            'is_exploring': is_exploring,
            'long_score': long_score,
            'short_score': short_score,
            'hold_score': hold_score
        }
        
        logger.debug(
            f"[ENSEMBLE] {symbol}: {final_action} @ {final_confidence:.3f} "
            f"(Explore={is_exploring}, L={long_score:.3f}, S={short_score:.3f})"
        )
        
        return final_action, final_confidence, meta
```

---

### 3. **backend/services/event_driven_executor.py**

Integrate reinforcement feedback loop when trades close:

```python
from backend.services.ai.reinforcement_signal_manager import ReinforcementSignal

class EventDrivenExecutor:
    async def _handle_trade_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        side: str,
        quantity: float,
        entry_signal: Signal,
        entry_time: datetime,
        exit_time: datetime
    ):
        """Handle trade exit with memory + reinforcement feedback"""
        
        # Calculate PnL
        if side == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl_usd = pnl_pct * quantity * entry_price
        
        # Duration
        duration_seconds = (exit_time - entry_time).total_seconds()
        
        # [EXISTING] Update memory state (Module 1)
        self.ai_engine.memory_manager.record_trade_outcome(
            symbol=symbol,
            action=entry_signal.action,
            confidence=entry_signal.confidence,
            pnl=pnl_usd,
            regime=entry_signal.regime,
            setup_hash=entry_signal.setup_hash
        )
        
        # [NEW] Process reinforcement signal (Module 2)
        rl_signal = self.ai_engine.reinforcement_manager.process_trade_outcome(
            symbol=symbol,
            action=entry_signal.action,
            confidence=entry_signal.original_confidence,  # Use original (before memory adjustment)
            pnl=pnl_usd,
            position_size=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            duration_seconds=duration_seconds,
            regime=entry_signal.regime,
            model_votes=entry_signal.model_votes,
            setup_hash=entry_signal.setup_hash
        )
        
        logger.info(
            f"[EXECUTOR] Trade closed: {symbol} {side} "
            f"PnL=${pnl_usd:.2f} ({pnl_pct:.2%}), "
            f"R_shaped={rl_signal.shaped_reward:.3f}, "
            f"Advantage={rl_signal.advantage:+.3f}"
        )
        
        # Log model contributions
        logger.debug(f"[EXECUTOR] Model contributions: {rl_signal.model_contributions}")
        
        # Existing trade logging code...
```

---

### 4. **backend/services/ai_hedgefund_os.py**

Monitor Reinforcement as subsystem:

```python
from backend.services.ai.reinforcement_signal_manager import ReinforcementSignalManager

class AIHedgeFundOS:
    def _collect_subsystem_states(self) -> Dict[str, SubsystemState]:
        """Collect state from all subsystems including Reinforcement"""
        states = {}
        
        # ... existing subsystems (AI Engine, Memory State, etc.) ...
        
        # [NEW] Reinforcement Learning subsystem
        if hasattr(self, 'reinforcement_manager'):
            rl_diag = self.reinforcement_manager.get_diagnostics()
            
            # Calculate health score
            health_score = 100
            
            # Check if weights are stable
            weight_variance = np.var([
                rl_diag['model_weights']['xgboost'],
                rl_diag['model_weights']['lightgbm'],
                rl_diag['model_weights']['nhits'],
                rl_diag['model_weights']['patchtst']
            ])
            
            if weight_variance > 0.05:  # High variance = unstable
                health_score = min(health_score, 60)
            
            # Check Brier scores (calibration)
            avg_brier = np.mean(list(rl_diag['calibration_metrics']['brier_scores'].values()))
            if avg_brier > 0.30:
                health_score = min(health_score, 70)
            
            # Check recent performance
            if rl_diag['recent_performance']['avg_shaped_reward'] < -0.5:
                health_score = min(health_score, 50)
            
            # Issues
            issues = []
            if weight_variance > 0.05:
                issues.append(f"Weight instability (variance={weight_variance:.4f})")
            
            if avg_brier > 0.30:
                issues.append(f"Poor calibration (avg Brier={avg_brier:.3f})")
            
            if rl_diag['exploration_rate'] > 0.15:
                issues.append(f"High exploration rate (ε={rl_diag['exploration_rate']:.2%})")
            
            states['reinforcement_learning'] = SubsystemState(
                name="Reinforcement Learning",
                status=SubsystemStatus.ACTIVE,
                health_score=health_score,
                last_updated=datetime.now(timezone.utc).isoformat(),
                metrics={
                    'total_trades': rl_diag['total_trades_processed'],
                    'weight_updates': rl_diag['weight_update_count'],
                    'exploration_rate': rl_diag['exploration_rate'],
                    'avg_brier_score': avg_brier,
                    'baseline_reward': rl_diag['baseline_reward']
                },
                issues=issues,
                performance_score=health_score
            )
        
        return states
```

---

### 5. **backend/main.py (Lifespan Integration)**

Initialize Reinforcement Manager in app lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # ... existing startup code ...
    
    # [NEW] Initialize Reinforcement Signal Manager
    if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
        logger.info("[STARTUP] Initializing Reinforcement Signal Manager...")
        
        # Reinforcement manager is already created in AITradingEngine.__init__
        # Just verify it's working
        rl_diag = app.state.ai_engine.reinforcement_manager.get_diagnostics()
        logger.info(
            f"[STARTUP] Reinforcement ready: "
            f"{rl_diag['total_trades_processed']} trades processed, "
            f"exploration_rate={rl_diag['exploration_rate']:.2%}"
        )
    
    yield
    
    # [SHUTDOWN] Save reinforcement state
    if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
        logger.info("[SHUTDOWN] Saving reinforcement state...")
        app.state.ai_engine.reinforcement_manager.checkpoint()
```

---

### 6. **backend/routes/ai.py (New API Endpoints)**

Add endpoints to query reinforcement diagnostics:

```python
from backend.services.ai.reinforcement_signal_manager import ReinforcementContext

@router.get("/reinforcement/diagnostics")
async def get_reinforcement_diagnostics(request: Request):
    """Get comprehensive reinforcement learning diagnostics"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'reinforcement_manager'):
        raise HTTPException(status_code=503, detail="Reinforcement manager not available")
    
    diagnostics = ai_engine.reinforcement_manager.get_diagnostics()
    
    return {
        "status": "success",
        "diagnostics": diagnostics
    }


@router.get("/reinforcement/weights")
async def get_model_weights(request: Request):
    """Get current model weights"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'reinforcement_manager'):
        raise HTTPException(status_code=503, detail="Reinforcement manager not available")
    
    rl_context = ai_engine.reinforcement_manager.get_reinforcement_context()
    
    return {
        "status": "success",
        "weights": rl_context.model_weights.to_dict(),
        "confidence_scalers": rl_context.confidence_scalers,
        "exploration_rate": rl_context.exploration_rate
    }


@router.get("/reinforcement/calibration")
async def get_calibration_metrics(request: Request):
    """Get confidence calibration metrics"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'reinforcement_manager'):
        raise HTTPException(status_code=503, detail="Reinforcement manager not available")
    
    diag = ai_engine.reinforcement_manager.get_diagnostics()
    
    return {
        "status": "success",
        "calibration": diag['calibration_metrics']
    }


@router.post("/reinforcement/reset_weights")
async def reset_model_weights(request: Request):
    """Emergency reset of model weights (use with caution)"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'reinforcement_manager'):
        raise HTTPException(status_code=503, detail="Reinforcement manager not available")
    
    ai_engine.reinforcement_manager.reset_weights()
    
    return {
        "status": "success",
        "message": "Model weights reset to initial values"
    }


@router.post("/reinforcement/reset_calibration")
async def reset_calibration(request: Request):
    """Emergency reset of calibration metrics"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'reinforcement_manager'):
        raise HTTPException(status_code=503, detail="Reinforcement manager not available")
    
    ai_engine.reinforcement_manager.reset_calibration()
    
    return {
        "status": "success",
        "message": "Calibration metrics reset"
    }
```

---

## Integration Flow Diagram

```
Market Data (10s tick)
  ↓
AITradingEngine.generate_signals()
  ↓
  ├─→ reinforcement_manager.get_reinforcement_context()
  │   ↓
  │   ReinforcementContext (weights, scalers, exploration_rate)
  │   ↓
  ├─→ EnsembleManager.predict_with_rl(rl_context)
  │   ↓
  │   ├─→ Get individual model predictions
  │   ├─→ Apply RL weights (explore or exploit)
  │   ├─→ Apply confidence scalers (calibration)
  │   ├─→ Weighted voting
  │   ↓
  │   (action, confidence, model_votes)
  ↓
Signals → OrchestratorPolicy (with memory context)
  ↓
Adjusted Policy → EventDrivenExecutor
  ↓
Execute Trade
  ↓
  [Trade runs...]
  ↓
Trade Exits
  ↓
  ├─→ memory_manager.record_trade_outcome() [Module 1]
  ├─→ reinforcement_manager.process_trade_outcome() [Module 2]
  │   ↓
  │   ├─→ Shape reward (PnL + Sharpe + Risk-adjusted)
  │   ├─→ Identify model contributions
  │   ├─→ Update model weights (exponential update)
  │   ├─→ Update confidence calibration (Brier score)
  │   ├─→ Update baseline reward
  │   ├─→ Auto-checkpoint
  │   ↓
  │   ReinforcementSignal (shaped_reward, advantage, contributions)
  ↓
Updated Weights + Calibration → Next iteration uses new parameters
```

---

## Configuration

Add to `.env`:

```bash
# Reinforcement Learning Configuration
RL_LEARNING_RATE=0.05                   # 0.01-0.10, weight update speed
RL_DISCOUNT_FACTOR=0.95                 # 0.90-0.99, temporal decay
RL_INITIAL_EXPLORATION_RATE=0.20        # 0.10-0.30, starting ε
RL_MIN_EXPLORATION_RATE=0.05            # 0.01-0.10, floor ε
RL_EXPLORATION_DECAY_TRADES=100         # Trades to decay ε
RL_REWARD_ALPHA=0.6                     # Direct PnL weight
RL_REWARD_BETA=0.3                      # Sharpe weight
RL_REWARD_GAMMA=0.1                     # Risk-adjusted weight
RL_CALIBRATION_KAPPA=0.5                # Calibration adjustment weight
RL_MIN_MODEL_WEIGHT=0.05                # Minimum model weight
RL_MAX_MODEL_WEIGHT=0.50                # Maximum model weight
RL_CHECKPOINT_PATH=/app/data/reinforcement_state.json
RL_CHECKPOINT_INTERVAL=60               # Seconds between checkpoints
```

---

## Testing Integration

Run integration test:

```python
# test_reinforcement_integration.py
import asyncio
from backend.services.ai_trading_engine import AITradingEngine

async def test_reinforcement_integration():
    # Initialize engine (creates reinforcement manager)
    engine = AITradingEngine()
    
    # Generate signals (should use RL weights)
    signals = await engine.generate_signals()
    
    # Verify RL context was used
    rl_context = engine.reinforcement_manager.get_reinforcement_context()
    print(f"RL Weights: {rl_context.model_weights.to_dict()}")
    print(f"Exploration Rate: {rl_context.exploration_rate:.2%}")
    
    # Simulate trade outcome
    if signals:
        signal = signals[0]
        
        # Simulate profitable trade
        rl_signal = engine.reinforcement_manager.process_trade_outcome(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.original_confidence,
            pnl=45.50,  # $45.50 profit
            position_size=0.1,
            entry_price=50000.0,
            exit_price=50900.0,
            duration_seconds=300,
            regime=signal.regime,
            model_votes=signal.model_votes,
            setup_hash=signal.setup_hash
        )
        
        print(f"\nReinforcement Signal:")
        print(f"  Raw Reward: ${rl_signal.raw_reward:.2f}")
        print(f"  Shaped Reward: {rl_signal.shaped_reward:.3f}")
        print(f"  Advantage: {rl_signal.advantage:+.3f}")
        print(f"  Model Contributions: {rl_signal.model_contributions}")
    
    # Get updated diagnostics
    diag = engine.reinforcement_manager.get_diagnostics()
    print(f"\nTotal trades processed: {diag['total_trades_processed']}")
    print(f"Model weights: {diag['model_weights']}")
    
    # Checkpoint
    engine.reinforcement_manager.checkpoint()
    print("Checkpoint saved")

if __name__ == "__main__":
    asyncio.run(test_reinforcement_integration())
```

Run:
```bash
docker exec quantum_backend python test_reinforcement_integration.py
```

Expected output:
```
RL Weights: {'xgboost': 0.25, 'lightgbm': 0.25, 'nhits': 0.30, 'patchtst': 0.20}
Exploration Rate: 20.00%

Reinforcement Signal:
  Raw Reward: $45.50
  Shaped Reward: 0.823
  Advantage: +0.823
  Model Contributions: {'xgboost': 0.593, 'lightgbm': 0.560, 'nhits': -0.452, 'patchtst': 0.535}

Total trades processed: 1
Model weights: {'xgboost': 0.256, 'lightgbm': 0.255, 'nhits': 0.296, 'patchtst': 0.205}
Checkpoint saved
```

Notice: N-HiTS weight decreased (voted wrong), others increased.

---

## Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/services/ai/reinforcement_signal_manager.py` | **NEW FILE** | Core reinforcement learning implementation |
| `backend/services/ai_trading_engine.py` | **MODIFIED** | Add reinforcement manager init, context retrieval |
| `backend/services/ensemble_manager.py` | **MODIFIED** | Add `predict_with_rl()` method for adaptive voting |
| `backend/services/event_driven_executor.py` | **MODIFIED** | Add reinforcement feedback on trade exit |
| `backend/services/ai_hedgefund_os.py` | **MODIFIED** | Monitor reinforcement as subsystem |
| `backend/main.py` | **MODIFIED** | Initialize and checkpoint reinforcement in lifespan |
| `backend/routes/ai.py` | **MODIFIED** | Add reinforcement diagnostic endpoints |
| `.env` | **MODIFIED** | Add reinforcement configuration variables |

---

## Verification Checklist

- [ ] Reinforcement manager initializes without errors
- [ ] RL context generated every 10s tick
- [ ] Model weights applied to ensemble voting
- [ ] Confidence scalers adjust model predictions
- [ ] Exploration-exploitation balance working (20% → 5% over 100 trades)
- [ ] Trade outcomes processed correctly
- [ ] Model weights update after each trade
- [ ] Brier scores calculated for calibration
- [ ] Baseline reward tracking working
- [ ] Checkpoint saves and restores state
- [ ] API endpoints return valid data
- [ ] Weight updates are stable (no oscillation)
- [ ] Calibration improves over time
- [ ] AI-HFOS sees reinforcement subsystem health
