# MEMORY STATES - SYSTEM INTEGRATION GUIDE

## Overview

This document describes how to integrate MemoryStateManager into Quantum Trader's existing architecture.

---

## Files That Must Be Updated

### 1. **backend/services/ai_trading_engine.py**

Add Memory State Manager initialization and integration:

```python
# At top of file
from backend.services.ai.memory_state_manager import (
    MemoryStateManager,
    MarketRegime,
    MemoryContext
)

class AITradingEngine:
    def __init__(self, ...):
        # ... existing code ...
        
        # [NEW] Initialize Memory State Manager
        self.memory_manager = MemoryStateManager(
            ewma_alpha=0.3,
            min_samples_for_memory=10,
            checkpoint_path="/app/data/memory_state.json"
        )
        logger.info("[AI ENGINE] Memory State Manager initialized")
    
    def generate_signals(self) -> List[Signal]:
        """Generate AI signals with memory context"""
        signals = []
        
        # [NEW] Get memory context BEFORE generating signals
        memory_context = self.memory_manager.get_memory_context()
        
        logger.info(
            f"[AI ENGINE] Memory Context: "
            f"ConfAdj={memory_context.confidence_adjustment:+.2f}, "
            f"RiskMult={memory_context.risk_multiplier:.2f}, "
            f"AllowEntries={memory_context.allow_new_entries}"
        )
        
        # Emergency stop check
        if not memory_context.allow_new_entries:
            logger.error("[AI ENGINE] Memory blocked new entries - returning empty signals")
            return []
        
        for symbol in self.symbols:
            # Skip blacklisted symbols
            if symbol in memory_context.symbol_blacklist:
                logger.debug(f"[AI ENGINE] Skipping blacklisted symbol: {symbol}")
                continue
            
            # Generate signal (existing code)
            ohlcv = self._fetch_market_data(symbol)
            features = self.feature_engineer.compute_features(ohlcv)
            action, confidence, meta = self.ensemble.predict(symbol, features)
            
            # [NEW] Adjust confidence based on memory
            # Memory can raise/lower threshold indirectly via Orchestrator
            # But we can also adjust confidence directly here
            memory_adjusted_confidence = confidence * (1.0 + memory_context.confidence_adjustment * 0.5)
            memory_adjusted_confidence = max(0.0, min(1.0, memory_adjusted_confidence))
            
            # Detect regime
            regime = self.regime_detector.detect(ohlcv)
            
            # [NEW] Update memory with current regime
            regime_confidence = meta.get('regime_confidence', 0.7)
            market_features = {
                'atr_pct': features.get('atr_pct', 0.02),
                'momentum': features.get('momentum', 0.0),
                'trend_strength': features.get('trend_strength', 0.0)
            }
            self.memory_manager.update_regime(
                new_regime=MarketRegime(regime),
                regime_confidence=regime_confidence,
                market_features=market_features
            )
            
            # Calculate dynamic TP/SL
            tp_pct, sl_pct = self._calculate_dynamic_tpsl(
                memory_adjusted_confidence, 
                regime
            )
            
            # [NEW] Create setup hash for pattern tracking
            vol_bucket = self._get_volatility_bucket(features['atr_pct'])
            mom_bucket = self._get_momentum_bucket(features['momentum'])
            trend_bucket = self._get_trend_bucket(features['trend_strength'])
            
            setup_hash = MemoryStateManager.hash_market_setup(
                symbol=symbol,
                regime=MarketRegime(regime),
                volatility_bucket=vol_bucket,
                momentum_bucket=mom_bucket,
                trend_strength_bucket=trend_bucket
            )
            
            # [NEW] Query pattern memory
            pattern_stats = self.memory_manager.query_pattern_memory(setup_hash)
            if pattern_stats and pattern_stats['sample_count'] >= 10:
                # Use historical pattern data to adjust confidence
                pattern_win_rate = pattern_stats['win_rate']
                if pattern_win_rate < 0.40:
                    memory_adjusted_confidence *= 0.7  # Reduce confidence
                    logger.info(
                        f"[AI ENGINE] Pattern {setup_hash[:8]} has poor history "
                        f"(WR={pattern_win_rate:.1%}) - reducing confidence"
                    )
            
            signals.append(Signal(
                symbol=symbol,
                action=action,
                confidence=memory_adjusted_confidence,
                original_confidence=confidence,
                tp_pct=tp_pct,
                sl_pct=sl_pct,
                regime=regime,
                setup_hash=setup_hash,
                memory_context=memory_context
            ))
        
        # [NEW] Auto-checkpoint check
        self.memory_manager.auto_checkpoint_check()
        
        return signals
    
    def _get_volatility_bucket(self, atr_pct: float) -> str:
        """Bucket volatility for pattern hashing"""
        if atr_pct < 0.02:
            return "LOW"
        elif atr_pct < 0.04:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_momentum_bucket(self, momentum: float) -> str:
        """Bucket momentum for pattern hashing"""
        abs_mom = abs(momentum)
        if abs_mom < 0.02:
            return "LOW"
        elif abs_mom < 0.05:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _get_trend_bucket(self, trend_strength: float) -> str:
        """Bucket trend strength for pattern hashing"""
        if trend_strength < 0.3:
            return "WEAK"
        elif trend_strength < 0.6:
            return "MODERATE"
        else:
            return "STRONG"
```

---

### 2. **backend/services/event_driven_executor.py**

Integrate memory feedback loop when trades close:

```python
from backend.services.ai.memory_state_manager import MarketRegime

class EventDrivenExecutor:
    def __init__(self, ...):
        # ... existing code ...
        
        # Get reference to memory manager from ai_engine
        self.memory_manager = self.ai_engine.memory_manager
    
    async def _handle_trade_exit(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        side: str,
        quantity: float,
        entry_signal: Signal
    ):
        """Handle trade exit and update memory"""
        # Calculate PnL
        if side == "LONG":
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price
        
        pnl_usd = pnl_pct * quantity * entry_price
        
        # [NEW] Record outcome in memory
        self.memory_manager.record_trade_outcome(
            symbol=symbol,
            action=entry_signal.action,
            confidence=entry_signal.confidence,
            pnl=pnl_usd,
            regime=MarketRegime(entry_signal.regime),
            setup_hash=entry_signal.setup_hash
        )
        
        logger.info(
            f"[EXECUTOR] Trade closed: {symbol} {side} "
            f"PnL=${pnl_usd:.2f} ({pnl_pct:.2%}) - Memory updated"
        )
        
        # Existing trade logging code...
```

---

### 3. **backend/services/orchestrator_policy.py**

Use Memory Context to adjust policy parameters:

```python
from backend.services.ai.memory_state_manager import MemoryContext, MemoryLevel

class OrchestratorPolicy:
    def create_policy(
        self,
        memory_context: Optional[MemoryContext] = None
    ) -> Policy:
        """
        Create unified policy with memory integration
        
        Args:
            memory_context: Optional memory context from MemoryStateManager
        """
        # Base policy from regime, risk, cost, etc.
        base_policy = self._create_base_policy()
        
        # [NEW] Apply memory adjustments
        if memory_context:
            # Adjust confidence threshold
            adjusted_threshold = (
                base_policy.min_confidence + 
                memory_context.confidence_adjustment
            )
            adjusted_threshold = max(0.20, min(0.60, adjusted_threshold))
            
            # Adjust risk multiplier
            adjusted_risk = (
                base_policy.max_risk_pct * 
                memory_context.risk_multiplier
            )
            adjusted_risk = max(0.3, min(3.0, adjusted_risk))
            
            # Check allow_new_entries flag
            allow_trades = (
                base_policy.allow_trades and 
                memory_context.allow_new_entries
            )
            
            logger.info(
                f"[ORCHESTRATOR] Memory adjustments applied: "
                f"Threshold {base_policy.min_confidence:.2f} → {adjusted_threshold:.2f}, "
                f"Risk {base_policy.max_risk_pct:.2f} → {adjusted_risk:.2f}, "
                f"Allow={allow_trades}"
            )
            
            return Policy(
                min_confidence=adjusted_threshold,
                max_risk_pct=adjusted_risk,
                allow_trades=allow_trades,
                max_open_positions=base_policy.max_open_positions,
                memory_level=memory_context.memory_level.value,
                memory_performance_score=memory_context.recent_performance_score
            )
        
        return base_policy
```

---

### 4. **backend/services/ai_hedgefund_os.py**

Monitor Memory State as subsystem:

```python
from backend.services.ai.memory_state_manager import MemoryStateManager, MemoryLevel

class AIHedgeFundOS:
    def _collect_subsystem_states(self) -> Dict[str, SubsystemState]:
        """Collect state from all subsystems including Memory"""
        states = {}
        
        # ... existing subsystems ...
        
        # [NEW] Memory State subsystem
        if hasattr(self, 'memory_manager'):
            memory_diag = self.memory_manager.get_diagnostics()
            
            # Calculate health score
            memory_level = MemoryLevel(memory_diag['memory_level'])
            if memory_level == MemoryLevel.HIGH:
                health_score = 100
            elif memory_level == MemoryLevel.MEDIUM:
                health_score = 75
            elif memory_level == MemoryLevel.LOW:
                health_score = 50
            else:
                health_score = 25
            
            # Adjust for consecutive losses
            consecutive_losses = memory_diag['performance_memory']['consecutive_losses']
            if consecutive_losses >= 5:
                health_score = min(health_score, 30)
            elif consecutive_losses >= 3:
                health_score = min(health_score, 60)
            
            # Check for issues
            issues = []
            if consecutive_losses >= 5:
                issues.append(f"5+ consecutive losses detected")
            
            if memory_diag['performance_memory']['recent_pnl_sum'] < -500:
                issues.append(f"Recent PnL: ${memory_diag['performance_memory']['recent_pnl_sum']:.2f}")
            
            brier_score = memory_diag['calibration']['brier_score']
            if brier_score > 0.3:
                issues.append(f"Poor calibration (Brier={brier_score:.3f})")
            
            states['memory_state'] = SubsystemState(
                name="Memory State Manager",
                status=SubsystemStatus.ACTIVE,
                health_score=health_score,
                last_updated=datetime.now(timezone.utc).isoformat(),
                metrics={
                    'total_trades': memory_diag['total_trades'],
                    'memory_level': memory_diag['memory_level'],
                    'brier_score': brier_score,
                    'consecutive_losses': consecutive_losses
                },
                issues=issues,
                performance_score=health_score
            )
        
        return states
```

---

### 5. **backend/main.py (Lifespan Integration)**

Initialize Memory Manager in app lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # ... existing startup code ...
    
    # [NEW] Initialize Memory State Manager in AI Engine
    if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
        logger.info("[STARTUP] Initializing Memory State Manager...")
        
        # Memory manager is already created in AITradingEngine.__init__
        # Just verify it's working
        memory_diag = app.state.ai_engine.memory_manager.get_diagnostics()
        logger.info(
            f"[STARTUP] Memory State ready: "
            f"{memory_diag['total_trades']} historical trades, "
            f"level={memory_diag['memory_level']}"
        )
    
    yield
    
    # [SHUTDOWN] Save memory state
    if hasattr(app.state, 'ai_engine') and app.state.ai_engine:
        logger.info("[SHUTDOWN] Saving memory state...")
        app.state.ai_engine.memory_manager.checkpoint()
```

---

### 6. **backend/routes/ai.py (New API Endpoint)**

Add endpoint to query memory diagnostics:

```python
from backend.services.ai.memory_state_manager import MemoryContext

@router.get("/memory/diagnostics")
async def get_memory_diagnostics(request: Request):
    """Get comprehensive memory diagnostics"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'memory_manager'):
        raise HTTPException(status_code=503, detail="Memory manager not available")
    
    diagnostics = ai_engine.memory_manager.get_diagnostics()
    
    return {
        "status": "success",
        "diagnostics": diagnostics
    }


@router.get("/memory/context")
async def get_memory_context(
    request: Request,
    symbol: Optional[str] = None
):
    """Get current memory context for trading decisions"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'memory_manager'):
        raise HTTPException(status_code=503, detail="Memory manager not available")
    
    context = ai_engine.memory_manager.get_memory_context(symbol=symbol)
    
    return {
        "status": "success",
        "context": context.to_dict()
    }


@router.get("/memory/symbol/{symbol}")
async def get_symbol_memory(
    request: Request,
    symbol: str
):
    """Get memory statistics for specific symbol"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'memory_manager'):
        raise HTTPException(status_code=503, detail="Memory manager not available")
    
    stats = ai_engine.memory_manager.get_symbol_statistics(symbol)
    
    return {
        "status": "success",
        "symbol_stats": stats
    }


@router.post("/memory/reset")
async def reset_memory(request: Request):
    """Emergency reset of memory (use with caution)"""
    ai_engine = request.app.state.ai_engine
    
    if not ai_engine or not hasattr(ai_engine, 'memory_manager'):
        raise HTTPException(status_code=503, detail="Memory manager not available")
    
    ai_engine.memory_manager.reset_performance_memory()
    
    return {
        "status": "success",
        "message": "Memory performance reset"
    }
```

---

## Integration Flow Diagram

```
Market Data (10s tick)
  ↓
AITradingEngine.generate_signals()
  ↓
  ├─→ memory_manager.get_memory_context()
  │   ↓
  │   MemoryContext (adjustments + blacklist)
  │   ↓
  ├─→ Filter blacklisted symbols
  ├─→ Adjust confidence scores
  ├─→ Update regime state
  ├─→ Hash market setup (pattern memory)
  ├─→ Query pattern memory
  ↓
Signals → OrchestratorPolicy.create_policy(memory_context)
  ↓
  ├─→ Apply confidence_adjustment to threshold
  ├─→ Apply risk_multiplier to position sizing
  ├─→ Check allow_new_entries flag
  ↓
Adjusted Policy → EventDrivenExecutor
  ↓
Execute Trade
  ↓
  [Trade runs...]
  ↓
Trade Exits
  ↓
memory_manager.record_trade_outcome()
  ↓
  ├─→ Update symbol win rates (EWMA)
  ├─→ Update regime performance
  ├─→ Update confidence calibration
  ├─→ Update pattern memory
  ├─→ Update consecutive wins/losses
  ├─→ Auto-checkpoint check
  ↓
Memory Updated → Next iteration uses new context
```

---

## Configuration

Add to `.env`:

```bash
# Memory State Configuration
MEMORY_EWMA_ALPHA=0.3                    # 0.1-0.5, decay factor for EWMA
MEMORY_MIN_SAMPLES=10                    # Min trades before trusting memory
MEMORY_REGIME_LOCK_DURATION=120          # Seconds to lock after oscillation
MEMORY_CHECKPOINT_PATH=/app/data/memory_state.json
MEMORY_CHECKPOINT_INTERVAL=60            # Seconds between checkpoints
```

---

## Testing Integration

Run integration test:

```python
# test_memory_integration.py
import asyncio
from backend.services.ai_trading_engine import AITradingEngine
from backend.services.ai.memory_state_manager import MarketRegime

async def test_memory_integration():
    # Initialize engine (this creates memory manager)
    engine = AITradingEngine()
    
    # Generate signals (should use memory context)
    signals = await engine.generate_signals()
    
    # Verify memory context was used
    memory_context = engine.memory_manager.get_memory_context()
    print(f"Memory Context: {memory_context.to_dict()}")
    
    # Simulate trade outcome
    if signals:
        signal = signals[0]
        engine.memory_manager.record_trade_outcome(
            symbol=signal.symbol,
            action=signal.action,
            confidence=signal.confidence,
            pnl=25.50,  # $25.50 profit
            regime=MarketRegime(signal.regime),
            setup_hash=signal.setup_hash
        )
    
    # Get updated diagnostics
    diag = engine.memory_manager.get_diagnostics()
    print(f"Total trades recorded: {diag['total_trades']}")
    
    # Checkpoint
    engine.memory_manager.checkpoint()
    print("Checkpoint saved")

if __name__ == "__main__":
    asyncio.run(test_memory_integration())
```

Run:
```bash
docker exec quantum_backend python test_memory_integration.py
```

Expected output:
```
Memory Context: {'confidence_adjustment': 0.0, 'risk_multiplier': 1.0, ...}
Total trades recorded: 1
Checkpoint saved
```

---

## Summary of Changes

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/services/ai/memory_state_manager.py` | **NEW FILE** | Core memory manager implementation |
| `backend/services/ai_trading_engine.py` | **MODIFIED** | Add memory manager init, signal adjustment, outcome recording |
| `backend/services/event_driven_executor.py` | **MODIFIED** | Add trade exit handling with memory feedback |
| `backend/services/orchestrator_policy.py` | **MODIFIED** | Accept and apply memory context |
| `backend/services/ai_hedgefund_os.py` | **MODIFIED** | Monitor memory as subsystem |
| `backend/main.py` | **MODIFIED** | Initialize and checkpoint memory in lifespan |
| `backend/routes/ai.py` | **MODIFIED** | Add memory diagnostic endpoints |
| `.env` | **MODIFIED** | Add memory configuration variables |

---

## Verification Checklist

- [ ] Memory manager initializes without errors
- [ ] Memory context generated every 10s tick
- [ ] Confidence adjustments applied to signals
- [ ] Risk multipliers affect position sizing
- [ ] Blacklist prevents trades on poor performers
- [ ] Trade outcomes recorded correctly
- [ ] EWMA updates working (win rates change smoothly)
- [ ] Regime transitions logged
- [ ] Pattern memory hash function working
- [ ] Checkpoint saves and restores state
- [ ] API endpoints return valid data
- [ ] Emergency stop triggers on 7+ losses
- [ ] Calibration tracking (Brier score calculated)
- [ ] AI-HFOS sees memory subsystem health
