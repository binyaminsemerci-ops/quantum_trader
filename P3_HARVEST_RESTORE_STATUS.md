# P3 HARVEST RESTORE - IMPLEMENTATION STATUS
**Quantum Trader Profit Harvesting System Recovery**
*February 4, 2026*

---

## ✅ COMPLETED CHANGES

### 1. Stream Migration (harvest_brain.py)
**Status**: ✅ COMPLETE
```python
# OLD: quantum:stream:execution.result
# NEW: quantum:stream:apply.result

self.stream_apply_result = os.getenv(
    'STREAM_APPLY_RESULT', 
    'quantum:stream:apply.result'
)
```
**Verification**: Line 61 in harvest_brain.py

---

### 2. P2 Risk Kernel Integration
**Status**: ✅ COMPLETE
```python
# Import added (lines 28-33)
from ai_engine.risk_kernel_harvest import (
    compute_harvest_proposal,
    HarvestTheta,
    PositionSnapshot,
    MarketState,
    P1Proposal
)
```

**Position Model Extended** (lines 147-163):
```python
@dataclass
class Position:
    ...
    age_sec: float = 0.0  # Position age for P2 kill_score
    peak_price: float = 0.0  # Highest price reached (LONG) or lowest (SHORT)
    trough_price: float = 0.0  # Lowest price reached (LONG) or highest (SHORT)
```

**Evaluate Method Updated** (lines 480-530):
```python
def evaluate(self, position: Position) -> List[HarvestIntent]:
    # Build P2 PositionSnapshot
    pos_snapshot = PositionSnapshot(...)
    
    # Fetch market state
    market_state = self._get_market_state(position.symbol)
    
    # Build P1 proposal
    p1_proposal = P1Proposal(...)
    
    # Get harvest theta
    theta = self._get_harvest_theta()
    
    # **RUN P2 HARVEST KERNEL**
    p2_result = compute_harvest_proposal(
        position=pos_snapshot,
        market_state=market_state,
        p1_proposal=p1_proposal,
        theta=theta
    )
    
    harvest_action = p2_result['harvest_action']
    r_net = p2_result['R_net']
    kill_score = p2_result['kill_score']
    
    logger.info(
        f"[HARVEST] {position.symbol} | "
        f"R={r_net:.2f}R | KILL_SCORE={kill_score:.3f} | "
        f"Action={harvest_action}"
    )
```

---

## ⚠️ PENDING CHANGES (Action Handling)

### Required: Replace Old Harvest Logic with P2 Actions
**Location**: harvest_brain.py lines 533-638
**Current**: Old volatility-adjusted ladder logic with MOVE_SL_TRAIL, MOVE_SL_BREAKEVEN
**Required**: P2 action handlers for PARTIAL_25/50/75, FULL_CLOSE_PROPOSED

**Patch Needed**:
```python
# After line 534 (after min_r check), replace with:

# Handle P2 harvest actions
exit_side = 'SELL' if position.side == 'LONG' else 'BUY'

if harvest_action == 'FULL_CLOSE_PROPOSED':
    intent = HarvestIntent(
        intent_type='FULL_CLOSE_PROPOSED',
        symbol=position.symbol,
        side=exit_side,
        qty=position.qty,
        reason=f'KILL_SCORE={kill_score:.3f}',
        r_level=r_net,
        ...
    )
    intents.append(intent)
    logger.warning(f"🔴 FULL_CLOSE: {position.symbol} | KILL_SCORE={kill_score:.3f}")

elif harvest_action == 'PARTIAL_25':
    intent = HarvestIntent(
        intent_type='PARTIAL_25',
        symbol=position.symbol,
        side=exit_side,
        qty=position.qty * 0.25,
        reason=f'R={r_net:.2f}R >= 2.0R (T1)',
        ...
    )
    intents.append(intent)
    logger.info(f"🟡 PARTIAL_25: {position.symbol} | R={r_net:.2f}R")

elif harvest_action == 'PARTIAL_50':
    # Similar for 50% close at 4.0R

elif harvest_action == 'PARTIAL_75':
    # Similar for 75% close at 6.0R

# Handle profit lock SL
new_sl_proposed = p2_result['new_sl_proposed']
if new_sl_proposed:
    sl_intent = HarvestIntent(
        intent_type='PROFIT_LOCK_SL',
        symbol=position.symbol,
        side='MOVE_SL',
        qty=position.qty,
        reason=f'Profit lock @ R={r_net:.2f}R → SL={new_sl_proposed:.2f}',
        ...
    )
    intents.append(sl_intent)
    logger.info(f"📍 PROFIT_LOCK: {position.symbol}")

return intents
```

---

### Required: Add Helper Methods
**Location**: After evaluate() method (around line 640)

```python
def _get_market_state(self, symbol: str) -> MarketState:
    """Fetch market state from Redis or return defaults"""
    try:
        key = f"quantum:market:{symbol}"
        data = self.redis.hgetall(key)
        
        if data:
            return MarketState(
                sigma=float(data.get('sigma', 0.01)),
                ts=float(data.get('ts', 0.35)),
                p_trend=float(data.get('p_trend', 0.5)),
                p_mr=float(data.get('p_mr', 0.3)),
                p_chop=float(data.get('p_chop', 0.2))
            )
    except Exception as e:
        logger.debug(f"Failed to fetch market state for {symbol}: {e}")
    
    # Default market state (neutral)
    return MarketState(sigma=0.01, ts=0.35, p_trend=0.5, p_mr=0.3, p_chop=0.2)

def _get_harvest_theta(self) -> HarvestTheta:
    """Get harvest theta from config or defaults"""
    return HarvestTheta(
        fallback_stop_pct=0.02,
        cost_bps=10.0,
        T1_R=2.0,
        T2_R=4.0,
        T3_R=6.0,
        lock_R=1.5,
        be_plus_pct=0.002,
        kill_threshold=0.6
    )
```

---

### Required: Add quantum:position Sync
**Location**: In HarvestBrainService class, add method:

```python
async def _sync_position_to_redis(self, position: Position) -> None:
    """Sync position to quantum:position:{symbol} Redis key"""
    try:
        key = f"quantum:position:{position.symbol}"
        
        position_data = {
            'symbol': position.symbol,
            'side': position.side,
            'qty': str(position.qty),
            'entry_price': str(position.entry_price),
            'current_price': str(position.current_price),
            'entry_risk': str(position.entry_risk),
            'stop_loss': str(position.stop_loss),
            'take_profit': str(position.take_profit) if position.take_profit else '',
            'unrealized_pnl': str(position.unrealized_pnl),
            'leverage': str(position.leverage),
            'age_sec': str(position.age_sec),
            'last_update_ts': str(position.last_update_ts),
            'source': 'harvest_brain'
        }
        
        self.redis.hset(key, mapping=position_data)
        self.redis.expire(key, 86400)  # 24h TTL
        
        logger.debug(f"[HARVEST] Synced quantum:position:{position.symbol}")
    except Exception as e:
        logger.warning(f"Failed to sync position to Redis: {e}")
```

**Call in process_apply_result** (after ingest_execution):
```python
if status in ['FILLED', 'PARTIAL_FILL']:
    self.tracker.ingest_execution(event)
    
    # Sync to quantum:position:{symbol} Redis key
    if symbol in self.tracker.positions:
        pos = self.tracker.positions[symbol]
        await self._sync_position_to_redis(pos)
```

---

## ✅ EXITBRAIN v3.5 LSF INTEGRATION

### Status: ALREADY IMPLEMENTED ✅
**File**: microservices/exitbrain_v3_5/exit_brain.py

**Verification** (lines 150-170):
```python
# Step 2: Calculate adaptive TP/SL levels using AdaptiveLeverageEngine
adaptive_levels = self.adaptive_engine.compute_levels(
    base_tp_pct=self.base_tp_pct,
    base_sl_pct=self.base_sl_pct,
    leverage=leverage_calc.leverage,
    volatility_factor=signal.atr_value / 100.0,
    funding_delta=funding_rate,
    exchange_divergence=exch_divergence
)

# Use adaptive levels as base TP/SL
base_tp = adaptive_levels.tp1_pct  # Use TP1 as primary target
base_sl = adaptive_levels.sl_pct

# Store full TP levels for partial harvesting
tp_levels = [adaptive_levels.tp1_pct, adaptive_levels.tp2_pct, adaptive_levels.tp3_pct]
harvest_scheme = adaptive_levels.harvest_scheme
```

**LSF Formula** (adaptive_leverage_engine.py lines 70-85):
```python
def compute_lsf(self, leverage: float) -> float:
    """
    Compute Leverage Sensitivity Factor (LSF)
    
    Formula: LSF = 1 / (1 + ln(leverage + 1))
    Higher leverage → Lower LSF → Tighter TP/SL
    """
    lev = max(float(leverage), 1.0)
    lsf = 1.0 / (1.0 + math.log(lev + 1.0))
    return lsf
```

**TP/SL Calculation** (adaptive_leverage_engine.py lines 132-140):
```python
# Core LSF-based formula
tp1 = base_tp * (0.6 + lsf)
tp2 = base_tp * (1.2 + lsf / 2.0)
tp3 = base_tp * (1.8 + lsf / 4.0)
sl = base_sl * (1.0 + (1.0 - lsf) * 0.8)
```

**Result**: ExitBrain v3.5 already fully integrated with LSF-based adaptive TP/SL ✅

---

## 📋 MANUAL COMPLETION STEPS

### Step 1: Complete harvest_brain.py Action Handling
```bash
# Edit harvest_brain.py lines 533-638
# Replace old ladder logic with P2 action handlers
```

### Step 2: Add Helper Methods
```bash
# Add _get_market_state() and _get_harvest_theta() after evaluate()
```

### Step 3: Add Position Sync
```bash
# Add _sync_position_to_redis() method to HarvestBrainService
# Call it after ingest_execution in process_apply_result
```

### Step 4: Deploy & Test
```bash
# Deploy to VPS
wsl ssh root@46.224.116.254 'cd /root/quantum_trader && git pull'

# Restart harvest-brain service
wsl ssh root@46.224.116.254 'systemctl restart quantum-harvest-brain'

# Monitor logs
wsl ssh root@46.224.116.254 'journalctl -u quantum-harvest-brain -f --no-pager'
```

### Step 5: Verify Proof
```bash
# Check Redis keys
redis-cli --scan --pattern "quantum:position:*"

# Check stream activity
redis-cli XLEN quantum:stream:apply.result
redis-cli XLEN quantum:stream:trade.intent

# Check logs for markers
journalctl -u quantum-harvest-brain | grep -E "\[HARVEST\]|R=|KILL_SCORE="
```

---

## 🎯 EXPECTED OUTCOMES

### Redis Keys
```
quantum:position:BTCUSDT → hash with position data
quantum:position:ETHUSDT → hash with position data
```

### Log Patterns
```
[HARVEST] BTCUSDT | R=2.35R | KILL_SCORE=0.412 | Action=PARTIAL_25 | Reasons=['harvest_partial_25']
🟡 PARTIAL_25: BTCUSDT | R=2.35R
📍 PROFIT_LOCK: ETHUSDT | SL 3050.00 → 3065.50 | R=1.85R
🔴 FULL_CLOSE: SOLUSDT | KILL_SCORE=0.721 > threshold | R=3.12R
```

### Stream Trace
```
apply.result (FILLED)
  ↓
harvest_brain (evaluates position)
  ↓
compute_harvest_proposal() (P2 kernel)
  ↓
trade.intent (PARTIAL_25/50/75 or FULL_CLOSE_PROPOSED)
  ↓
apply.plan (intent-bridge)
  ↓
apply-layer (execution)
```

---

## 📊 CURRENT STATUS SUMMARY

| Component | Status | Notes |
|-----------|--------|-------|
| Stream migration (execution.result → apply.result) | ✅ COMPLETE | Line 61 |
| P2 risk_kernel import | ✅ COMPLETE | Lines 28-33 |
| Position model extension | ✅ COMPLETE | Lines 147-163 |
| Evaluate method with P2 kernel | ✅ COMPLETE | Lines 480-530 |
| P2 action handlers (PARTIAL_25/50/75, FULL_CLOSE) | ⚠️ PENDING | Lines 533-638 need replacement |
| Helper methods (_get_market_state, _get_harvest_theta) | ⚠️ PENDING | Add after evaluate() |
| quantum:position sync | ⚠️ PENDING | Add _sync_position_to_redis() |
| ExitBrain v3.5 LSF integration | ✅ COMPLETE | Already implemented |

**Overall Progress**: 70% complete
**Remaining Work**: ~150 lines of code (action handlers + helpers + sync)

---

## 🔧 QUICK PATCH SCRIPT

Create `p3_harvest_patch.py` to apply remaining changes:

```python
#!/usr/bin/env python3
"""P3 Harvest Restore - Automated Patch"""
import re

def patch_harvest_brain():
    file_path = 'microservices/harvest_brain/harvest_brain.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace old evaluate logic (lines 533-638)
    old_pattern = r'(# Skip if below min_r threshold.*?return intents)\n\n(# Check for trailing stop.*?return intents)'
    
    new_logic = '''# Skip if below min_r threshold
        if r_net < self.config.min_r:
            logger.debug(f"{position.symbol}: R={r_net:.2f} < min_r={self.config.min_r}")
            return intents
        
        # Handle P2 harvest actions
        exit_side = 'SELL' if position.side == 'LONG' else 'BUY'
        
        if harvest_action == 'FULL_CLOSE_PROPOSED':
            # ... (full close handler)
        elif harvest_action == 'PARTIAL_25':
            # ... (25% close handler)
        # ... (rest of handlers)
        
        return intents'''
    
    content = re.sub(old_pattern, new_logic, content, flags=re.DOTALL)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✅ Patched harvest_brain.py")

if __name__ == '__main__':
    patch_harvest_brain()
```

---

**END OF P3 HARVEST RESTORE IMPLEMENTATION STATUS**
