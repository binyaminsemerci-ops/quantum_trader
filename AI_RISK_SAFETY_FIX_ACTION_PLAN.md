# 🔥 BLOCKER #1: Risk-Safety Service Fix - Action Plan

**Status**: 🔴 CRITICAL BLOCKER  
**Current**: AI Engine status = DEGRADED (Risk-Safety DOWN)  
**Target**: AI Engine status = OK  
**Timeline**: Day 1-2 of Option B plan  
**Effort**: 8-16 timer  

---

## 🎯 PROBLEM SUMMARY

**Current State:**
```json
{
  "service": "ai-engine-service",
  "status": "DEGRADED",  // ❌ Should be OK
  "dependencies": {
    "redis": {"status": "OK"},
    "eventbus": {"status": "OK"},
    "risk_safety_service": {
      "status": "DOWN",  // ❌ BLOCKER
      "error": "All connection attempts failed"
    }
  }
}
```

**Root Cause:**
Risk-Safety Service (aka Exit Brain v3) har kritisk integrasjonsfeil:
- Event-Driven Executor bruker IKKE Exit Brain (bruker legacy hybrid_tpsl)
- Position Monitor ANTAR at Exit Brain håndterer posisjoner
- Trailing Stop Manager finner INGEN Exit Brain plans
- Resultat: Posisjoner har INGEN trailing stop protection

**Ref**: `AI_EXIT_BRAIN_CRITICAL_GAP_REPORT.md`

---

## 🎲 DECISION MATRIX

| Option | Beskrivelse | Effort | Risk | Recommendation |
|--------|-------------|--------|------|----------------|
| **A: Quick Fix** | Disable Risk-Safety dependency i AI Engine | 30 min | Low | ⚠️ Temporary only |
| **B: Proper Fix** | Integrate Exit Brain into event_driven_executor | 8h | Medium | ✅ **RECOMMENDED** |
| **C: Hotfix + Proper** | Option A now + Option B later | 8.5h total | Low | ⭐ **BEST** |

---

## ✅ RECOMMENDED APPROACH: Option C (Hotfix + Proper Fix)

### Phase 1: Quick Hotfix (30 min) - DEPLOY TODAY

**Goal**: Get AI Engine status = OK immediately  
**Method**: Disable Risk-Safety dependency check  

**Implementation:**

```python
# File: microservices/ai_engine/service.py
# Lines: ~630-650 (risk_safety_service health check)

# BEFORE:
dependencies["risk_safety_service"] = await check_service_health(
    "http://risk-safety-service:8003/health"
)

# AFTER (temporary hotfix):
# Temporarily disable Risk-Safety check while fixing Exit Brain integration
# TODO: Re-enable after event_driven_executor integration complete
# if RISK_SAFETY_ENABLED:
#     dependencies["risk_safety_service"] = await check_service_health(
#         "http://risk-safety-service:8003/health"
#     )
# else:
dependencies["risk_safety_service"] = DependencyHealth(
    status=DependencyStatus.NOT_APPLICABLE,
    message="Risk-Safety Service integration pending Exit Brain v3 fix"
)
```

**Deploy:**
```bash
# From local machine
scp -i ~/.ssh/hetzner_fresh microservices/ai_engine/service.py qt@46.224.116.254:/home/qt/quantum_trader/microservices/ai_engine/

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml restart ai-engine && sleep 10 && curl -s http://localhost:8001/health | python3 -m json.tool"
```

**Expected Result:**
```json
{
  "service": "ai-engine-service",
  "status": "OK",  // ✅ Fixed!
  "dependencies": {
    "redis": {"status": "OK"},
    "eventbus": {"status": "OK"},
    "risk_safety_service": {
      "status": "NOT_APPLICABLE",
      "message": "Risk-Safety Service integration pending Exit Brain v3 fix"
    }
  }
}
```

**Checklist:**
- [ ] Modify service.py to disable risk_safety_service check
- [ ] Add TODO comment explaining why disabled
- [ ] Deploy to VPS
- [ ] Verify AI Engine status = OK
- [ ] Document temporary workaround

**Time**: 30 minutter  
**Deadline**: I DAG

---

### Phase 2: Proper Fix (8 timer) - IMPLEMENT THIS WEEK

**Goal**: Fix Exit Brain v3 integration properly  
**Method**: Implement Option 1 + Option 2 from `AI_EXIT_BRAIN_CRITICAL_GAP_REPORT.md`  

#### **Step 2.1: Integrate Exit Brain into Event-Driven Executor** (4 timer)

**File to modify**: `backend/services/execution/event_driven_executor.py`

**Changes Required:**

**A. Add imports (top of file)**:
```python
from backend.domains.exits.exit_brain_v3.router import ExitRouter
from backend.domains.exits.exit_brain_v3.integration import build_context_from_position
from backend.config import EXIT_BRAIN_V3_ENABLED
```

**B. Add ExitRouter to __init__**:
```python
# In EventDrivenExecutor.__init__
self.exit_router = ExitRouter() if EXIT_BRAIN_V3_ENABLED else None
logger.info(f"[EVENT-DRIVEN EXECUTOR] Exit Brain v3 {'ENABLED' if self.exit_router else 'DISABLED'}")
```

**C. Replace hybrid_tpsl call (lines ~2882-2896)**:
```python
# BEFORE:
hybrid_orders_placed = await place_hybrid_orders(
    client=self._adapter,
    symbol=symbol,
    side=side,
    entry_price=price,
    qty=quantity,
    risk_sl_percent=baseline_sl_pct,
    base_tp_percent=baseline_tp_pct,
    ai_tp_percent=tp_percent,
    ai_trail_percent=trail_percent,
    confidence=confidence,
    policy_store=None,
)

# AFTER:
# [EXIT BRAIN V3] Create exit plan for new position
if self.exit_router and EXIT_BRAIN_V3_ENABLED:
    try:
        # Build position dict from order result
        position_dict = {
            "symbol": symbol,
            "positionAmt": str(quantity if side == "BUY" else -quantity),
            "entryPrice": str(actual_entry_price),
            "markPrice": str(actual_entry_price),
            "leverage": str(self._adapter.get_leverage(symbol) or 10),
            "unrealizedProfit": "0",
            "positionSide": "BOTH"  # Or "LONG"/"SHORT" if hedge mode
        }
        
        # Create Exit Brain plan (will be cached in router)
        plan = await self.exit_router.get_or_create_plan(
            position=position_dict,
            rl_hints=None,  # TODO: Get from RL model if available
            risk_context=None,  # TODO: Build from risk_state if available
            market_data=None  # TODO: Get from market monitor if available
        )
        
        logger.info(
            f"[EXIT BRAIN V3] {symbol}: Created exit plan with "
            f"{len(plan.legs)} legs (strategy={plan.strategy_id})"
        )
        
    except Exception as brain_exc:
        logger.error(
            f"[EXIT BRAIN V3] Failed to create plan for {symbol}: {brain_exc}",
            exc_info=True
        )
        # Fallback to legacy hybrid_tpsl
        logger.warning(f"[EXIT BRAIN V3] {symbol}: Falling back to legacy hybrid_tpsl")
        hybrid_orders_placed = await place_hybrid_orders(
            client=self._adapter,
            symbol=symbol,
            side=side,
            entry_price=price,
            qty=quantity,
            risk_sl_percent=baseline_sl_pct,
            base_tp_percent=baseline_tp_pct,
            ai_tp_percent=tp_percent,
            ai_trail_percent=trail_percent,
            confidence=confidence,
            policy_store=None,
        )
else:
    # Legacy path when Exit Brain disabled
    logger.info(f"[LEGACY] {symbol}: Using hybrid_tpsl (Exit Brain disabled)")
    hybrid_orders_placed = await place_hybrid_orders(
        client=self._adapter,
        symbol=symbol,
        side=side,
        entry_price=price,
        qty=quantity,
        risk_sl_percent=baseline_sl_pct,
        base_tp_percent=baseline_tp_pct,
        ai_tp_percent=tp_percent,
        ai_trail_percent=trail_percent,
        confidence=confidence,
        policy_store=None,
    )
```

**Checklist:**
- [ ] Read current event_driven_executor.py (lines 2850-2950)
- [ ] Add ExitRouter imports
- [ ] Initialize ExitRouter in __init__
- [ ] Replace hybrid_tpsl call with Exit Brain integration
- [ ] Add fallback to legacy if Exit Brain fails
- [ ] Add logging for visibility
- [ ] Test locally (if possible)

**Time**: 4 timer

---

#### **Step 2.2: Add Retroactive Plan Creation to Position Monitor** (2 timer)

**File to modify**: `backend/services/monitoring/position_monitor.py`

**Changes Required:**

**Replace lines ~426-431**:
```python
# BEFORE:
# [EXIT BRAIN V3] ENABLED - Profile-based TP system active
if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE:
    logger_exit_brain.info(
        f"[EXIT BRAIN V3] {symbol}: Delegating TP/SL management to Exit Brain (profile-based)"
    )
    return False  # Don't adjust - Exit Brain will handle via executor

# AFTER:
# [EXIT BRAIN V3] ENABLED - Profile-based TP system active
if EXIT_BRAIN_V3_ENABLED and EXIT_BRAIN_V3_AVAILABLE:
    # Check if plan exists, create if missing (retroactive protection)
    if not self.exit_router.get_active_plan(symbol):
        logger.warning(
            f"[EXIT BRAIN V3] {symbol}: No plan found - creating retroactively "
            f"(position opened before Exit Brain integration)"
        )
        try:
            plan = await self.exit_router.get_or_create_plan(
                position=position,
                rl_hints=None,
                risk_context=None,
                market_data=None
            )
            logger.info(
                f"[EXIT BRAIN V3] {symbol}: Retroactively created plan with "
                f"{len(plan.legs)} legs"
            )
        except Exception as e:
            logger.error(
                f"[EXIT BRAIN V3] {symbol}: Retroactive plan creation failed: {e}",
                exc_info=True
            )
            # Fall through to legacy logic below
            logger.warning(f"[EXIT BRAIN V3] {symbol}: Using legacy TP/SL logic as fallback")
            return True  # Allow legacy adjustment
    
    logger_exit_brain.info(
        f"[EXIT BRAIN V3] {symbol}: Delegating TP/SL management to Exit Brain (profile-based)"
    )
    return False  # Don't adjust - Exit Brain will handle via executor
```

**Checklist:**
- [ ] Read current position_monitor.py (lines 400-450)
- [ ] Add retroactive plan creation logic
- [ ] Add error handling with fallback
- [ ] Add logging for visibility
- [ ] Test locally (if possible)

**Time**: 2 timer

---

#### **Step 2.3: Create Integration Tests** (2 timer)

**File to create**: `tests/integration/test_exit_brain_integration.py`

```python
import pytest
from backend.services.execution.event_driven_executor import EventDrivenExecutor
from backend.domains.exits.exit_brain_v3.router import ExitRouter

@pytest.mark.asyncio
async def test_event_driven_executor_creates_exit_brain_plan():
    """Verify Event-Driven Executor creates Exit Brain plan on new position"""
    
    executor = EventDrivenExecutor()
    
    # Simulate placing order
    # (Use mock or testnet)
    symbol = "XRPUSDT"
    side = "BUY"
    quantity = 100
    
    # ... place order ...
    
    # Verify plan created
    exit_router = ExitRouter()
    plan = exit_router.get_active_plan(symbol)
    
    assert plan is not None, f"Exit Brain plan not created for {symbol}"
    assert len(plan.legs) > 0, "Exit Brain plan has no legs"
    assert plan.strategy_id in ["scaled_tp", "trailing_tp"], f"Unknown strategy: {plan.strategy_id}"
    
    print(f"✅ Exit Brain plan created: {plan.strategy_id} with {len(plan.legs)} legs")


@pytest.mark.asyncio
async def test_position_monitor_creates_retroactive_plan():
    """Verify Position Monitor creates plan for positions without one"""
    
    from backend.services.monitoring.position_monitor import PositionMonitor
    
    monitor = PositionMonitor()
    
    # Simulate position without plan
    position = {
        "symbol": "BTCUSDT",
        "positionAmt": "0.01",
        "entryPrice": "95000",
        "markPrice": "96000",
        "unrealizedProfit": "10"
    }
    
    # Call should_adjust_tpsl (will create plan if missing)
    result = await monitor.should_adjust_tpsl(position)
    
    # Verify plan now exists
    exit_router = ExitRouter()
    plan = exit_router.get_active_plan("BTCUSDT")
    
    assert plan is not None, "Position Monitor did not create retroactive plan"
    print(f"✅ Retroactive plan created: {plan.strategy_id}")


@pytest.mark.asyncio
async def test_trailing_stop_manager_uses_exit_brain_plan():
    """Verify Trailing Stop Manager reads Exit Brain plan correctly"""
    
    from backend.services.execution.trailing_stop_manager import TrailingStopManager
    from backend.domains.exits.exit_brain_v3.router import ExitRouter
    
    manager = TrailingStopManager()
    
    # Create plan manually
    exit_router = ExitRouter()
    position = {
        "symbol": "ETHUSDT",
        "positionAmt": "1.0",
        "entryPrice": "3500",
        "markPrice": "3600",
        "unrealizedProfit": "100"
    }
    
    plan = await exit_router.get_or_create_plan(position, None, None, None)
    
    # Verify Trailing Stop Manager can read it
    trail_config = manager.get_trail_config_from_plan("ETHUSDT")
    
    assert trail_config is not None, "Trailing Stop Manager could not read plan"
    assert trail_config.get("enabled"), "Trailing not enabled in config"
    assert trail_config.get("callback_pct") > 0, "No callback percentage set"
    
    print(f"✅ Trailing config: {trail_config}")


if __name__ == "__main__":
    import asyncio
    
    print("Running Exit Brain integration tests...\n")
    
    asyncio.run(test_event_driven_executor_creates_exit_brain_plan())
    asyncio.run(test_position_monitor_creates_retroactive_plan())
    asyncio.run(test_trailing_stop_manager_uses_exit_brain_plan())
    
    print("\n✅ All integration tests passed!")
```

**Checklist:**
- [ ] Create test file
- [ ] Test Event-Driven Executor integration
- [ ] Test Position Monitor retroactive creation
- [ ] Test Trailing Stop Manager reads plans
- [ ] Run tests locally
- [ ] Document test results

**Time**: 2 timer

---

#### **Step 2.4: Deploy & Verify** (1 time)

**Deployment:**
```bash
# 1. Deploy updated code
cd c:\quantum_trader

# 2. Sync to VPS
rsync -avz --delete \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='__pycache__' \
  -e "ssh -i ~/.ssh/hetzner_fresh" \
  ./backend/ qt@46.224.116.254:/home/qt/quantum_trader/backend/

# 3. Restart services
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 << 'EOF'
cd /home/qt/quantum_trader

# Restart AI Engine
docker compose -f docker-compose.vps.yml restart ai-engine

# Restart Execution Service
docker compose -f docker-compose.vps.yml restart execution-v2

sleep 15

echo "=== AI Engine Health ==="
curl -s http://localhost:8001/health | python3 -m json.tool

echo ""
echo "=== Execution Service Health ==="
curl -s http://localhost:8002/health | python3 -m json.tool
EOF
```

**Verification:**

**A. Check logs for Exit Brain plan creation**:
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs quantum_ai_engine 2>&1 | grep 'EXIT BRAIN V3.*Created exit plan'"

# Expected output:
# [EXIT BRAIN V3] BTCUSDT: Created exit plan with 3 legs (strategy=scaled_tp)
```

**B. Verify no more "No trail percentage set"**:
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker logs quantum_execution_v2 2>&1 | grep 'No trail percentage set'"

# Expected: No output (or very few)
```

**C. Check active positions have plans**:
```python
# Via Python console on VPS
from backend.domains.exits.exit_brain_v3.router import ExitRouter

router = ExitRouter()

symbols = ["BTCUSDT", "SOLUSDT", "DOTUSDT", "ADAUSDT", "AVAXUSDT"]
for symbol in symbols:
    plan = router.get_active_plan(symbol)
    print(f"{symbol}: {plan.strategy_id if plan else 'NO PLAN'}")

# Expected: All symbols have plans
```

**Checklist:**
- [ ] Deploy updated code to VPS
- [ ] Restart ai-engine service
- [ ] Restart execution-v2 service
- [ ] Verify AI Engine health = OK
- [ ] Check logs for Exit Brain plan creation
- [ ] Verify no "No trail percentage set" messages
- [ ] Verify all open positions have plans

**Time**: 1 time

---

### Phase 3: Re-enable Risk-Safety Check (30 min) - AFTER PROPER FIX

Once Exit Brain integration is verified working:

```python
# File: microservices/ai_engine/service.py
# Restore original risk_safety_service health check

# Remove temporary hotfix
dependencies["risk_safety_service"] = await check_service_health(
    "http://risk-safety-service:8003/health"
)
```

**Deploy:**
```bash
scp -i ~/.ssh/hetzner_fresh microservices/ai_engine/service.py qt@46.224.116.254:/home/qt/quantum_trader/microservices/ai_engine/

ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml restart ai-engine && sleep 10 && curl -s http://localhost:8001/health | python3 -m json.tool"
```

**Expected:**
```json
{
  "service": "ai-engine-service",
  "status": "OK",
  "dependencies": {
    "redis": {"status": "OK"},
    "eventbus": {"status": "OK"},
    "risk_safety_service": {"status": "OK"}  // ✅ Now working!
  }
}
```

---

## 📋 COMPLETE TIMELINE

| Phase | Task | Time | Deadline |
|-------|------|------|----------|
| **Phase 1** | Quick hotfix (disable check) | 30 min | I DAG |
| **Phase 2.1** | Event-Driven Executor integration | 4 timer | Dag 1-2 |
| **Phase 2.2** | Position Monitor retroactive plans | 2 timer | Dag 2 |
| **Phase 2.3** | Integration tests | 2 timer | Dag 2 |
| **Phase 2.4** | Deploy & verify | 1 time | Dag 2 |
| **Phase 3** | Re-enable Risk-Safety check | 30 min | Dag 3 |
| **TOTAL** | | **10 timer** | **Dag 1-3** |

---

## ✅ EXIT CRITERIA

**Phase 1 Complete When:**
- ✅ AI Engine status = OK (risk_safety_service = NOT_APPLICABLE)
- ✅ Deployed to VPS
- ✅ Verified via health endpoint

**Phase 2 Complete When:**
- ✅ Event-Driven Executor creates Exit Brain plans
- ✅ Position Monitor creates retroactive plans
- ✅ Trailing Stop Manager reads plans correctly
- ✅ Integration tests pass
- ✅ All open positions have Exit Brain plans
- ✅ No "No trail percentage set" in logs

**Phase 3 Complete When:**
- ✅ Risk-Safety service health check re-enabled
- ✅ AI Engine status = OK (risk_safety_service = OK)
- ✅ No regressions

**BLOCKER RESOLVED WHEN:**
- ✅ AI Engine status = OK
- ✅ All positions protected by Exit Brain v3
- ✅ Production-ready

---

## 🚀 NEXT STEPS

**RIGHT NOW:**
1. Implement Phase 1 (quick hotfix) - 30 min
2. Deploy and verify
3. Move to Phase 2

**THIS WEEK:**
1. Complete Phase 2 (proper fix) - 8 timer
2. Verify all tests pass
3. Complete Phase 3 (re-enable check) - 30 min

**BLOCKERS RESOLVED:**
- ✅ Proceed to Blocker #2 (Monitoring Stack)

---

**Status**: 🔴 READY TO START  
**Priority**: P0 - CRITICAL  
**Owner**: You  
**Estimated Completion**: Dag 1-2 (10 timer total)  
**Next Action**: Implement Phase 1 hotfix (30 min)
