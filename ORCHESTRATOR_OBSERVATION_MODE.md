# Orchestrator Observation Mode - Integration Guide

## Overview

The OrchestratorPolicy has been successfully integrated into Quantum Trader in **OBSERVATION MODE**. This means:

✅ **Policy is computed** every trading cycle  
✅ **All decisions are logged** for analysis  
❌ **No enforcement** - trading behavior is unchanged  

This allows us to:
1. Validate policy logic against real market conditions
2. Tune policy parameters before going live
3. Build confidence in orchestrator decisions
4. Ensure zero disruption to current trading

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 EventDrivenExecutor                         │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Collect Subsystem Inputs                       │    │
│  │     - RegimeDetector → regime_tag, vol_level      │    │
│  │     - GlobalRiskController → risk_state           │    │
│  │     - SymbolPerformanceManager → symbol_perf      │    │
│  │     - CostModel → cost_metrics                    │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  2. OrchestratorPolicy.update_policy(...)          │    │
│  │     Returns: TradingPolicy                         │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  3. 👁️ OBSERVATION MODE                           │    │
│  │     PolicyObserver.log_policy_update(...)          │    │
│  │     - Log what policy says                         │    │
│  │     - Log what actually happened                   │    │
│  │     - Compare policy vs reality                    │    │
│  └────────────────────────────────────────────────────┘    │
│                        ↓                                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  4. Continue with EXISTING logic                   │    │
│  │     - Use fixed confidence threshold (0.45)        │    │
│  │     - Use existing filters                         │    │
│  │     - No policy enforcement                        │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Components

### 1. OrchestratorIntegrationConfig
**File:** `backend/services/orchestrator_config.py`

```python
# Current configuration (OBSERVE mode)
config = OrchestratorIntegrationConfig.create_observe_mode()

# Key flags:
config.enable_orchestrator = True
config.mode = OrchestratorMode.OBSERVE  # ← Critical!
config.use_for_signal_filter = False     # Not enforced
config.use_for_confidence_threshold = False
config.use_for_risk_sizing = False
config.use_for_position_limits = False
config.use_for_trading_gate = False
config.use_for_exit_mode = False
```

### 2. PolicyObserver
**File:** `backend/services/policy_observer.py`

Logs two types of observations:

#### A. Policy Updates (once per cycle)
**Location:** `data/policy_observations/policy_obs_YYYY-MM-DD.jsonl`

```json
{
  "timestamp": "2025-11-22T12:00:00Z",
  "mode": "OBSERVE",
  "inputs": {
    "regime_tag": "TRENDING",
    "vol_level": "NORMAL",
    "risk_state": {...},
    "symbol_performance": [...]
  },
  "policy": {
    "allow_new_trades": true,
    "min_confidence": 0.48,
    "risk_per_trade_pct": 1.0,
    "disallowed_symbols": ["DOGEUSD", "SHIBUSDT"]
  },
  "actual": {
    "confidence_used": 0.45,
    "trading_allowed": true
  },
  "comparison": {
    "would_raise_confidence": true,
    "confidence_delta": 0.03
  }
}
```

#### B. Signal Decisions (one per signal)
**Location:** `data/policy_observations/signals_YYYY-MM-DD.jsonl`

```json
{
  "timestamp": "2025-11-22T12:00:00Z",
  "type": "signal_decision",
  "signal": {
    "symbol": "BTCUSDT",
    "action": "BUY",
    "confidence": 0.52
  },
  "actual_decision": "ALLOWED",
  "policy_verdict": "ALLOW",
  "agreement": true
}
```

### 3. EventDrivenExecutor Integration
**File:** `backend/services/event_driven_executor.py`

**Initialization:**
```python
# Orchestrator (computes policy)
self.orchestrator = OrchestratorPolicy(config=...)

# Integration config (controls enforcement)
self.orch_config = OrchestratorIntegrationConfig.create_observe_mode()

# Observer (logs decisions)
self.policy_observer = PolicyObserver(log_dir="data/policy_observations")
```

**In _check_and_execute() loop:**
```python
# 1. Collect inputs
risk_state = create_risk_state(...)
symbol_perf_list = [...]
cost_metrics = create_cost_metrics(...)

# 2. Compute policy
policy = self.orchestrator.update_policy(
    regime_tag=regime_tag,
    vol_level=vol_level,
    risk_state=risk_state,
    symbol_performance=symbol_perf_list,
    cost_metrics=cost_metrics
)

# 3. 👁️ OBSERVE: Log but don't enforce
if self.orch_config.is_observe_mode():
    logger.info(f"👁️ OBSERVE MODE - Policy: {policy.note}")
    # Continue with existing logic
    effective_confidence = self.confidence_threshold  # NOT policy.min_confidence

# 4. Log full observation
self.policy_observer.log_policy_update(
    policy=policy,
    regime_tag=regime_tag,
    ...
)
```

---

## What Gets Logged

### Console Output
```
✅ Orchestrator in OBSERVE mode (logging only, no enforcement)
👁️ OBSERVE MODE - Policy computed but NOT enforced: Normal conditions: good model quality
📊 POLICY OBSERVATION | Allow=True | MinConf=0.45 (vs 0.45) | Risk=1.0% | MaxPos=8 | Blocked=0 symbols
```

### File Logs

**Daily policy observations:**
```
data/policy_observations/
├── policy_obs_2025-11-22.jsonl   # Policy updates (one per cycle)
└── signals_2025-11-22.jsonl      # Signal decisions (one per signal)
```

**In-memory history:**
- Last 100 policy observations
- Accessible via `policy_observer.get_recent_policies()`

---

## Current Behavior (OBSERVE Mode)

| Component | Status | Notes |
|-----------|--------|-------|
| **Policy Computation** | ✅ Active | Computed every cycle |
| **Policy Logging** | ✅ Active | All decisions logged to disk |
| **Signal Filtering** | ❌ Not enforced | policy.disallowed_symbols ignored |
| **Confidence Threshold** | ❌ Not enforced | Fixed 0.45, not policy.min_confidence |
| **Risk Sizing** | ❌ Not enforced | Fixed 1%, not policy.risk_per_trade_pct |
| **Position Limits** | ❌ Not enforced | Fixed 8, not policy.max_open_positions |
| **Trading Gate** | ❌ Not enforced | policy.allow_new_trades ignored |
| **Exit Mode** | ❌ Not enforced | policy.exit_mode_override ignored |

**Trading behavior:** Identical to before orchestrator integration

---

## Transitioning to LIVE Mode

### Phase 1: Gradual Rollout (Recommended)

**Step 1: Enable Signal Filtering Only**
```python
config = OrchestratorIntegrationConfig(
    enable_orchestrator=True,
    mode=OrchestratorMode.LIVE,
    use_for_signal_filter=True,       # ✅ Enable
    use_for_confidence_threshold=False,
    use_for_risk_sizing=False,
    use_for_position_limits=False,
    use_for_trading_gate=False,
    use_for_exit_mode=False
)
```

**Change in code:**
```python
# In _check_and_execute(), enable this block:
if self.orch_config.is_live_mode() and self.orch_config.use_for_signal_filter:
    if policy and symbol in policy.disallowed_symbols:
        logger.debug(f"⏭️ Skipping disallowed symbol {symbol} (policy)")
        continue  # NOW ENFORCED
```

**Step 2: Enable Confidence Adjustment**
```python
config.use_for_confidence_threshold = True
```

**Change in code:**
```python
if self.orch_config.is_live_mode() and self.orch_config.use_for_confidence_threshold:
    effective_confidence = max(self.confidence_threshold, policy.min_confidence)
else:
    effective_confidence = self.confidence_threshold
```

**Step 3: Enable Risk Sizing**
```python
config.use_for_risk_sizing = True
```

**Step 4: Enable Position Limits**
```python
config.use_for_position_limits = True
```

**Step 5: Enable Trading Gate**
```python
config.use_for_trading_gate = True
```

**Step 6: Enable Exit Mode Override**
```python
config.use_for_exit_mode = True
```

### Phase 2: Full LIVE Mode

```python
# Switch to full enforcement
config = OrchestratorIntegrationConfig.create_live_mode_full()
```

Update `event_driven_executor.py` to respect all policy fields:
```python
if self.orch_config.is_live_mode():
    # Signal filtering
    if self.orch_config.use_for_signal_filter and symbol in policy.disallowed_symbols:
        continue
    
    # Confidence threshold
    if self.orch_config.use_for_confidence_threshold:
        effective_confidence = max(self.confidence_threshold, policy.min_confidence)
    
    # Trading gate
    if self.orch_config.use_for_trading_gate and not policy.allow_new_trades:
        logger.warning(f"⛔ Orchestrator blocks trading: {policy.note}")
        return
    
    # Risk sizing
    if self.orch_config.use_for_risk_sizing:
        risk_pct = policy.risk_per_trade_pct
    
    # Position limits
    if self.orch_config.use_for_position_limits:
        max_positions = policy.max_open_positions
    
    # Exit mode
    if self.orch_config.use_for_exit_mode and policy.exit_mode_override:
        exit_mode = policy.exit_mode_override
```

---

## Validation & Testing

### 1. Check Observation Logs
```python
from backend.services.policy_observer import PolicyObserver

observer = PolicyObserver()
stats = observer.get_policy_stats()
print(stats)
```

Output:
```python
{
    'total_observations': 150,
    'recent_sample_size': 20,
    'blocked_trading_pct': 0.0,
    'avg_min_confidence': 0.45,
    'avg_risk_per_trade': 1.0,
    'log_file': 'data/policy_observations/policy_obs_2025-11-22.jsonl'
}
```

### 2. Analyze Policy vs Reality
```python
import json

# Load observations
with open("data/policy_observations/policy_obs_2025-11-22.jsonl") as f:
    observations = [json.loads(line) for line in f]

# Check agreement rate
agreements = [
    obs for obs in observations 
    if obs['comparison']['would_raise_confidence'] == False
]
print(f"Policy agreement: {len(agreements)}/{len(observations)}")
```

### 3. Signal Decision Analysis
```python
with open("data/policy_observations/signals_2025-11-22.jsonl") as f:
    signals = [json.loads(line) for line in f]

# Count disagreements
disagreements = [s for s in signals if not s['agreement']]
print(f"Disagreements: {len(disagreements)}/{len(signals)}")
```

---

## Safety Features

✅ **Zero impact on trading:** Observe mode guarantees no behavior change  
✅ **Comprehensive logging:** Every decision recorded for audit  
✅ **Gradual rollout:** Enable features one at a time  
✅ **Rollback ready:** Switch back to OBSERVE with one line  
✅ **Error isolation:** Policy failures don't crash trading loop  

---

## Configuration Summary

### Current State (OBSERVE)
```python
# In event_driven_executor.py __init__:
self.orch_config = OrchestratorIntegrationConfig.create_observe_mode()
```

### When Ready for LIVE
```python
# Phase 1: Gradual
self.orch_config = OrchestratorIntegrationConfig.create_live_mode_gradual()

# Phase 2: Full
self.orch_config = OrchestratorIntegrationConfig.create_live_mode_full()
```

---

## Next Steps

1. **Monitor observations** for 24-48 hours
2. **Validate policy logic** against real market data
3. **Tune policy parameters** if needed
4. **Begin gradual rollout** (signal filtering first)
5. **Monitor each phase** before enabling next feature

---

## Files Created/Modified

### New Files:
- `backend/services/policy_observer.py` - Observation logger
- `backend/services/orchestrator_config.py` - Integration config
- `ORCHESTRATOR_OBSERVATION_MODE.md` - This guide

### Modified Files:
- `backend/services/event_driven_executor.py` - Integrated observation mode

### Log Directories:
- `data/policy_observations/` - Policy and signal logs (auto-created)

---

## Status: ✅ OBSERVATION MODE ACTIVE

The orchestrator is now computing policies and logging decisions without affecting trading behavior. All systems nominal.
