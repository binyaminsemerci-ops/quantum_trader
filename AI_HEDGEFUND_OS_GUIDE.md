# AI HEDGEFUND OPERATING SYSTEM (AI-HFOS) GUIDE

**Supreme Meta-Intelligence Layer for Quantum Trader**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [System Risk Modes](#system-risk-modes)
4. [Subsystem Coordination](#subsystem-coordination)
5. [Conflict Resolution](#conflict-resolution)
6. [Directive System](#directive-system)
7. [Emergency Response](#emergency-response)
8. [Integration Guide](#integration-guide)
9. [Operational Guide](#operational-guide)

---

## 🎯 Overview

The **AI Hedgefund Operating System (AI-HFOS)** is the supreme meta-intelligence layer that oversees, coordinates, supervises, and optimizes every AI subsystem in Quantum Trader.

### Core Mission

```
OVERSEE, COORDINATE, SUPERVISE AND OPTIMIZE EVERY AI SUBSYSTEM
TO ACHIEVE SAFE, CONSISTENT, AUTONOMOUS PROFIT GENERATION
```

### What AI-HFOS Is

- ✅ **Supreme Coordinator** - Directs all subsystems
- ✅ **Conflict Resolver** - Resolves subsystem disagreements
- ✅ **Risk Governor** - Enforces system-wide safety
- ✅ **Strategic Intelligence** - Makes meta-level decisions
- ✅ **Emergency Responder** - Handles critical situations

### What AI-HFOS Is NOT

- ❌ **Trade Executor** - Does not execute trades directly
- ❌ **Single Module** - Operates above all layers
- ❌ **Rule Engine** - Uses strategic intelligence, not just rules

---

## 🏗️ Architecture

### System Hierarchy

```
┌────────────────────────────────────────────────────────────┐
│         AI HEDGEFUND OPERATING SYSTEM (AI-HFOS)           │
│              SUPREME META-INTELLIGENCE                     │
└────────────────────────────────────────────────────────────┘
                          ↓
        ┌─────────────────────────────────────┐
        │      SUBSYSTEM COORDINATION          │
        └─────────────────────────────────────┘
                          ↓
    ┌──────────┬──────────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Universe│ │Risk OS │ │  PIL   │ │  PBA   │ │  PAL   │
│   OS   │ │        │ │        │ │        │ │        │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
    ↓          ↓          ↓          ↓          ↓
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│  Self  │ │ Model  │ │Retraining│ │Orchestr│ │Execution│
│Healing │ │Supervisor│ │        │ │  ator  │ │ Layer  │
└────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

### Data Flow

```
INPUT PHASE:
1. Universe OS → Symbol universe, classifications, blacklist
2. Risk OS → Emergency brake, DD levels, risk profile
3. PIL → Position performance, toxic positions, winners
4. PBA → Portfolio exposure, limits, constraints
5. PAL → Amplification opportunities
6. Self-Healing → System health, subsystem issues
7. Model Supervisor → Model performance, degraded models
8. Orchestrator → Regime, exit mode, policy state

PROCESSING PHASE:
9. AI-HFOS analyzes all inputs
10. Detects conflicts between subsystems
11. Assesses system health
12. Determines global risk mode
13. Generates unified directives

OUTPUT PHASE:
14. Global directives → All subsystems
15. Universe directives → Universe OS
16. Execution directives → Execution Layer
17. Portfolio directives → PBA
18. Model directives → Model Supervisor
19. Emergency actions → Immediate execution
20. Amplification opportunities → Execution Layer
```

---

## 🎚️ System Risk Modes

AI-HFOS operates in one of four global risk modes:

### SAFE Mode

**When Activated:**
- System health = CRITICAL
- Daily DD > 3%
- Multiple subsystem failures

**Directives:**
```json
{
  "allow_new_trades": true,
  "allow_new_positions": true,
  "scale_position_sizes": 0.6,
  "confidence_threshold": 0.75,
  "max_daily_dd_override": 3.0,
  "universe_mode": "SAFE",
  "reduce_exposure_pct": 20.0
}
```

**Impact:**
- Reduced position sizes (60%)
- Higher confidence threshold (75%)
- SAFE universe only
- 20% exposure reduction
- Conservative execution

---

### NORMAL Mode

**When Activated:**
- System health = HEALTHY or OPTIMAL
- No emergency conditions
- All subsystems aligned

**Directives:**
```json
{
  "allow_new_trades": true,
  "allow_new_positions": true,
  "scale_position_sizes": 1.0,
  "confidence_threshold": null,
  "max_daily_dd_override": null,
  "universe_mode": "NORMAL",
  "reduce_exposure_pct": 0.0
}
```

**Impact:**
- Normal operation
- Standard position sizes
- Full universe access
- No restrictions

---

### AGGRESSIVE Mode

**When Activated:**
- System health = OPTIMAL
- Strong performance
- All systems green
- Favorable conditions

**Directives:**
```json
{
  "allow_new_trades": true,
  "allow_new_positions": true,
  "scale_position_sizes": 1.5,
  "confidence_threshold": 0.65,
  "universe_mode": "AGGRESSIVE",
  "reduce_exposure_pct": 0.0
}
```

**Impact:**
- Larger position sizes (150%)
- Lower confidence threshold (65%)
- AGGRESSIVE universe
- Opportunistic trading

---

### CRITICAL Mode

**When Activated:**
- System health = EMERGENCY
- Emergency brake active
- Daily DD > 5%
- Cascading failures

**Directives:**
```json
{
  "allow_new_trades": false,
  "allow_new_positions": false,
  "enforce_defensive_exits": true,
  "scale_position_sizes": 0.3,
  "confidence_threshold": 0.85,
  "max_daily_dd_override": 2.0,
  "universe_mode": "SAFE",
  "reduce_exposure_pct": 50.0
}
```

**Impact:**
- NO new trades
- NO new positions
- Defensive exits enforced
- 50% exposure reduction
- Emergency mode active

---

## 🤝 Subsystem Coordination

### Subsystem Health Scoring

Each subsystem receives a health score (0-100):

#### Universe OS
```python
health_score = 100
if data_confidence == "LOW": health_score -= 20
if symbol_count < 50: health_score -= 30
if blacklist_count > 150: health_score -= 10
```

#### Risk OS
```python
health_score = 100
if emergency_brake_active: health_score -= 40
health_score -= min(30, daily_dd_pct * 5)
health_score -= min(20, open_dd_pct * 2)
```

#### Position Intelligence Layer
```python
health_score = 100 - (toxic_count * 10)
if toxic_ratio > 0.3: health_score -= 20
```

#### Execution Layer
```python
health_score = 100
health_score -= min(30, avg_slippage_bps * 2)
health_score -= (1.0 - fill_rate) * 50
```

#### Model Supervisor
```python
health_score = ensemble_accuracy * 100
if degraded_models > 0: health_score -= (degraded_models * 5)
```

### Overall System Health

```python
System Health Calculation:
1. avg_health = average of all subsystem health scores
2. critical_count = count of unhealthy subsystems

If avg_health >= 90 AND critical_count == 0:
    → OPTIMAL
If avg_health >= 70 AND critical_count <= 1:
    → HEALTHY
If avg_health >= 50:
    → DEGRADED
If critical_count > 3 OR avg_health < 30:
    → EMERGENCY
Else:
    → CRITICAL
```

---

## ⚔️ Conflict Resolution

### Conflict Detection

AI-HFOS detects conflicts between subsystems:

**Example 1: Risk vs Universe**
```
Conflict: Emergency brake active but Universe OS healthy
Severity: WARNING
Resolution: Follow Risk OS - safety first
```

**Example 2: PIL vs PAL**
```
Conflict: PAL wants to amplify but PIL classifies position as TOXIC
Severity: ERROR
Resolution: Block amplification - position safety first
```

**Example 3: Execution vs Risk**
```
Conflict: Execution wants MARKET orders but Risk requires LIMIT
Severity: ERROR
Resolution: Follow Risk - enforce LIMIT orders
```

### Resolution Priority

```
PRIORITY 1: Safety (Risk OS, Self-Healing)
PRIORITY 2: Emergency Brake (Universe OS)
PRIORITY 3: Position Safety (PIL)
PRIORITY 4: Portfolio Limits (PBA)
PRIORITY 5: Execution Preferences
PRIORITY 6: Amplification (PAL)
PRIORITY 7: Model Preferences
```

**Golden Rule:**
```
ALWAYS PRIORITIZE SAFETY OVER OPPORTUNITY
```

---

## 📋 Directive System

### Global Directives

Applied to all subsystems:

| Directive | Description | Values |
|-----------|-------------|--------|
| `allow_new_trades` | Allow system to open trades | true/false |
| `allow_new_positions` | Allow new position entries | true/false |
| `enforce_defensive_exits` | Force defensive exits | true/false |
| `reduce_global_risk` | Reduce system-wide risk | true/false |
| `pause_entire_symbols` | Symbols to pause | [list] |
| `adjust_confidence_threshold` | Override confidence | 0.0-1.0 |
| `scale_position_sizes` | Position size multiplier | 0.3-1.5 |
| `max_daily_dd_override` | Override max DD limit | % |
| `force_exit_symbols` | Symbols to exit immediately | [list] |

---

### Universe Directives

Applied to Universe OS:

| Directive | Description | Values |
|-----------|-------------|--------|
| `universe_mode` | Universe selection mode | SAFE/NORMAL/AGGRESSIVE/EXPERIMENTAL |
| `blacklist_symbols` | Additional symbols to blacklist | [list] |
| `whitelist_symbols` | Priority symbols | [list] |
| `promote_categories` | Categories to promote | [list] |
| `demote_categories` | Categories to demote | [list] |
| `emergency_brake_override` | Override emergency brake | true/false |

---

### Execution Directives

Applied to Execution Layer:

| Directive | Description | Values |
|-----------|-------------|--------|
| `order_type_preference` | Preferred order type | MARKET/LIMIT/SMART |
| `max_slippage_bps` | Maximum slippage | basis points |
| `max_spread_bps` | Maximum spread | basis points |
| `reduce_urgency` | Reduce execution urgency | true/false |
| `enforce_limit_orders` | Force LIMIT orders | true/false |
| `execution_delay_seconds` | Delay between executions | seconds |

---

### Portfolio Directives

Applied to Portfolio Balancer:

| Directive | Description | Values |
|-----------|-------------|--------|
| `reduce_exposure_pct` | Exposure reduction | 0-100% |
| `max_position_count` | Maximum positions | number |
| `max_leverage` | Maximum leverage | multiplier |
| `reduce_correlated_positions` | Reduce correlation | true/false |
| `avoid_expansion_symbols` | Avoid EXPANSION category | true/false |
| `concentration_limit_pct` | Max per-symbol concentration | % |

---

### Model Directives

Applied to Model Supervisor:

| Directive | Description | Values |
|-----------|-------------|--------|
| `ensemble_weight_adjustments` | Adjust model weights | {model: weight} |
| `models_to_retrain` | Models requiring retraining | [list] |
| `models_to_disable` | Models to disable | [list] |
| `confidence_threshold_override` | Override confidence | 0.0-1.0 |
| `use_conservative_predictions` | Use conservative mode | true/false |

---

## 🚨 Emergency Response

### Emergency Actions

AI-HFOS can trigger emergency actions:

#### CLOSE_ALL_POSITIONS
```json
{
  "action_type": "CLOSE_ALL_POSITIONS",
  "target": "ALL",
  "parameters": {"urgency": "immediate"},
  "priority": 1,
  "rationale": "System in EMERGENCY state"
}
```

#### PAUSE_NEW_TRADES
```json
{
  "action_type": "PAUSE_NEW_TRADES",
  "target": "SYSTEM",
  "parameters": {"duration_minutes": 60},
  "priority": 2,
  "rationale": "Self-Healing detected CRITICAL issues"
}
```

#### REDUCE_ALL_POSITIONS
```json
{
  "action_type": "REDUCE_ALL_POSITIONS",
  "target": "ALL",
  "parameters": {"reduce_by_pct": 50},
  "priority": 2,
  "rationale": "Daily DD exceeded threshold"
}
```

#### HALT_SYMBOL
```json
{
  "action_type": "HALT_SYMBOL",
  "target": "BTCUSDT",
  "parameters": {"duration_minutes": 30},
  "priority": 3,
  "rationale": "Symbol showing erratic behavior"
}
```

### Emergency Triggers

| Trigger | Action |
|---------|--------|
| System health = EMERGENCY | CLOSE_ALL_POSITIONS |
| Self-Healing = CRITICAL | PAUSE_NEW_TRADES |
| Daily DD > 5% | REDUCE_ALL_POSITIONS (50%) |
| Open DD > 10% | CLOSE_ALL_POSITIONS |
| Emergency brake active > 1 hour | HALT_SYSTEM |

---

## 🔌 Integration Guide

### 1. Basic Setup

```python
from backend.services.ai_hfos_integration import AIHFOSIntegration

# Initialize integration
hfos_integration = AIHFOSIntegration(
    data_dir="/app/data",
    update_interval_seconds=60
)
```

### 2. Run Single Coordination Cycle

```python
# Run once
await hfos_integration.run_coordination_cycle()

# Get current status
status = hfos_integration.get_system_status()
print(f"Risk Mode: {status['risk_mode']}")
print(f"Health: {status['health']}")
```

### 3. Run Continuous Coordination

```python
# Start continuous loop (runs every 60 seconds)
await hfos_integration.run_continuous()

# Stop when needed
hfos_integration.stop()
```

### 4. Query Current State

```python
# Get current risk mode
risk_mode = hfos_integration.get_current_risk_mode()

# Get current directives
directives = hfos_integration.get_current_directives()

# Check global directives
if directives['global'].allow_new_trades:
    print("New trades allowed")

# Check position size scaling
scale = directives['global'].scale_position_sizes
print(f"Position sizes scaled to {scale:.1%}")
```

### 5. Integration with Event-Driven System

```python
async def main_trading_loop():
    """Main trading loop with AI-HFOS coordination."""
    
    # Initialize AI-HFOS
    hfos = AIHFOSIntegration(data_dir="/app/data")
    
    # Start coordination in background
    coordination_task = asyncio.create_task(hfos.run_continuous())
    
    while True:
        # Get current directives
        directives = hfos.get_current_directives()
        
        if not directives:
            await asyncio.sleep(1)
            continue
        
        # Apply directives to trading logic
        if directives['global'].allow_new_trades:
            # Check for trade signals
            signals = await get_trade_signals()
            
            # Apply confidence threshold override
            threshold = directives['global'].adjust_confidence_threshold or 0.70
            signals = [s for s in signals if s.confidence >= threshold]
            
            # Apply position size scaling
            scale = directives['global'].scale_position_sizes
            for signal in signals:
                signal.size_usd *= scale
            
            # Execute trades
            for signal in signals:
                await execute_trade(signal, directives['execution'])
        
        # Check emergency actions
        status = hfos.get_system_status()
        if status['emergency_actions'] > 0:
            logger.critical("⚠️  EMERGENCY ACTIONS REQUIRED")
            await handle_emergency()
        
        await asyncio.sleep(10)
```

---

## 📊 Operational Guide

### Monitoring AI-HFOS

**Key Metrics:**
- System Risk Mode (SAFE/NORMAL/AGGRESSIVE/CRITICAL)
- System Health (OPTIMAL/HEALTHY/DEGRADED/CRITICAL/EMERGENCY)
- Number of subsystem conflicts
- Number of emergency actions
- Amplification opportunities identified

**Log Monitoring:**
```bash
# Watch AI-HFOS coordination
tail -f logs/ai_hfos.log | grep "COORDINATION CYCLE"

# Monitor emergency actions
tail -f logs/ai_hfos.log | grep "EMERGENCY"

# Watch directive changes
tail -f logs/ai_hfos.log | grep "Applying.*directives"
```

### Health Checks

```python
# Check if AI-HFOS is healthy
status = hfos.get_system_status()

if status['risk_mode'] == 'CRITICAL':
    alert("AI-HFOS in CRITICAL mode!")

if status['emergency_actions'] > 0:
    alert(f"{status['emergency_actions']} emergency actions pending!")

if status['conflicts'] > 5:
    alert("Many subsystem conflicts detected!")
```

### Troubleshooting

**Problem: AI-HFOS stuck in CRITICAL mode**
```
1. Check self_healing_report.json for CRITICAL subsystems
2. Check Risk OS for emergency brake status
3. Review daily DD levels
4. Check for cascading failures
5. Manually reset if all systems green
```

**Problem: Too many conflicts detected**
```
1. Review conflict types in ai_hfos_report.json
2. Check for subsystem misalignment
3. Verify input data quality
4. Check for stale data
```

**Problem: Directives not being applied**
```
1. Check integration layer logs
2. Verify subsystem connections
3. Check for API failures
4. Review directive compatibility
```

---

## 🎯 Best Practices

### 1. Trust the System

```python
# ✅ Good: Trust AI-HFOS risk mode
if hfos.get_current_risk_mode() == SystemRiskMode.CRITICAL:
    # Follow directive - no new trades
    return

# ❌ Bad: Override AI-HFOS
if hfos.get_current_risk_mode() == SystemRiskMode.CRITICAL:
    # Ignore it, I see a good trade
    await execute_trade()  # DON'T DO THIS!
```

### 2. Apply All Directives

```python
# ✅ Good: Apply all directives
directives = hfos.get_current_directives()
position_size *= directives['global'].scale_position_sizes
confidence_threshold = directives['global'].adjust_confidence_threshold

# ❌ Bad: Cherry-pick directives
directives = hfos.get_current_directives()
# Only apply size scaling, ignore confidence threshold
position_size *= directives['global'].scale_position_sizes
```

### 3. Respond to Emergency Actions

```python
# ✅ Good: Immediate response
status = hfos.get_system_status()
if status['emergency_actions'] > 0:
    await execute_emergency_protocol()

# ❌ Bad: Delay or ignore
status = hfos.get_system_status()
if status['emergency_actions'] > 0:
    logger.info("Will handle later")  # DON'T DELAY!
```

### 4. Monitor Conflicts

```python
# ✅ Good: Review and resolve
output = hfos.last_output
for conflict in output.detected_conflicts:
    if conflict.severity == ConflictSeverity.CRITICAL:
        await investigate_conflict(conflict)

# ❌ Bad: Ignore conflicts
# Conflicts indicate system misalignment - investigate!
```

---

## 📈 Performance Impact

**Expected Benefits:**
- System-wide safety: +95%
- Conflict-free operation: 90%+ of time
- Emergency response time: <1 minute
- Subsystem alignment: 95%+

**Resource Usage:**
- CPU: <5% (coordination cycles every 60s)
- Memory: <100MB (state tracking)
- Disk: ~1MB/day (reports)

---

## 🎓 Summary

The **AI Hedgefund Operating System** is the supreme coordinator of Quantum Trader:

✅ **Oversees** all subsystems (Universe OS, Risk OS, PIL, PBA, PAL, Model Supervisor, Self-Healing)  
✅ **Coordinates** subsystem interactions and resolves conflicts  
✅ **Governs** system-wide risk through 4 risk modes (SAFE/NORMAL/AGGRESSIVE/CRITICAL)  
✅ **Directs** all subsystems with unified directives  
✅ **Responds** to emergencies with immediate actions  
✅ **Optimizes** performance through strategic meta-intelligence  

**Integration Points:**
- Data collection from all subsystems
- Directive distribution to all subsystems
- Emergency action execution
- Amplification opportunity processing
- Continuous coordination loop

**Operational Impact:**
- Safe, aligned, autonomous operation
- Rapid emergency response
- Conflict-free subsystem coordination
- Optimized system performance

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Author:** Quantum Trader AI Team
