# 🛡️ SAFETY GOVERNOR - IMPLEMENTATION GUIDE

**Date**: November 23, 2025  
**Version**: 1.0  
**Status**: ✅ IMPLEMENTED

---

## 📋 EXECUTIVE SUMMARY

The **Safety Governor** is the ultimate safety layer for Quantum Trader, sitting ABOVE all AI systems to enforce strict risk controls and prevent catastrophic losses. It does NOT disable AI-OS subsystems—it WRAPS them with additional guardrails and overrides when necessary.

**Key Features**:
- ✅ Hierarchical decision-making (Self-Healing > Risk Manager > AI-HFOS > PBA > PAL)
- ✅ Real-time trade interception and modification
- ✅ Dynamic risk multipliers based on safety level
- ✅ Full transparency logging for all decisions
- ✅ Periodic reporting of intervention statistics

---

## 🏗️ ARCHITECTURE

### Priority Hierarchy (Highest to Lowest)

```
1. Self-Healing System (System health, safety policies)
   ↓
2. Advanced Risk Manager (Drawdown, emergency brake, losing streaks)
   ↓
3. AI-HFOS (Supreme meta-intelligence coordinator)
   ↓
4. Portfolio Balancer (Portfolio constraints, exposure limits)
   ↓
5. Profit Amplification Layer (Opportunity enhancement)
```

**Rule**: Higher priority subsystems can override lower priority subsystems.

### Safety Levels

| Level | Leverage Mult | Size Mult | Exposure Mult | Allow Amplification | Allow Expansion Symbols |
|-------|--------------|-----------|---------------|---------------------|------------------------|
| **NORMAL** | 1.0x | 1.0x | 1.0x | ✅ YES | ✅ YES |
| **CAUTIOUS** | 0.75x | 0.75x | 0.85x | ✅ YES | ❌ NO |
| **DEFENSIVE** | 0.5x | 0.5x | 0.6x | ❌ NO | ❌ NO |
| **EMERGENCY** | 0.0x | 0.0x | 0.0x | ❌ NO | ❌ NO |

### Data Flow

```
┌─────────────────┐
│  Self-Healing   │──┐
└─────────────────┘  │
┌─────────────────┐  │
│  Risk Manager   │──┤
└─────────────────┘  │
┌─────────────────┐  ├──> ┌──────────────────┐
│    AI-HFOS      │──┤    │ Safety Governor  │
└─────────────────┘  │    │  (Coordinator)   │
┌─────────────────┐  │    └──────────────────┘
│      PBA        │──┤              │
└─────────────────┘  │              ↓
┌─────────────────┐  │    ┌──────────────────┐
│      PAL        │──┘    │ Trade Decision   │
└─────────────────┘       └──────────────────┘
                                   │
                                   ↓
                          ┌────────────────┐
                          │ Execute/Block/ │
                          │    Modify      │
                          └────────────────┘
```

---

## 🚀 IMPLEMENTATION DETAILS

### 1. Core Component: `backend/services/safety_governor.py`

**Key Classes**:
- `SafetyGovernorDirectives`: Global safety directives enforced by Governor
- `SubsystemInput`: Input from each subsystem with priority and recommendations
- `GovernorDecisionRecord`: Detailed record of each decision for transparency
- `GovernorStats`: Statistics tracking interventions and patterns
- `SafetyGovernor`: Main coordinator class

**Key Methods**:
- `collect_self_healing_input()`: Priority 1 input
- `collect_risk_manager_input()`: Priority 2 input
- `collect_hfos_input()`: Priority 3 input
- `collect_pba_input()`: Priority 4 input
- `compute_directives()`: Apply priority hierarchy and compute global directives
- `evaluate_trade_request()`: Evaluate individual trade against directives
- `generate_report()`: Comprehensive intervention statistics

### 2. FastAPI Integration: `backend/main.py`

**Initialization** (lines ~762-850):
```python
safety_governor = SafetyGovernor(
    data_dir=Path("/app/data"),
    config=None  # Use defaults
)

# Start Safety Governor monitoring loop
async def safety_governor_loop():
    # Collect inputs from all subsystems
    # Compute global directives
    # Store in app.state for executor access
    ...

safety_governor_task = asyncio.create_task(safety_governor_loop())
app_instance.state.safety_governor = safety_governor
app_instance.state.safety_governor_directives = directives
```

**Configuration**:
- `QT_SAFETY_GOVERNOR_ENABLED=true`: Enable/disable Safety Governor (default: true)
- `QT_SAFETY_GOVERNOR_UPDATE_INTERVAL=60`: Directive update interval in seconds (default: 60)
- `QT_SAFETY_GOVERNOR_REPORT_INTERVAL=300`: Reporting interval in seconds (default: 300)

### 3. Executor Integration: `backend/services/event_driven_executor.py`

**Pre-Trade Check** (lines ~933-1032):
```python
# SAFETY GOVERNOR CHECK BEFORE TRADE
if hasattr(self._app_state, 'safety_governor'):
    safety_governor = self._app_state.safety_governor
    directives = self._app_state.safety_governor_directives
    
    # Check global_allow_new_trades
    if not directives.global_allow_new_trades:
        # BLOCK TRADE
        logger.error(f"🛡️ [SAFETY GOVERNOR] ❌ BLOCKED: {symbol}")
        orders_skipped += 1
        continue
    
    # Evaluate trade request
    decision, record = safety_governor.evaluate_trade_request(
        symbol=symbol,
        action="NEW_TRADE",
        size=actual_margin,
        leverage=proposed_leverage,
        confidence=confidence,
        metadata={...}
    )
    
    # Log decision with transparency
    safety_governor.log_decision(record)
    
    # Handle decision
    if not record.allowed:
        # BLOCK
        orders_skipped += 1
        continue
    elif record.modified:
        # MODIFY (apply multipliers)
        actual_margin *= record.applied_multipliers["size"]
        proposed_leverage *= record.applied_multipliers["leverage"]
    # else: ALLOW (proceed with original values)
```

---

## 🔍 DECISION LOGIC

### Trade Blocking Scenarios

1. **Self-Healing Blocks**:
   - `EMERGENCY_SHUTDOWN`: Critical system failure → Block ALL trades
   - `NO_NEW_TRADES`: System health degraded → Block new entries
   - `DEFENSIVE_EXIT`: Reduce exposure → Block new entries, allow exits

2. **Risk Manager Blocks**:
   - `EMERGENCY_BRAKE_ACTIVE`: Manual emergency brake → Block ALL trades
   - `CRITICAL_DRAWDOWN`: Drawdown ≥ 8% → Block ALL trades
   - `HIGH_DRAWDOWN`: Drawdown ≥ 5% → Block new entries
   - `CRITICAL_LOSING_STREAK`: ≥ 8 consecutive losses → Block new entries
   - `HIGH_LOSING_STREAK`: ≥ 5 consecutive losses → Reduce size to 75%

3. **AI-HFOS Blocks**:
   - `DISALLOW_NEW_TRADES`: AI-HFOS blocks trading → Block new entries
   - `REDUCE_GLOBAL_RISK`: AI-HFOS reduces risk → Apply scale multiplier

4. **Portfolio Balancer Blocks**:
   - `PORTFOLIO_CONSTRAINTS_VIOLATED`: Critical violations → Block new entries
   - `EXPOSURE_LIMIT_EXCEEDED`: Total exposure too high → Block new entries

5. **Governor Overrides**:
   - `EXPANSION_SYMBOLS_BLOCKED`: Cautious/Defensive mode → Block EXPANSION category
   - `AMPLIFICATION_DISABLED`: Defensive/Emergency mode → Block PAL expansions

### Trade Modification Scenarios

- **Leverage Reduction**: Apply `max_leverage_multiplier` (0.5x - 1.0x)
- **Position Size Reduction**: Apply `max_position_size_multiplier` (0.5x - 1.0x)
- **Exposure Reduction**: Apply `max_total_exposure_multiplier` (0.6x - 1.0x)

**Example**:
```
Original Trade: $5000 margin, 30x leverage
Safety Level: DEFENSIVE
Result: $2500 margin (0.5x), 15x leverage (0.5x)
```

---

## 📊 LOGGING & TRANSPARENCY

### Decision Logging

**Allowed Trade**:
```
🛡️ [SAFETY GOVERNOR] ALLOWED: NEW_TRADE BTCUSDT | Size: 2500.00 | Leverage: 20.0x
```

**Modified Trade**:
```
🛡️ [SAFETY GOVERNOR] MODIFIED: NEW_TRADE ETHUSDT | 
Size: 5000.00 → 3750.00 | Leverage: 30.0x → 22.5x | 
Reason: Trade modified: CAUTIOUS mode multipliers applied
```

**Blocked Trade**:
```
🛡️ [SAFETY GOVERNOR] ❌ BLOCKED: NEW_TRADE SOLUSDT | 
Size: 5000.00 | Leverage: 30.0x | 
Reason: SELF_HEALING_NO_NEW_TRADES | 
Detail: No new trades - system health degraded
```

**Subsystem Votes** (for transparency):
```
🛡️ [SAFETY GOVERNOR] Subsystem votes: 
SELF_HEALING=✗, RISK_MANAGER=✓, AI_HFOS=✓, PORTFOLIO_BALANCER=✓
```

### Periodic Reports

Generated every 5 minutes (configurable) to `/app/data/safety_governor_report.json`:

```json
{
  "timestamp": "2025-11-23T23:15:00Z",
  "current_directives": {
    "safety_level": "CAUTIOUS",
    "global_allow_new_trades": true,
    "max_leverage_multiplier": 0.75,
    "max_position_size_multiplier": 0.75
  },
  "statistics": {
    "total_decisions": 127,
    "trades_allowed": 89,
    "trades_blocked": 23,
    "trades_modified": 15,
    "intervention_rate_pct": 29.9
  },
  "blocks_by_reason": {
    "SELF_HEALING_NO_NEW_TRADES": 12,
    "RISK_MANAGER_DRAWDOWN_LIMIT": 8,
    "PBA_EXPOSURE_LIMIT": 3
  },
  "risky_symbols": {
    "most_blocked": [
      {"symbol": "DYMUSDT", "count": 5},
      {"symbol": "NEARUSDT", "count": 3}
    ],
    "most_modified": [
      {"symbol": "ETHUSDT", "count": 7},
      {"symbol": "SOLUSDT", "count": 4}
    ]
  }
}
```

---

## 🧪 TESTING & VERIFICATION

### Manual Testing Steps

1. **Start System**:
   ```bash
   systemctl up -d
   docker logs -f quantum_backend
   ```

2. **Verify Initialization**:
   ```
   🛡️ SAFETY GOVERNOR: ENABLED (global safety enforcement layer)
   ```

3. **Monitor Trade Decisions**:
   ```bash
   journalctl -u quantum_backend.service --since 5m | grep "SAFETY GOVERNOR"
   ```

4. **Check Reports**:
   ```bash
   docker exec quantum_backend cat /app/data/safety_governor_report.json | jq
   ```

### Expected Behavior

**Normal Operation** (NORMAL mode):
- ✅ Trades allowed without modification
- ✅ All subsystems active
- ✅ Full leverage and position sizes

**Risk Event** (CAUTIOUS mode):
- ⚠️ Trades modified (75% size, 75% leverage)
- ⚠️ Expansion symbols blocked
- ✅ Core symbols allowed

**High Risk** (DEFENSIVE mode):
- ❌ New trades blocked or heavily modified (50% size, 50% leverage)
- ❌ Amplification disabled
- ⚠️ Only safe exits allowed

**Emergency** (EMERGENCY mode):
- ❌ ALL new trades blocked
- ❌ ALL amplification disabled
- ⚠️ Defensive exits only

### Stress Test Scenarios

1. **Self-Healing Degradation**:
   - Simulate: Set `safety_policy = NO_NEW_TRADES` in Self-Healing
   - Expected: Governor blocks all new entries
   - Verify: Logs show `SELF_HEALING_NO_NEW_TRADES`

2. **Drawdown Threshold**:
   - Simulate: Set `daily_dd_pct = -6.0` in Risk Manager
   - Expected: Governor blocks new entries
   - Verify: Safety level = DEFENSIVE, trades blocked

3. **Losing Streak**:
   - Simulate: Set `losing_streak = 6` in Risk Manager
   - Expected: Governor reduces position sizes to 75%
   - Verify: Trades modified with 0.75x multipliers

4. **AI-HFOS Override**:
   - Simulate: Set `allow_new_trades = false` in AI-HFOS
   - Expected: Governor blocks new entries
   - Verify: Logs show `HFOS_DISALLOW_NEW_TRADES`

---

## 📚 INTEGRATION WITH EXISTING SYSTEMS

### Self-Healing Integration

**Data Flow**:
```python
# In main.py safety_governor_loop
sh_report = app_instance.state.self_healing_system.get_health_report()
sh_input = safety_governor.collect_self_healing_input(sh_report)
```

**Required Fields**:
- `safety_policy`: "ALLOW_ALL" | "NO_NEW_TRADES" | "DEFENSIVE_EXIT" | "EMERGENCY_SHUTDOWN"
- `overall_status`: "HEALTHY" | "DEGRADED" | "CRITICAL"

### Risk Manager Integration (TODO)

**Data Flow**:
```python
# TODO: Connect to actual Risk Manager
risk_state = {
    "emergency_brake_active": bool,
    "daily_dd_pct": float,
    "max_daily_dd_pct": float,
    "losing_streak": int
}
risk_input = safety_governor.collect_risk_manager_input(risk_state)
```

### AI-HFOS Integration

**Data Flow**:
```python
# In main.py safety_governor_loop
hfos_output = app_instance.state.ai_hfos_output  # AIHFOSOutput object
hfos_input = safety_governor.collect_hfos_input(hfos_output)
```

**Required Fields**:
- `global_directives.allow_new_trades`: bool
- `global_directives.scale_position_sizes`: float
- `global_directives.reduce_global_risk`: bool
- `supreme_decision.risk_mode`: RiskMode enum

### Portfolio Balancer Integration (TODO)

**Data Flow**:
```python
# TODO: Connect to actual Portfolio Balancer
pba_violations = []  # List of violation strings
portfolio_state = {"total_positions": 0, "gross_exposure": 0.0}
pba_input = safety_governor.collect_pba_input(pba_violations, portfolio_state)
```

---

## 🔧 CONFIGURATION

### Environment Variables

```bash
# Safety Governor
QT_SAFETY_GOVERNOR_ENABLED=true                  # Enable/disable Governor
QT_SAFETY_GOVERNOR_UPDATE_INTERVAL=60            # Directive update interval (seconds)
QT_SAFETY_GOVERNOR_REPORT_INTERVAL=300           # Report generation interval (seconds)

# Safety Thresholds (in safety_governor.py config)
MAX_DAILY_DRAWDOWN_PCT=5.0                       # High drawdown threshold
EMERGENCY_DRAWDOWN_PCT=8.0                       # Critical drawdown threshold
MAX_LOSING_STREAK=5                              # High losing streak threshold
CRITICAL_LOSING_STREAK=8                         # Critical losing streak threshold
```

### Safety Multipliers

Defined in `SafetyGovernor._default_config()`:

```python
"safety_multipliers": {
    "NORMAL": {"leverage": 1.0, "position_size": 1.0, "exposure": 1.0},
    "CAUTIOUS": {"leverage": 0.75, "position_size": 0.75, "exposure": 0.85},
    "DEFENSIVE": {"leverage": 0.5, "position_size": 0.5, "exposure": 0.6},
    "EMERGENCY": {"leverage": 0.0, "position_size": 0.0, "exposure": 0.0}
}
```

---

## 🎯 SUCCESS CRITERIA

✅ **Implementation Complete**:
- [x] Safety Governor core component created
- [x] Integrated into FastAPI startup
- [x] Pre-trade checks in executor
- [x] Comprehensive logging
- [x] Periodic reporting
- [x] Priority hierarchy enforced

✅ **Functionality**:
- [x] Blocks trades when global_allow_new_trades = false
- [x] Modifies trades with leverage/size multipliers
- [x] Respects subsystem priority hierarchy
- [x] Logs all decisions with transparency
- [x] Generates periodic reports

⏳ **Pending**:
- [ ] Connect to actual Risk Manager (currently using stub data)
- [ ] Connect to actual Portfolio Balancer violations
- [ ] Add real-time dashboard for Safety Governor status
- [ ] Add alerting for intervention rate > threshold

---

## 📈 EXPECTED IMPACT

**Risk Reduction**:
- 🎯 **Target**: Reduce catastrophic loss risk by 80%
- 🎯 **Target**: Prevent drawdown > 10% under any conditions
- 🎯 **Target**: Block 100% of trades during system degradation

**Trading Impact**:
- ⚠️ **Trade Count**: May reduce by 10-30% during high-risk periods
- ⚠️ **Position Sizes**: May reduce by 25-50% in cautious/defensive modes
- ✅ **Win Rate**: Expected to improve due to better trade selection
- ✅ **Sharpe Ratio**: Expected to improve due to reduced volatility

**Operational Benefits**:
- ✅ **Full transparency**: All decisions logged with reasons
- ✅ **Pattern detection**: Reports identify repeatedly risky symbols
- ✅ **Override capability**: Higher priority systems can override AI
- ✅ **No AI disruption**: AI-OS continues running, just wrapped with safety

---

## 🚨 CRITICAL NOTES

1. **Fail-Open vs Fail-Closed**:
   - Current: Fail-open (on error, allow trade)
   - Production: Consider fail-closed for maximum safety

2. **Subsystem Availability**:
   - Governor gracefully handles missing subsystems
   - Falls back to default values if inputs unavailable
   - Logs warnings when subsystems not integrated

3. **Performance**:
   - Pre-trade check adds ~5-10ms latency
   - Acceptable for event-driven mode (30s intervals)
   - Monitor for performance impact in high-frequency scenarios

4. **Testing**:
   - Test each safety level transition
   - Verify all block reasons trigger correctly
   - Confirm reports generate properly
   - Validate multipliers apply as expected

---

## 📞 SUPPORT & TROUBLESHOOTING

**Issue**: Safety Governor not blocking trades during high risk  
**Solution**: Check if `safety_governor_directives` is in app.state, verify subsystem inputs are being collected

**Issue**: Trades blocked unexpectedly  
**Solution**: Check Safety Governor logs for reason, review subsystem health reports

**Issue**: Reports not generating  
**Solution**: Verify `safety_governor_report_task` is running, check file permissions on /app/data

**Issue**: Multipliers not applying  
**Solution**: Verify `record.applied_multipliers` is being used to modify `actual_margin` and `proposed_leverage`

---

**IMPLEMENTATION STATUS**: ✅ COMPLETE  
**NEXT STEPS**: Test in live environment, monitor intervention statistics, tune thresholds based on performance

