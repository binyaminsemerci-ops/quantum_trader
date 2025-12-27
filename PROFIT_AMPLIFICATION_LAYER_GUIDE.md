# PROFIT AMPLIFICATION LAYER (PAL) GUIDE

**Complete Integration & Reference Documentation**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Amplification Scoring](#amplification-scoring)
4. [Amplification Techniques](#amplification-techniques)
5. [Risk Management](#risk-management)
6. [Integration Guide](#integration-guide)
7. [Testing](#testing)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

The **Profit Amplification Layer (PAL)** increases average R and total profit by smartly extending, scaling, and enhancing high-quality trading opportunities.

### Core Philosophy

```
PAL DOES NOT CREATE NEW TRADES
PAL AMPLIFIES EXISTING WINNERS
```

### Mission

**Increase average R and total profit** by:
- ✅ Identifying amplifiable positions
- ✅ Recommending safe amplification techniques
- ✅ Protecting capital through risk-aware decisions

### Design Principles

1. **Advisory, Not Autonomous** - PAL recommends, doesn't execute
2. **Risk-First** - Never amplify when risky
3. **Winner Enhancement** - Only amplify quality trades
4. **Transparent Logic** - Clear rationale for every decision

---

## 🏗️ Architecture

### System Components

```
┌──────────────────────────────────────────────────────────┐
│              PROFIT AMPLIFICATION LAYER (PAL)            │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Candidate  │→ │Amplification │→ │ Recommend-    │  │
│  │Identification│  │   Scoring    │  │   ations      │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│         ↓                 ↓                  ↓           │
│  ┌──────────────────────────────────────────────────┐   │
│  │          Risk-Aware Decision Engine              │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
           ↑                    ↓
    ┌──────────┐        ┌──────────────┐
    │Position  │        │  Execution   │
    │Intel (PIL)│        │   Layer      │
    └──────────┘        └──────────────┘
```

### Data Flow

```
1. PIL Classifications → PAL
2. PAL analyzes positions
3. PAL identifies amplifiable candidates
4. PAL generates recommendations
5. Recommendations → Execution Layer
6. Execution Layer decides to act or not
```

---

## 🎯 Amplification Scoring

### Scoring Algorithm (0-100 points)

PAL calculates an amplification score for each position:

```python
Total Score = R_Score + Trend_Score + DD_Score + Rank_Score + Vol_Score
```

#### Component Scores

**1. R Score (30% weight, 0-30 points)**
```python
r_score = min(current_R / 5.0, 1.0) * 30

Examples:
- 0.5R → 3 points
- 1.0R → 6 points
- 2.5R → 15 points
- 5.0R → 30 points (max)
```

**2. Trend Score (25% weight, 0-25 points)**
```python
Trend Strength → Score:
- VERY_STRONG → 25 points (100%)
- STRONG      → 20 points (80%)
- MODERATE    → 12.5 points (50%)
- WEAK        → 5 points (20%)
- NONE        → 0 points (0%)
```

**3. Drawdown Score (20% weight, 0-20 points)**
```python
dd_score = max(1.0 - (DD_from_peak_R / 0.5), 0) * 20

Examples:
- 0% DD → 20 points (max)
- 10% DD → 16 points
- 25% DD → 10 points
- 50% DD → 0 points
```

**4. Symbol Rank Score (15% weight, 0-15 points)**
```python
rank_score = max(1.0 - (symbol_rank / 100), 0) * 15

Examples:
- Rank 1 → 15 points
- Rank 10 → 13.5 points
- Rank 50 → 7.5 points
- Rank 100 → 0 points
```

**5. Volatility Score (10% weight, 0-10 points)**
```python
Volatility Regime → Score:
- LOW    → 10 points (100%)
- NORMAL → 8 points (80%)
- HIGH   → 4 points (40%)
```

### Score Interpretation

| Score Range | Classification | Action |
|-------------|---------------|--------|
| 90-100 | Exceptional | Aggressive amplification |
| 70-89 | High Priority | Standard amplification |
| 50-69 | Medium Priority | Conservative amplification |
| 30-49 | Low Priority | Minimal amplification |
| 0-29 | Not Amplifiable | No action |

---

## 🚀 Amplification Techniques

### 1. SCALE-IN (ADD_SIZE)

**When to Use:**
- Current R ≥ 1.5
- Trend strength = STRONG or VERY_STRONG
- Drawdown from peak ≤ 10%
- Volatility = LOW or NORMAL
- Risk budget available

**Parameters:**
```python
{
    "scale_size_usd": 2000,              # Amount to add
    "max_total_leverage": 12.0,          # Max leverage after scale
    "stop_loss_adjust": "trail_at_breakeven",  # Move SL to breakeven
    "timing": "on_next_confirmation"     # Wait for confirmation
}
```

**Example:**
```
Position: BTCUSDT LONG
Current: $10,000 @ 10x leverage, 2.5R, Trend=STRONG
DD from peak: 5%

Recommendation:
- Add $2,000 (20% of current size)
- New total: $12,000 @ 12x leverage
- Move SL to breakeven
- Expected R increase: +0.10R
```

**Risk Assessment:**
- Risk Score: 4.0/10 (Low-moderate)
- Rationale: Adding to winning position with protective stop

---

### 2. EXTEND_HOLD

**When to Use:**
- Current R ≥ 1.0
- Trend strength = MODERATE or higher
- Drawdown from peak ≤ 15%
- Position has momentum

**Parameters:**
```python
{
    "exit_strategy": "trend_follow",     # Switch to trend-following exit
    "trail_stop_type": "ATR_based",      # Use ATR trailing stop
    "trail_distance_atr": 2.0,           # 2x ATR distance
    "min_hold_additional_hours": 12      # Hold at least 12 more hours
}
```

**Example:**
```
Position: ETHUSDT LONG
Current: $10,000 @ 8x leverage, 1.8R, Trend=VERY_STRONG
DD from peak: 3%

Recommendation:
- Switch from fixed TP to trend-following exit
- Trail stop at 2.0 ATR
- Expected R increase: +1.0R
- Confidence: 65%
```

**Risk Assessment:**
- Risk Score: 2.0/10 (Low)
- Rationale: Protected by trailing stop, riding strong trend

---

### 3. PARTIAL_TAKE_PROFIT

**When to Use:**
- Current R ≥ 2.0
- Peak R ≥ 3.0 (has been to 3R+)
- Symbol category = CORE or TACTICAL
- Lock gains, reduce risk

**Parameters:**
```python
{
    "take_profit_pct": 0.50,            # Take 50% off
    "leave_runner_pct": 0.50,           # Leave 50% as runner
    "runner_exit": "trend_follow",       # Runner exits on trend break
    "lock_profit_r": 1.0                # Lock 1.0R profit
}
```

**Example:**
```
Position: BTCUSDT LONG
Current: $10,000 @ 10x leverage, 2.5R, Peak=3.5R

Recommendation:
- Take 50% profit ($5,000)
- Lock 1.25R
- Leave 50% runner with trend-following exit
- Expected additional R from runner: +0.5R
```

**Risk Assessment:**
- Risk Score: 1.0/10 (Very low)
- Rationale: Locking in gains, reducing exposure

---

### 4. SWITCH_TO_TREND_FOLLOW

**When to Use:**
- Position approaching fixed TP
- Trend still strong
- Want to capture extended move

**Parameters:**
```python
{
    "exit_trigger": "trend_break",       # Exit on trend break
    "trail_method": "structure_based",   # Trail based on structure
    "allow_drawdown_pct": 5.0           # Allow 5% DD before exit
}
```

---

## 🛡️ Risk Management

### Blocking Factors

PAL will **NOT** amplify if any of these are true:

#### 1. High Drawdown
```python
if position.drawdown_from_peak_R > 0.15:  # 15% DD
    BLOCK: AmplificationReason.HIGH_DRAWDOWN
```

#### 2. Weak Trend
```python
if position.trend_strength in [WEAK, NONE]:
    BLOCK: AmplificationReason.WEAK_TREND
```

#### 3. Risk Profile Reduced
```python
if risk_profile == "REDUCED":
    BLOCK: AmplificationReason.RISK_PROFILE_REDUCED
```

#### 4. Emergency Brake Active
```python
if emergency_brake_active:
    BLOCK: AmplificationReason.EMERGENCY_BRAKE_ACTIVE
```

#### 5. Unstable Symbol
```python
if volatility == "HIGH" and symbol_rank > 50:
    BLOCK: AmplificationReason.UNSTABLE_SYMBOL
```

#### 6. Insufficient Risk Budget
```python
if portfolio_risk_budget_pct < 1.0:
    BLOCK: AmplificationReason.INSUFFICIENT_RISK_BUDGET
```

### Risk-Aware Amplification

**Safe Amplification Conditions:**
- ✅ Risk profile = SAFE or AGGRESSIVE
- ✅ Strong regime alignment
- ✅ Portfolio diversification healthy
- ✅ Emergency brake inactive
- ✅ Universe OS confirms symbol stable

**Amplify MORE when:**
- SAFE profile + strong trend
- CORE symbol + low volatility
- Regime alignment + low DD

**Amplify LESS when:**
- AGGRESSIVE profile (already risky)
- TACTICAL/OPPORTUNISTIC symbols
- HIGH volatility regime

---

## 🔌 Integration Guide

### 1. Basic Setup

```python
from backend.services.profit_amplification import (
    ProfitAmplificationLayer,
    PositionSnapshot,
    TrendStrength
)

# Initialize PAL
pal = ProfitAmplificationLayer(
    data_dir="/app/data",
    min_R_for_amplification=1.0,
    min_R_for_scale_in=1.5,
    min_R_for_extend_hold=1.0,
    max_dd_from_peak_pct=15.0,
    max_dd_for_scale_in_pct=10.0,
    min_trend_strength_for_amplification=TrendStrength.MODERATE,
    max_additional_leverage=5.0
)
```

### 2. Create Position Snapshots

```python
from backend.services.position_intelligence import PositionIntelligenceLayer

# Get positions from PIL
pil = PositionIntelligenceLayer()
positions = pil.get_all_positions()

# Convert to PAL format
pal_positions = []
for pos in positions:
    snapshot = PositionSnapshot(
        symbol=pos.symbol,
        side=pos.side,
        current_R=pos.current_R,
        peak_R=pos.peak_R,
        unrealized_pnl=pos.unrealized_pnl,
        unrealized_pnl_pct=pos.unrealized_pnl_pct,
        drawdown_from_peak_R=pos.drawdown_from_peak_R,
        drawdown_from_peak_pnl_pct=pos.drawdown_from_peak_pnl_pct,
        current_leverage=pos.leverage,
        position_size_usd=pos.size_usd,
        risk_pct=pos.risk_pct,
        hold_time_hours=pos.hold_time_hours,
        entry_time=pos.entry_time,
        pil_classification=pos.classification.value,
        trend_strength=TrendStrength.STRONG,  # From technical analysis
        volatility_regime="NORMAL",           # From volatility analysis
        symbol_rank=pos.symbol_rank,          # From Universe OS
        symbol_category=pos.symbol_category   # From Universe OS
    )
    pal_positions.append(snapshot)
```

### 3. Run Analysis

```python
# Get current system state
risk_profile = portfolio_balancer.get_risk_profile()
portfolio_risk_budget = portfolio_balancer.get_available_risk_pct()
regime = orchestrator.get_current_regime()
exit_mode = orchestrator.get_exit_mode()

# Run PAL analysis
report = pal.analyze_positions(
    positions=pal_positions,
    risk_profile=risk_profile,
    portfolio_risk_budget_pct=portfolio_risk_budget,
    regime=regime,
    exit_mode=exit_mode
)

print(f"Candidates: {len(report.amplification_candidates)}")
print(f"Recommendations: {len(report.recommendations)}")
```

### 4. Process Recommendations

```python
# Execute high-priority recommendations
for rec in report.recommendations:
    if rec.priority <= 2 and rec.confidence >= 60:
        
        if rec.action == AmplificationAction.ADD_SIZE:
            # Scale into position
            scale_size = rec.parameters["scale_size_usd"]
            max_leverage = rec.parameters["max_total_leverage"]
            
            await execution_layer.scale_in_position(
                symbol=rec.candidate.position.symbol,
                additional_size_usd=scale_size,
                max_leverage=max_leverage,
                rationale=rec.rationale
            )
        
        elif rec.action == AmplificationAction.EXTEND_HOLD:
            # Switch to trend-following exit
            trail_distance = rec.parameters["trail_distance_atr"]
            
            await execution_layer.switch_exit_strategy(
                symbol=rec.candidate.position.symbol,
                strategy="trend_follow",
                trail_atr=trail_distance,
                rationale=rec.rationale
            )
        
        elif rec.action == AmplificationAction.PARTIAL_TAKE_PROFIT:
            # Take partial profits
            take_pct = rec.parameters["take_profit_pct"]
            
            await execution_layer.partial_close_position(
                symbol=rec.candidate.position.symbol,
                close_pct=take_pct,
                leave_runner=True,
                rationale=rec.rationale
            )
```

### 5. Integration with Event-Driven Executor

```python
async def amplification_cycle():
    """Periodic amplification check."""
    
    # Get positions
    positions = await get_current_positions()
    
    # Convert to PAL format
    pal_positions = convert_to_pal_format(positions)
    
    # Run analysis
    report = pal.analyze_positions(
        positions=pal_positions,
        risk_profile=get_risk_profile(),
        portfolio_risk_budget_pct=get_risk_budget(),
        regime=get_regime(),
        exit_mode=get_exit_mode()
    )
    
    # Process recommendations
    for rec in report.recommendations:
        if rec.priority <= 2:  # High priority only
            await execute_amplification(rec)
    
    # Log results
    logger.info(
        f"[PAL] Processed {len(report.recommendations)} recommendations, "
        f"Avg score: {report.avg_amplification_score:.1f}"
    )

# Add to main event loop
async def main_loop():
    while True:
        # ... existing trade logic ...
        
        # Run amplification check every 15 minutes
        if time_for_amplification_check():
            await amplification_cycle()
        
        await asyncio.sleep(60)
```

---

## 🧪 Testing

### Standalone Test

```bash
cd backend/services
python profit_amplification.py
```

**Expected Output:**

```
================================================================================
PROFIT AMPLIFICATION LAYER (PAL) - Standalone Test
================================================================================

[OK] PAL initialized
  Min R for amplification: 1.0R
  Min R for scale-in: 1.5R
  Max DD from peak: 15.0%

================================================================================
Creating test positions...
================================================================================

[OK] Created 3 test positions
  1. BTCUSDT: 2.5R, Trend=strong, DD=10.7%
  2. ETHUSDT: 1.8R, Trend=very_strong, DD=5.3%
  3. ADAUSDT: 0.8R, Trend=weak, DD=60.0%

================================================================================
Running PAL analysis...
================================================================================

[OK] Analysis complete
  Candidates found: 2
  High priority: 2
  Recommendations: 2
  Avg amplification score: 76.0

================================================================================
AMPLIFICATION CANDIDATES
================================================================================

  Candidate 1: BTCUSDT
    Score: 73.6/100
    Current R: 2.5R
    Qualifications:
      - Scale-in: ❌
      - Extend hold: ✅
      - Partial take: ❌

  Candidate 2: ETHUSDT
    Score: 78.4/100
    Current R: 1.8R
    Qualifications:
      - Scale-in: ✅
      - Extend hold: ✅
      - Partial take: ❌

================================================================================
RECOMMENDATIONS
================================================================================

  Recommendation 1: BTCUSDT
    Action: extend_hold
    Priority: 2
    Rationale: Trend still strong at 2.5R, switch to trend-following exit
    Expected R increase: +1.00R
    Confidence: 65%
```

### Integration Test Scenarios

**Scenario 1: Strong Winner, Low DD**
```python
position = PositionSnapshot(
    symbol="BTCUSDT",
    current_R=2.5,
    peak_R=2.8,
    drawdown_from_peak_R=0.11,  # 11% DD
    trend_strength=TrendStrength.STRONG,
    volatility_regime="NORMAL"
)

# Expected: EXTEND_HOLD recommendation
```

**Scenario 2: Early Winner, Very Strong Trend**
```python
position = PositionSnapshot(
    symbol="ETHUSDT",
    current_R=1.8,
    peak_R=1.9,
    drawdown_from_peak_R=0.05,  # 5% DD
    trend_strength=TrendStrength.VERY_STRONG,
    volatility_regime="LOW"
)

# Expected: ADD_SIZE recommendation
```

**Scenario 3: High DD from Peak**
```python
position = PositionSnapshot(
    symbol="ADAUSDT",
    current_R=0.8,
    peak_R=2.0,
    drawdown_from_peak_R=0.60,  # 60% DD!
    trend_strength=TrendStrength.WEAK
)

# Expected: BLOCKED (HIGH_DRAWDOWN, WEAK_TREND)
```

---

## 🎓 Best Practices

### 1. **Use PAL as Advisory, Not Autonomous**

```python
# ✅ Good: Review recommendations
report = pal.analyze_positions(positions)
for rec in report.recommendations:
    if rec.confidence >= 70 and rec.risk_score <= 3:
        await execute_amplification(rec)

# ❌ Bad: Auto-execute everything
for rec in report.recommendations:
    await execute_amplification(rec)  # No review!
```

### 2. **Respect Blocking Factors**

```python
# ✅ Good: Check blocking factors
if not candidate.blocked_by:
    consider_amplification(candidate)
else:
    logger.warning(f"Blocked: {candidate.blocked_by}")

# ❌ Bad: Ignore blocks
if candidate.amplification_score > 50:
    amplify()  # Ignores blocks!
```

### 3. **Monitor Amplification Impact**

```python
# Track results
amplification_results = {
    "total_amplifications": 0,
    "successful": 0,
    "avg_r_increase": 0.0
}

# After amplification
if final_R > initial_R:
    amplification_results["successful"] += 1
    amplification_results["avg_r_increase"] += (final_R - initial_R)
```

### 4. **Adjust Thresholds Based on Performance**

```python
# If amplifications are too aggressive
pal.min_R_for_scale_in = 2.0  # Increase from 1.5
pal.max_dd_for_scale_in_pct = 5.0  # Tighter from 10

# If missing opportunities
pal.min_R_for_amplification = 0.8  # Lower from 1.0
pal.max_dd_from_peak_pct = 20.0  # Looser from 15
```

### 5. **Integrate with Risk Management**

```python
# Before amplification
if portfolio_balancer.can_add_exposure():
    if risk_guard.allows_scale_in(symbol):
        await execute_amplification(rec)
```

---

## 📊 Performance Metrics

### Key Metrics to Track

1. **Amplification Rate**
   - % of positions amplified
   - Target: 20-30% of winners

2. **Average R Increase**
   - Additional R from amplification
   - Target: +0.3R to +0.8R per amplified position

3. **Success Rate**
   - % of amplifications that increased final R
   - Target: >60%

4. **Risk-Adjusted Return**
   - R increase / risk_score
   - Target: >0.1

5. **Blocking Rate**
   - % of candidates blocked
   - Monitor: Should be 20-40%

---

## 🎯 Summary

The **Profit Amplification Layer** enhances profitability by:

✅ **Identifying Winners** - Score-based candidate selection (0-100)  
✅ **Smart Amplification** - 4 techniques (ADD_SIZE, EXTEND_HOLD, PARTIAL_TAKE, TREND_FOLLOW)  
✅ **Risk-Aware** - 6 blocking factors protect capital  
✅ **Advisory System** - Clear recommendations with rationale  
✅ **Integration Ready** - Works with PIL, PBA, Orchestrator  

**Key Integration Points:**
- Position Intelligence Layer (position data)
- Portfolio Balancer (risk budget, constraints)
- Orchestrator (regime, exit mode)
- Universe OS (symbol ranking, category)
- Execution Layer (action implementation)

**Expected Impact:**
- Average R increase: +15-25%
- Total profit increase: +10-20%
- Risk-adjusted returns: Improved

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Author:** Quantum Trader AI Team
