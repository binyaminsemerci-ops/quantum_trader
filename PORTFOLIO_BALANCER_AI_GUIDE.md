# PORTFOLIO BALANCER AI (PBA) — Complete Integration Guide

**Version:** 1.0  
**Date:** November 23, 2025  
**Status:** ✅ IMPLEMENTED & TESTED

---

## 🎯 MISSION

The **Portfolio Balancer AI (PBA)** is the global portfolio state manager for Quantum Trader. It controls total exposure, diversification, and risk across ALL open positions and candidate signals.

**Key Principle:** PBA does NOT manage single trades in isolation — it manages the ENTIRE PORTFOLIO as a unified system.

---

## 📊 WHAT DOES PBA DO?

### Core Responsibilities

1. **Exposure Analysis**
   - Tracks long/short/net exposure
   - Monitors sector concentration
   - Detects symbol over-concentration
   - Identifies correlation clusters

2. **Constraint Enforcement**
   - Max positions (global & per-symbol)
   - Max total risk percentage
   - Max leverage limits
   - Max sector exposure

3. **Trade Prioritization**
   - Ranks signals by confidence, category, stability
   - Filters trades based on portfolio constraints
   - Prefers CORE symbols over EXPANSION

4. **Risk Mode Management**
   - SAFE: Defensive, minimal new positions
   - NEUTRAL: Balanced approach
   - AGGRESSIVE: Maximum opportunity capture

5. **Advisory Output**
   - Violations & warnings
   - Allowed vs dropped trades
   - Portfolio recommendations
   - Required actions

---

## 🏗️ ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                    PORTFOLIO BALANCER AI (PBA)                   │
└─────────────────────────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           INPUT AGGREGATION                   │
        ├──────────────────────────────────────────────┤
        │ • Open Positions (symbol, size, margin)      │
        │ • Candidate Trades (signals)                 │
        │ • Total Equity / Margin                      │
        │ • Orchestrator Policy                        │
        │ • Risk Manager State                         │
        │ • Universe OS Data                           │
        └──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           PORTFOLIO STATE COMPUTATION        │
        ├──────────────────────────────────────────────┤
        │ • Total exposure (long/short/net)            │
        │ • Risk percentage                            │
        │ • Concentration metrics                      │
        │ • Diversification analysis                   │
        └──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           VIOLATION DETECTION                │
        ├──────────────────────────────────────────────┤
        │ • Max positions breached?                    │
        │ • Risk limits exceeded?                      │
        │ • Over-concentration detected?               │
        │ • Leverage too high?                         │
        └──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           RISK MODE DETERMINATION            │
        ├──────────────────────────────────────────────┤
        │ • SAFE: Critical violations / high DD        │
        │ • NEUTRAL: Normal conditions                 │
        │ • AGGRESSIVE: Low risk, healthy portfolio    │
        └──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           TRADE FILTERING & PRIORITY         │
        ├──────────────────────────────────────────────┤
        │ • Rank by confidence + category + stability  │
        │ • Filter by constraints                      │
        │ • Block duplicates & over-leveraged          │
        │ • Select top N trades                        │
        └──────────────────────────────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │           OUTPUT GENERATION                  │
        ├──────────────────────────────────────────────┤
        │ • Portfolio state summary                    │
        │ • Violations list                            │
        │ • Allowed trades                             │
        │ • Dropped trades (with reasons)              │
        │ • Recommendations                            │
        │ • Required actions                           │
        └──────────────────────────────────────────────┘
```

---

## 🔧 INTEGRATION STEPS

### Step 1: Import PBA

```python
from backend.services.portfolio_balancer import (
    PortfolioBalancerAI,
    Position,
    CandidateTrade,
    PortfolioConstraints,
    RiskMode
)
```

### Step 2: Create PBA Instance

```python
# Custom constraints (optional)
constraints = PortfolioConstraints(
    max_positions=8,
    max_positions_per_symbol=1,
    max_total_risk_pct=15.0,
    max_per_trade_risk_pct=2.0,
    max_symbol_concentration_pct=30.0,
    max_sector_concentration_pct=40.0,
    max_leverage=30.0
)

# Initialize balancer
balancer = PortfolioBalancerAI(
    constraints=constraints,
    data_dir="/app/data"
)
```

### Step 3: Prepare Input Data

```python
# Convert open positions to Position objects
positions = []
for pos_data in open_positions_from_exchange:
    positions.append(Position(
        symbol=pos_data["symbol"],
        side=pos_data["side"],  # LONG or SHORT
        size=pos_data["size"],
        entry_price=pos_data["entry_price"],
        current_price=pos_data["current_price"],
        margin=pos_data["margin"],
        leverage=pos_data["leverage"],
        category=pos_data.get("category", "EXPANSION"),
        sector=pos_data.get("sector", "unknown"),
        risk_amount=pos_data.get("risk_amount", 0)
    ))

# Convert signals to CandidateTrade objects
candidates = []
for signal in ai_signals:
    if signal["confidence"] >= confidence_threshold:
        candidates.append(CandidateTrade(
            symbol=signal["symbol"],
            action=signal["action"],  # BUY or SELL
            confidence=signal["confidence"],
            size=calculated_position_size,
            margin_required=calculated_margin,
            risk_amount=calculated_risk,
            category=signal.get("category", "EXPANSION"),
            sector=signal.get("sector", "unknown"),
            stability_score=signal.get("stability", 0.0),
            cost_score=signal.get("cost", 0.0),
            recent_performance=signal.get("performance", 0.0)
        ))
```

### Step 4: Run Analysis

```python
# Get portfolio equity and margin info
total_equity = account_data["total_equity"]
used_margin = account_data["used_margin"]
free_margin = account_data["free_margin"]

# Get orchestrator and risk manager states (optional)
orchestrator_policy = orchestrator.get_current_policy()
risk_manager_state = risk_manager.get_state()

# Run PBA analysis
output = balancer.analyze_portfolio(
    positions=positions,
    candidates=candidates,
    total_equity=total_equity,
    used_margin=used_margin,
    free_margin=free_margin,
    orchestrator_policy=orchestrator_policy,
    risk_manager_state=risk_manager_state
)
```

### Step 5: Use Output

```python
# Check risk mode
if output.risk_mode == "SAFE":
    logger.warning("🛡️ PBA in SAFE mode - reducing trading activity")

# Check for critical violations
critical_violations = [v for v in output.violations if v.severity == "CRITICAL"]
if critical_violations:
    logger.error("🚨 CRITICAL violations detected - blocking trades")
    for v in critical_violations:
        logger.error(f"  {v.message}")
    return  # Stop execution

# Process allowed trades
for trade in output.allowed_trades:
    logger.info(f"✅ APPROVED: {trade.symbol} {trade.action} (priority: {trade.priority_score:.2f})")
    # Execute trade via smart_execution
    await smart_execution.execute_trade(
        symbol=trade.symbol,
        action=trade.action,
        size=trade.recommended_size
    )

# Log dropped trades
for trade in output.dropped_trades:
    logger.info(f"❌ DROPPED: {trade.symbol} {trade.action} - {trade.reason}")

# Log recommendations
logger.info("📊 Portfolio Recommendations:")
for rec in output.recommendations:
    logger.info(f"  • {rec}")

# Execute required actions
if output.actions_required:
    logger.warning("⚠️ ACTIONS REQUIRED:")
    for action in output.actions_required:
        logger.warning(f"  • {action}")
```

---

## 📋 CONFIGURATION

### PortfolioConstraints Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_positions` | 8 | Maximum number of open positions |
| `max_positions_per_symbol` | 1 | Max positions per symbol (prevent duplicates) |
| `max_total_risk_pct` | 15.0 | Maximum total portfolio risk (% of equity) |
| `max_per_trade_risk_pct` | 2.0 | Maximum risk per trade (% of equity) |
| `max_symbol_concentration_pct` | 30.0 | Max exposure in single symbol (% of equity) |
| `max_sector_concentration_pct` | 40.0 | Max exposure in single sector (% of equity) |
| `max_leverage` | 30.0 | Maximum allowed leverage |
| `max_long_exposure_pct` | 100.0 | Max long exposure (% of equity) |
| `max_short_exposure_pct` | 100.0 | Max short exposure (% of equity) |
| `max_net_exposure_pct` | 80.0 | Max net exposure (% of equity) |

---

## 🎮 RISK MODES

### SAFE Mode

**Triggered by:**
- Critical violations present
- Daily drawdown > 3%
- Losing streak ≥ 5
- Max positions reached
- Total risk > 80% of limit

**Behavior:**
- Block most new trades
- Reduce position sizes
- Prefer closing positions
- Only allow highest priority CORE trades

### NEUTRAL Mode

**Default state:**
- No critical violations
- Portfolio within normal parameters
- Balanced risk/reward

**Behavior:**
- Normal trade execution
- Standard filtering rules
- Balanced CORE/EXPANSION mix

### AGGRESSIVE Mode

**Triggered by:**
- Low risk exposure
- Healthy portfolio
- Low volatility
- No violations

**Behavior:**
- Maximum opportunity capture
- Looser filtering
- More positions allowed
- Higher risk tolerance

---

## 📊 OUTPUT STRUCTURE

### BalancerOutput

```python
{
    "timestamp": "2025-11-23T12:00:00Z",
    "risk_mode": "NEUTRAL",
    "portfolio_state": {
        "total_equity": 10000.0,
        "used_margin": 1500.0,
        "free_margin": 8500.0,
        "total_exposure_long": 40000.0,
        "total_exposure_short": 0.0,
        "net_exposure": 40000.0,
        "total_risk_pct": 1.5,
        "total_positions": 2,
        "long_positions": 2,
        "short_positions": 0,
        "max_symbol_concentration_pct": 20.0,
        "category_distribution": {"CORE": 2}
    },
    "violations": [],
    "allowed_trades": [
        {
            "symbol": "SOLUSDT",
            "action": "BUY",
            "allowed": true,
            "reason": "APPROVED",
            "priority_score": 95.0,
            "recommended_size": 10.0
        }
    ],
    "dropped_trades": [
        {
            "symbol": "DOGEUSDT",
            "action": "BUY",
            "allowed": false,
            "reason": "LOW_PRIORITY",
            "priority_score": 65.0
        }
    ],
    "recommendations": [
        "RISK MODE: NEUTRAL",
        "Portfolio healthy - continue normal operations"
    ],
    "actions_required": []
}
```

---

## 🔍 MONITORING

### Key Metrics to Watch

1. **Risk Mode Transitions**
   - Track SAFE mode frequency (should be rare)
   - Monitor AGGRESSIVE mode duration

2. **Violation Patterns**
   - Which constraints violated most often?
   - Are violations recurring?

3. **Trade Rejection Reasons**
   - Why are trades being dropped?
   - Optimize constraints if too restrictive

4. **Portfolio Health**
   - Concentration levels
   - Risk percentage trends
   - Long/short balance

### Log Output

```
2025-11-23 12:00:00 - PORTFOLIO BALANCER AI — ANALYSIS STARTING
2025-11-23 12:00:00 - Total Equity: $10,000.00
2025-11-23 12:00:00 - Open Positions: 2
2025-11-23 12:00:00 - Candidate Trades: 5
2025-11-23 12:00:00 - Portfolio State Computed:
2025-11-23 12:00:00 -   Total Positions: 2 (L:2 S:0)
2025-11-23 12:00:00 -   Gross Exposure: $40,000.00
2025-11-23 12:00:00 -   Total Risk: 1.50%
2025-11-23 12:00:00 - ✅ No constraint violations
2025-11-23 12:00:00 - ✅ Risk Mode: NEUTRAL
2025-11-23 12:00:00 - Trade Filtering Results:
2025-11-23 12:00:00 -   ✅ Allowed: 3
2025-11-23 12:00:00 -   ❌ Dropped: 2
```

---

## 🧪 TESTING

Run tests:

```bash
cd backend
pytest tests/test_portfolio_balancer.py -v
```

Expected output:
```
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_portfolio_state_computation PASSED
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_no_violations_healthy_portfolio PASSED
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_max_positions_violation PASSED
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_risk_exceeded_violation PASSED
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_trade_prioritization PASSED
tests/test_portfolio_balancer.py::TestPortfolioBalancerAI::test_full_analysis_workflow PASSED

======================== 21 passed ========================
```

---

## 🚀 DEPLOYMENT

### Integration Points

1. **Event-Driven Executor**
   - Call PBA before executing trades
   - Use `allowed_trades` list
   - Respect `actions_required`

2. **AI Trading Engine**
   - Pass signals as candidates
   - Filter by PBA approval
   - Log rejection reasons

3. **Position Monitor**
   - Provide current positions
   - Update with portfolio state
   - Monitor violations

4. **Orchestrator Policy**
   - Share policy state with PBA
   - Align risk modes
   - Coordinate constraints

### Recommended Workflow

```python
async def execute_trading_cycle():
    # 1. Get AI signals
    signals = await ai_engine.generate_signals()
    
    # 2. Get open positions
    positions = await exchange.get_open_positions()
    
    # 3. Convert to PBA format
    pba_positions = convert_to_pba_positions(positions)
    pba_candidates = convert_to_pba_candidates(signals)
    
    # 4. Run PBA analysis
    pba_output = balancer.analyze_portfolio(
        positions=pba_positions,
        candidates=pba_candidates,
        total_equity=account.equity,
        used_margin=account.used_margin,
        free_margin=account.free_margin
    )
    
    # 5. Check for critical violations
    if any(v.severity == "CRITICAL" for v in pba_output.violations):
        logger.error("Critical violations - skipping execution")
        return
    
    # 6. Execute approved trades only
    for trade in pba_output.allowed_trades:
        await execute_trade(trade)
    
    # 7. Log dropped trades for analysis
    for trade in pba_output.dropped_trades:
        logger.info(f"Trade dropped: {trade.symbol} - {trade.reason}")
```

---

## 📚 BEST PRACTICES

1. **Always Check Output**
   - Never ignore `actions_required`
   - Respect SAFE mode restrictions
   - Log all violations for analysis

2. **Regular Monitoring**
   - Review PBA logs daily
   - Track violation patterns
   - Adjust constraints if needed

3. **Constraint Tuning**
   - Start conservative (current defaults)
   - Gradually adjust based on performance
   - Document changes

4. **Integration Testing**
   - Test with real market data
   - Simulate constraint violations
   - Verify risk mode transitions

5. **Performance Optimization**
   - Cache universe data
   - Batch position queries
   - Async operations where possible

---

## ✅ CHECKLIST

- [ ] PBA integrated into event-driven executor
- [ ] Position data conversion implemented
- [ ] Signal data conversion implemented
- [ ] Critical violation handling added
- [ ] Risk mode respected in execution
- [ ] Logging configured
- [ ] Tests passing
- [ ] Monitoring dashboards updated
- [ ] Documentation reviewed
- [ ] Team trained on PBA outputs

---

## 📞 SUPPORT

**File:** `backend/services/portfolio_balancer.py`  
**Tests:** `backend/tests/test_portfolio_balancer.py`  
**Output:** `/app/data/portfolio_balancer_output.json`

**Common Issues:**

1. **All trades dropped** → Check constraints, likely at limits
2. **SAFE mode constantly** → Review risk manager settings
3. **No violations but trades blocked** → Check duplicate symbols
4. **High concentration warnings** → Diversify across more symbols

---

**Status:** ✅ READY FOR DEPLOYMENT  
**Version:** 1.0  
**Last Updated:** November 23, 2025
