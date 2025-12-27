# Meta Strategy Controller (MSC AI) - Quick Start Guide

## 🚀 Quick Start

### Prerequisites
- Quantum Trader backend running
- Database with execution_journal table populated
- (Optional) Redis for fast policy access

### Start the System

1. **Start Backend with MSC AI**
```bash
cd c:\quantum_trader
python backend/main.py
```

Look for these startup messages:
```
[OK] Meta Strategy Controller (MSC AI) available
🧠 META STRATEGY CONTROLLER: ENABLED (supreme AI decision brain)
[MSC Scheduler] Started - will run every 30 minutes
```

2. **Check MSC AI Status**
```bash
curl http://localhost:8000/api/msc/status
```

3. **Trigger Manual Evaluation**
```bash
curl -X POST http://localhost:8000/api/msc/evaluate
```

4. **View Strategy Rankings**
```bash
curl http://localhost:8000/api/msc/strategies
```

---

## 🔍 Test Example Results

When you run the examples file, you'll see MSC AI in action:

```bash
python backend/services/meta_strategy_controller_examples.py
```

**Expected Output:**
```
=============================================================================
EXAMPLE 1: Normal Operation (Healthy System)
=============================================================================

[SYSTEM HEALTH]
  Drawdown: 3.00%
  Global Winrate: 55.0%
  Equity Slope: +1.00%/day
  Regime: BULL_TRENDING
  Volatility: NORMAL

[RISK MODE] Selected: AGGRESSIVE
  - Positive trend: 1.00%/day (AGGRESSIVE)
  - Trending regime: BULL_TRENDING (AGGRESSIVE)

[STRATEGY SCORES] Evaluated 5 strategies:
   1. STRAT_001 | Score: 1.011 | Trades: 120 | PF: 0.400 | WR: 0.225
   2. STRAT_002 | Score: 0.786 | Trades:  85 | PF: 0.380 | WR: 0.188
   3. STRAT_003 | Score: 0.772 | Trades:  65 | PF: 0.300 | WR: 0.138
   4. STRAT_004 | Score: 0.475 | Trades:  45 | PF: 0.240 | WR: 0.089
   5. STRAT_005 | Score: 0.230 | Trades:  30 | PF: 0.160 | WR: 0.000

[SELECTION] Selected 4 strategies:
  ✓ STRAT_001
  ✓ STRAT_002
  ✓ STRAT_003
  ✓ STRAT_004

[POLICY] Updated successfully
  Risk Mode: AGGRESSIVE
  Max Risk/Trade: 1.50%
  Min Confidence: 50.0%
  Max Positions: 15
  Active Strategies: 4

✓ Example 1 passed: AGGRESSIVE mode selected due to strong performance!
```

---

## 📊 Policy Adaptation Scenarios

### Scenario 1: Healthy System → Aggressive Mode
```
System Metrics:
- Drawdown: 2.0%
- Winrate: 60%
- Equity: +1.2%/day
- Regime: BULL_TRENDING

MSC AI Decision:
✅ Risk Mode: AGGRESSIVE
✅ Max Risk: 1.5% per trade
✅ Max Positions: 15
✅ Min Confidence: 50%
✅ Strategies: 6 active
```

### Scenario 2: Drawdown Starts → Normal Mode
```
System Metrics:
- Drawdown: 4.5%
- Winrate: 52%
- Equity: +0.5%/day
- Regime: RANGING
- Volatility: HIGH

MSC AI Decision:
⚠️ Risk Mode: NORMAL
⚠️ Max Risk: 0.75% per trade
⚠️ Max Positions: 10
⚠️ Min Confidence: 60%
⚠️ Strategies: 4 active
```

### Scenario 3: Crisis Mode → Defensive
```
System Metrics:
- Drawdown: 7.0%
- Winrate: 45%
- Equity: -0.8%/day
- Regime: CHOPPY
- Volatility: EXTREME

MSC AI Decision:
🛡️ Risk Mode: DEFENSIVE
🛡️ Max Risk: 0.3% per trade
🛡️ Max Positions: 4
🛡️ Min Confidence: 70%
🛡️ Strategies: 2 active (only most reliable)
```

---

## 🎯 Integration with Trading Components

### Event-Driven Executor Integration

Add this to your Event-Driven Executor:

```python
from backend.services.msc_ai_integration import QuantumPolicyStoreMSC

# Before executing trades
policy_store = QuantumPolicyStoreMSC()
policy = policy_store.read_policy()

if policy:
    # Apply MSC AI constraints
    allowed_strategies = set(policy["allowed_strategies"])
    min_confidence = policy["global_min_confidence"]
    max_positions = policy["max_positions"]
    max_risk_per_trade = policy["max_risk_per_trade"]
    
    # Filter signals
    filtered_signals = []
    for signal in raw_signals:
        # Check strategy allowed
        if signal.strategy_id not in allowed_strategies:
            logger.info(f"[MSC] Skipping {signal.symbol} - strategy {signal.strategy_id} not allowed")
            continue
        
        # Check confidence threshold
        if signal.confidence < min_confidence:
            logger.info(f"[MSC] Skipping {signal.symbol} - confidence {signal.confidence:.2f} below {min_confidence:.2f}")
            continue
        
        filtered_signals.append(signal)
    
    # Check position limit
    if len(current_positions) >= max_positions:
        logger.warning(f"[MSC] Position limit reached: {len(current_positions)}/{max_positions}")
        return  # Don't open new positions
    
    # Apply risk sizing from MSC AI
    for signal in filtered_signals:
        signal.risk_pct = max_risk_per_trade  # Override with MSC AI risk
```

---

## 📈 Monitoring Dashboard Ideas

### Real-Time Policy Display
```
┌─────────────────────────────────────┐
│   META STRATEGY CONTROLLER STATUS   │
├─────────────────────────────────────┤
│ Risk Mode:      🚀 AGGRESSIVE       │
│ Active Since:   10 minutes ago      │
│ Next Eval:      20 minutes          │
├─────────────────────────────────────┤
│ System Health:                      │
│  • Drawdown:    2.5% ✅             │
│  • Winrate:     58% ✅              │
│  • Equity:      +0.8%/day ✅        │
│  • Regime:      BULL_TRENDING       │
├─────────────────────────────────────┤
│ Policy Parameters:                  │
│  • Max Risk:    1.5% per trade      │
│  • Max Pos:     15 concurrent       │
│  • Min Conf:    50%                 │
│  • Daily Limit: 50 trades           │
├─────────────────────────────────────┤
│ Active Strategies: 4/8              │
│  ✅ STRAT_001 (Score: 1.011)        │
│  ✅ STRAT_002 (Score: 0.786)        │
│  ✅ STRAT_003 (Score: 0.772)        │
│  ✅ STRAT_004 (Score: 0.475)        │
└─────────────────────────────────────┘
```

---

## 🔄 Feedback Loop

MSC AI creates a complete feedback loop:

```
┌─────────────────────────────────────────────┐
│          QUANTUM TRADER SYSTEM              │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│     STRATEGIES GENERATE TRADE SIGNALS       │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│    EVENT-DRIVEN EXECUTOR EXECUTES TRADES    │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│   TRADES RECORDED IN EXECUTION_JOURNAL      │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│        MSC AI EVALUATES PERFORMANCE         │
│   • Reads metrics from database             │
│   • Calculates system health                │
│   • Scores all strategies                   │
│   • Determines risk mode                    │
│   • Builds new policy                       │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│   POLICY WRITTEN TO REDIS + DATABASE        │
└─────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────┐
│  EXECUTOR READS POLICY FOR NEXT DECISIONS   │
│   • Filters by allowed strategies           │
│   • Applies confidence thresholds           │
│   • Respects position limits                │
│   • Uses MSC AI risk sizing                 │
└─────────────────────────────────────────────┘
              ↓
         [LOOP BACK TO TOP]
```

**This creates a self-optimizing system where poor performance automatically triggers defensive measures, and strong performance unlocks aggressive growth.**

---

## 🛠️ Customization

### Adjust Evaluation Frequency
```bash
# .env file
MSC_EVALUATION_INTERVAL_MINUTES=15  # Every 15 minutes (more responsive)
# or
MSC_EVALUATION_INTERVAL_MINUTES=60  # Every hour (more stable)
```

### Modify Risk Thresholds
Edit `backend/services/meta_strategy_controller.py`:

```python
class RiskModeSelector:
    def select_risk_mode(self, health: SystemHealth) -> RiskMode:
        # Customize these thresholds
        DD_THRESHOLD_DEFENSIVE = 5.0  # Default: 5%
        DD_THRESHOLD_NORMAL = 3.0     # Default: 3%
        WR_THRESHOLD_AGGRESSIVE = 0.60 # Default: 60%
        WR_THRESHOLD_DEFENSIVE = 0.45  # Default: 45%
```

### Change Strategy Limits
```python
controller = MetaStrategyController(
    metrics_repo=metrics_repo,
    strategy_repo=strategy_repo,
    policy_store=policy_store,
    evaluation_period_days=30,  # Look back 30 days
    min_strategies=3,           # Minimum 3 strategies
    max_strategies=10           # Maximum 10 strategies
)
```

---

## 🎊 Success Criteria

MSC AI is working correctly when you see:

1. ✅ Scheduler running every 30 minutes
2. ✅ Policy updates logged with reasoning
3. ✅ `/api/msc/status` returns current policy
4. ✅ Database `msc_policies` table has records
5. ✅ Strategy rankings change based on performance
6. ✅ Risk mode adapts to system health
7. ✅ Prometheus metrics updating

---

## 🚨 Important Notes

- **First Evaluation**: MSC AI needs trading data to work. If no trades exist, it will use conservative defaults.
- **Strategy Selection**: Only strategies in `runtime_strategies` table with status='LIVE' are considered.
- **Redis Optional**: MSC AI works without Redis, using database-only mode.
- **Mode Changes**: Logged at WARNING level for visibility in monitoring tools.

---

## 📞 Support

If MSC AI isn't working as expected:

1. Check logs: `grep "MSC" logs/quantum_trader.log`
2. Verify database has trade data
3. Test manual evaluation: `POST /api/msc/evaluate`
4. Check scheduler status in `/api/msc/status`

---

**The Meta Strategy Controller is now the supreme brain of your trading system, continuously optimizing risk and strategy selection based on real performance data! 🧠🚀**
