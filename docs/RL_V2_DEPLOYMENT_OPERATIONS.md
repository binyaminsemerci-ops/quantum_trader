# RL v2 Deployment & Operations Guide

**Status**: ✅ Production Ready  
**Date**: December 2, 2025  
**Version**: 2.0

---

## 🚀 Quick Start Deployment

### 1. Verify Installation

```powershell
# Verify all RL v2 modules are present
python -c "from backend.domains.learning.rl_v2 import *; print('✅ RL v2 modules loaded')"

# Run integration tests
$env:PYTHONPATH="c:\quantum_trader"
python tests/integration/test_rl_v2_pipeline.py
```

### 2. Start Backend with RL v2

```powershell
# Start backend (RL v2 auto-initializes)
python backend/main.py
```

The RL v2 system will automatically:
- Initialize Q-learning agents
- Subscribe to EventBus events
- Start learning from signals and trades
- Save Q-tables to `data/rl_v2/`

---

## 📊 Monitoring Commands

### Real-Time Dashboard

```powershell
# One-time snapshot
python scripts/monitor_rl_v2.py

# Continuous monitoring (updates every 60s)
python scripts/monitor_rl_v2.py --continuous

# Custom update interval (e.g., 30s)
python scripts/monitor_rl_v2.py --continuous --interval 30
```

**Dashboard shows**:
- Q-table growth (states, state-action pairs)
- Update counts
- Exploration rate (epsilon)
- Q-value statistics (min, max, avg)
- Learning progress
- Health checks

### Expected Initial Output

```
================================================================================
RL v2 MONITORING DASHBOARD
================================================================================
Timestamp: 2025-12-02T15:30:00

📊 META STRATEGY AGENT
--------------------------------------------------------------------------------
  States Learned:        0
  State-Action Pairs:    0
  ⚠️  Q-table not found - agent not yet trained

📈 POSITION SIZING AGENT
--------------------------------------------------------------------------------
  States Learned:        0
  State-Action Pairs:    0
  ⚠️  Q-table not found - agent not yet trained

🎯 OVERALL STATISTICS
--------------------------------------------------------------------------------
  Total Updates:         0
  Total States:          0
  Total Q-Table Size:    0.00 KB

🏥 HEALTH CHECK
--------------------------------------------------------------------------------
  ✅ Q-tables at healthy size
  🔍 High exploration rate - still learning
  🌱 Early learning stage (< 100 updates)
```

### After 1 Hour of Trading

```
================================================================================
RL v2 MONITORING DASHBOARD
================================================================================

📊 META STRATEGY AGENT
--------------------------------------------------------------------------------
  States Learned:        24
  State-Action Pairs:    156
  Avg Actions/State:     6.5
  Total Updates:         47
  Exploration Rate (ε):  0.0950
  Q-Value Range:         [-0.1234, 2.4567]
  Avg Q-Value:           1.2345
  Q-Table Size:          8.45 KB

📈 POSITION SIZING AGENT
--------------------------------------------------------------------------------
  States Learned:        18
  State-Action Pairs:    112
  Avg Actions/State:     6.2
  Total Updates:         52
  Exploration Rate (ε):  0.0930
  Q-Value Range:         [-0.2345, 2.1234]
  Avg Q-Value:           1.0987
  Q-Table Size:          6.23 KB

📈 LEARNING PROGRESS (Last vs Current)
--------------------------------------------------------------------------------
  Meta Agent:
    New States:          +8
    New Updates:         +15
  Sizing Agent:
    New States:          +6
    New Updates:         +18

🏥 HEALTH CHECK
--------------------------------------------------------------------------------
  ⚡ Q-tables growing normally
  🔍 High exploration rate - still learning
  📚 Active learning (100-1000 updates)
```

---

## 🔧 Hyperparameter Tuning

### Get Recommendations

```powershell
# Analyze performance and get recommendations
python scripts/tune_rl_v2_hyperparams.py
```

**Output example**:

```
================================================================================
RL v2 HYPERPARAMETER RECOMMENDATIONS
================================================================================

📊 META STRATEGY AGENT
--------------------------------------------------------------------------------
  Recommended Changes:
    Alpha (α):    0.0150
    Gamma (γ):    0.9900
    Epsilon (ε):  0.0800

  Rationale:
    • [Meta Strategy] Reduced alpha to 0.015 for more stable learning
    • [Meta Strategy] Reduced epsilon to 0.080 as agent converges

📈 POSITION SIZING AGENT
--------------------------------------------------------------------------------
  ✅ Current hyperparameters are optimal

🔧 TO APPLY RECOMMENDATIONS
--------------------------------------------------------------------------------
  Run the following command:

  python scripts/tune_rl_v2_hyperparams.py --apply --meta-alpha 0.015 --meta-epsilon 0.08
```

### Apply Manual Hyperparameters

```powershell
# Apply custom hyperparameters
python scripts/tune_rl_v2_hyperparams.py --apply `
  --meta-alpha 0.005 `
  --meta-gamma 0.99 `
  --meta-epsilon 0.05 `
  --sizing-alpha 0.01 `
  --sizing-gamma 0.98 `
  --sizing-epsilon 0.1
```

### Hyperparameter Guidelines

| Parameter | Early Learning | Active Learning | Mature Agent | Purpose |
|-----------|---------------|-----------------|--------------|---------|
| **Alpha (α)** | 0.01-0.02 | 0.01-0.015 | 0.001-0.005 | Learning rate - how fast to update Q-values |
| **Gamma (γ)** | 0.95-0.99 | 0.98-0.99 | 0.99-0.999 | Discount factor - long-term vs short-term rewards |
| **Epsilon (ε)** | 0.2-0.3 | 0.1-0.2 | 0.01-0.05 | Exploration rate - random vs greedy actions |

**Tuning tips**:
- **High variance in rewards?** → Lower gamma (0.95)
- **Agent not learning?** → Increase alpha (0.02)
- **Stuck in local optimum?** → Increase epsilon (0.2)
- **Mature but unstable?** → Lower alpha (0.005) and epsilon (0.01)

---

## 🔬 A/B Testing: RL v1 vs v2

### Run Comparison

```powershell
# Compare performance metrics
python scripts/ab_test_rl_v1_vs_v2.py `
  --v1-pnl 1000 --v1-winrate 0.55 --v1-sharpe 1.2 --v1-drawdown 150 `
  --v2-pnl 1200 --v2-winrate 0.58 --v2-sharpe 1.5 --v2-drawdown 120
```

**Output**:

```
================================================================================
RL v1 vs v2 A/B TEST COMPARISON
================================================================================

📊 PERFORMANCE METRICS
--------------------------------------------------------------------------------
Metric                    RL v1           RL v2           Diff %
--------------------------------------------------------------------------------
Total Pnl                 $1,000.00       $1,200.00       +20.0% ✅
Win Rate                  55.00%          58.00%          +5.5% ✅
Sharpe Ratio              1.20            1.50            +25.0% ✅
Max Drawdown              150.00          120.00          -20.0% ✅

🏆 RESULT
--------------------------------------------------------------------------------
  Winner: RL v2 🎉
  Confidence: 100.0%
  Recommendation: Deploy RL v2 to production
```

### View Historical Tests

```powershell
# View all historical A/B test results
python scripts/ab_test_rl_v1_vs_v2.py --history
```

---

## 📈 Growth Expectations

### Day 1 (First 24 Hours)

- **Updates**: 50-200
- **States**: 10-50
- **Q-table size**: 2-10 KB
- **Epsilon**: ~0.09 (started at 0.1)
- **Behavior**: High exploration, learning basic patterns

### Week 1 (7 Days)

- **Updates**: 500-2000
- **States**: 100-500
- **Q-table size**: 20-100 KB
- **Epsilon**: ~0.07-0.08
- **Behavior**: Reduced exploration, forming preferences

### Month 1 (30 Days)

- **Updates**: 2000-10000
- **States**: 500-2000
- **Q-table size**: 100-500 KB
- **Epsilon**: ~0.05-0.06
- **Behavior**: Mostly exploiting, occasional exploration

### Month 3+ (Mature)

- **Updates**: 10000+
- **States**: 2000-5000
- **Q-table size**: 500-2000 KB
- **Epsilon**: ~0.01-0.05
- **Behavior**: Highly exploitative, minimal exploration

---

## 🛠️ Maintenance Tasks

### Daily

```powershell
# Check Q-table growth
python scripts/monitor_rl_v2.py

# Verify backend health
Invoke-RestMethod -Uri "http://localhost:8000/health"
```

### Weekly

```powershell
# Get hyperparameter recommendations
python scripts/tune_rl_v2_hyperparams.py

# Backup Q-tables
Copy-Item data/rl_v2/*.json data/rl_v2/backups/weekly_$(Get-Date -Format 'yyyy-MM-dd')/ -Force
```

### Monthly

```powershell
# Run A/B test comparison
# (Collect metrics from monitoring dashboard and RL v1 system)
python scripts/ab_test_rl_v1_vs_v2.py --v1-pnl <v1_pnl> --v1-winrate <v1_wr> --v2-pnl <v2_pnl> --v2-winrate <v2_wr>

# Review historical A/B tests
python scripts/ab_test_rl_v1_vs_v2.py --history

# Consider hyperparameter tuning
python scripts/tune_rl_v2_hyperparams.py --apply <recommended_params>
```

---

## 🔍 Troubleshooting

### Q-Tables Not Growing

**Symptoms**: States remain at 0 after hours of trading

**Solutions**:
1. Verify backend is running: `Invoke-RestMethod -Uri "http://localhost:8000/health"`
2. Check RL subscriber is active: `grep "RL v2 Subscriber" logs/*.log`
3. Verify signals are being generated: `grep "SIGNAL_GENERATED" logs/*.log`
4. Check Q-table directory exists: `Test-Path data/rl_v2/`

### High Q-Value Variance

**Symptoms**: Q-values range from -10 to +10 (very wide)

**Solutions**:
1. Lower gamma to 0.95 for short-term focus
2. Reduce alpha to 0.005 for stability
3. Check reward calculations in logs
4. Verify state representation is consistent

### Agent Not Exploiting (Always Exploring)

**Symptoms**: Epsilon stays high (> 0.2), random-looking behavior

**Solutions**:
1. Manually lower epsilon: `python scripts/tune_rl_v2_hyperparams.py --apply --meta-epsilon 0.05 --sizing-epsilon 0.05`
2. Verify Q-values are being updated (check `update_count`)
3. Check that rewards are meaningful (not all zeros)

### Q-Tables Too Large (> 10 MB)

**Symptoms**: Q-table files exceed 10 MB, slow loading

**Solutions**:
1. Reduce state space dimensionality (fewer bins/categories)
2. Implement state aggregation
3. Consider function approximation (DQN) - see Advanced Features below
4. Prune rarely-visited states

---

## 🚀 Advanced Features (Future)

### 1. Deep Q-Network (DQN)

Replace Q-tables with neural networks for continuous state spaces.

**Implementation checklist**:
- [ ] Create `backend/domains/learning/rl_v2/dqn_core.py`
- [ ] Implement experience replay buffer
- [ ] Add target network for stability
- [ ] Train on historical trade data
- [ ] Compare DQN vs Q-learning performance

### 2. Proximal Policy Optimization (PPO)

State-of-the-art policy gradient method.

**Implementation checklist**:
- [ ] Create `backend/domains/learning/rl_v2/ppo_agent.py`
- [ ] Implement advantage estimation (GAE)
- [ ] Add clipped objective function
- [ ] Tune clipping epsilon and learning rate
- [ ] Benchmark vs Q-learning and DQN

### 3. Multi-Agent Coordination

Multiple RL agents with different specializations.

**Implementation checklist**:
- [ ] Create `backend/domains/learning/rl_v2/multi_agent_coordinator.py`
- [ ] Implement centralized training, decentralized execution (CTDE)
- [ ] Add communication protocol between agents
- [ ] Agents: trend-following, mean-reversion, volatility-based, regime-switching
- [ ] Ensemble voting mechanism

### 4. Hierarchical RL

Two-level hierarchy: meta-controller and sub-policies.

**Implementation checklist**:
- [ ] Meta-controller: selects high-level strategy (e.g., "aggressive", "conservative")
- [ ] Sub-policies: execute strategy-specific actions
- [ ] Temporal abstraction with options framework
- [ ] Multi-timescale learning

---

## 📊 Performance Benchmarks

### Target Metrics (Month 1)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Win Rate | > 52% | > 55% | > 60% |
| Sharpe Ratio | > 1.0 | > 1.5 | > 2.0 |
| Max Drawdown | < 15% | < 10% | < 5% |
| Avg Trade Duration | < 4h | < 3h | < 2h |
| Total PnL | Positive | > 10% | > 20% |

### RL v2 vs v1 Improvements

| Metric | RL v1 | RL v2 Target | Improvement |
|--------|-------|--------------|-------------|
| Regime Awareness | ❌ No | ✅ Yes | +15% PnL |
| Volatility Adaptation | Limited | Advanced | -20% drawdown |
| Position Sizing | Static | Dynamic | +10% Sharpe |
| State Richness | 3 features | 11 features | Better decisions |
| Action Space | 20 actions | 100 actions | Finer control |

---

## 📚 Documentation Links

- **Implementation**: `docs/RL_V2_IMPLEMENTATION.md`
- **Verification Report**: `docs/RL_V2_VERIFICATION_REPORT.md`
- **Quick Reference**: `docs/RL_V2_QUICK_REFERENCE.md`
- **This Guide**: `docs/RL_V2_DEPLOYMENT_OPERATIONS.md`

---

## ✅ Deployment Checklist

### Pre-Deployment

- [x] All 15 RL v2 files implemented
- [x] Integration tests passing (100%)
- [x] Q-learning formulas verified
- [x] Event integration tested
- [x] Documentation complete

### Deployment

- [ ] Backend started with RL v2 enabled
- [ ] Q-tables initialized in `data/rl_v2/`
- [ ] Monitoring dashboard running
- [ ] Initial metrics recorded
- [ ] Alerts configured

### Post-Deployment (First Week)

- [ ] Day 1: Verify Q-table growth
- [ ] Day 3: Check hyperparameter recommendations
- [ ] Day 7: Run first A/B test vs RL v1
- [ ] Day 7: Backup Q-tables
- [ ] Day 7: Review performance metrics

### Ongoing Operations

- [ ] Daily: Monitor dashboard
- [ ] Weekly: Review hyperparameters
- [ ] Monthly: A/B test vs RL v1
- [ ] Quarterly: Consider advanced features (DQN, PPO)

---

## 🎯 Success Criteria

**RL v2 deployment is successful when**:

1. ✅ Q-tables growing consistently (new states daily)
2. ✅ Win rate ≥ RL v1 win rate
3. ✅ Sharpe ratio ≥ RL v1 Sharpe ratio
4. ✅ Max drawdown ≤ RL v1 drawdown
5. ✅ Agent exploiting more than exploring (epsilon < 0.1)
6. ✅ Q-values converging (variance decreasing)
7. ✅ No critical errors in logs

**When to revert to RL v1**:

- ⚠️ RL v2 consistently underperforms RL v1 (3+ A/B tests)
- ⚠️ Q-tables corrupt or not saving
- ⚠️ Critical bugs affecting trading decisions
- ⚠️ Resource usage too high (> 10 MB Q-tables without improvement)

---

## 📞 Support

**Issues/Questions**: Check logs in `logs/` directory  
**Performance Problems**: Run `python scripts/tune_rl_v2_hyperparams.py`  
**Bugs**: Review `docs/RL_V2_VERIFICATION_REPORT.md` for known issues

---

**System Status**: 🟢 PRODUCTION READY  
**Deployment Date**: December 2, 2025  
**Next Review**: January 2, 2026
