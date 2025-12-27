# AdaptiveLeverageEngine - Usage Guide

## 📊 Overview

The **AdaptiveLeverageEngine** provides leverage-aware dynamic TP/SL calculation with multi-stage profit harvesting. It replaces static TP/SL percentages with adaptive levels that scale intelligently based on leverage, volatility, and market conditions.

---

## 🎯 Core Features

### 1. **Leverage Scaling Factor (LSF)**
```python
LSF = 1 / (1 + log(leverage + 1))
```
- **High leverage (50x):** LSF ≈ 0.25 → Tighter TP, wider SL
- **Low leverage (5x):** LSF ≈ 0.55 → Wider TP, tighter SL
- **Effect:** Automatically protects high-leverage positions

### 2. **Multi-Stage Take Profit**
```python
TP1 = base_tp × (0.6 + LSF)      # Conservative
TP2 = base_tp × (1.2 + LSF/2)    # Aggressive  
TP3 = base_tp × (1.8 + LSF/4)    # Very Aggressive
```
- **TP1:** Quick profit capture (30-50% of position)
- **TP2:** Balanced profit (30-40% of position)
- **TP3:** Moonshot target (20-40% of position)

### 3. **Dynamic Stop Loss**
```python
SL = base_sl × (1 + (1 - LSF) × 0.8)
```
- **High leverage:** Wider SL to avoid premature stops
- **Low leverage:** Tighter SL for capital protection
- **Clamps:** Always between 0.1% - 2.0%

### 4. **Harvest Schemes**
| Leverage | Scheme | Description |
|----------|--------|-------------|
| ≤10x | [0.3, 0.3, 0.4] | Conservative (larger TP3) |
| ≤30x | [0.4, 0.4, 0.2] | Aggressive (balanced) |
| >30x | [0.5, 0.3, 0.2] | Ultra-aggressive (front-loaded) |

---

## 🚀 Quick Start

### Integration in Trading Signal
```python
from microservices.exitbrain_v3_5.exit_brain import ExitBrainV35

# Initialize ExitBrain
exit_brain = ExitBrainV35(redis_client)

# Build exit plan (adaptive levels included automatically)
exit_plan = exit_brain.build_exit_plan(signal_data)

# Access adaptive levels
tp1_pct = exit_plan['tp1_pct']
tp2_pct = exit_plan['tp2_pct']
tp3_pct = exit_plan['tp3_pct']
sl_pct = exit_plan['sl_pct']
harvest = exit_plan['calculation_details']['adaptive_levels']['harvest_scheme']
```

### Direct Engine Usage
```python
from microservices.exitbrain_v3_5.adaptive_leverage_engine import AdaptiveLeverageEngine

# Initialize engine
engine = AdaptiveLeverageEngine(
    base_tp_pct=0.01,    # 1% base TP
    base_sl_pct=0.005    # 0.5% base SL
)

# Compute adaptive levels
levels = engine.compute_levels(
    leverage=20.0,
    base_tp=0.01,
    base_sl=0.005,
    volatility_factor=0.3,
    funding_rate_factor=0.0001,
    divergence_factor=0.05
)

print(f"LSF: {levels.lsf:.4f}")
print(f"TP1: {levels.tp1:.4f} ({levels.tp1*100:.2f}%)")
print(f"TP2: {levels.tp2:.4f} ({levels.tp2*100:.2f}%)")
print(f"TP3: {levels.tp3:.4f} ({levels.tp3*100:.2f}%)")
print(f"SL: {levels.sl:.4f} ({levels.sl*100:.2f}%)")
print(f"Harvest: {levels.harvest_scheme}")
```

---

## 📈 Production Monitoring

### 1. **Analysis Mode** (Statistics Report)
```bash
# Get last 100 calculations
python monitor_adaptive_leverage.py 100

# Get last 500 calculations
python monitor_adaptive_leverage.py 500
```

**Output Sections:**
- Overall statistics (total calculations, clamp frequencies)
- Per-symbol breakdown (top 10 by activity)
- Recent examples (last 5 with full details)
- Harvest scheme distribution

### 2. **Watch Mode** (Real-Time)
```bash
python monitor_adaptive_leverage.py watch
```

**Features:**
- Real-time stream monitoring (30s refresh)
- Instant alerts for SL_CLAMPED, TP_MIN events
- Color-coded warnings
- Continuous statistics

### 3. **Redis Stream Schema**
**Stream Key:** `quantum:stream:adaptive_levels`

**Fields:**
```python
{
    'timestamp': '2024-12-19T10:30:45.123456',
    'symbol': 'BTCUSDT',
    'side': 'LONG',
    'leverage': '20.0',
    'lsf': '0.2472',
    'tp1_pct': '0.0085',  # 0.85%
    'tp2_pct': '0.0132',  # 1.32%
    'tp3_pct': '0.0186',  # 1.86%
    'sl_pct': '0.0080',   # 0.80%
    'harvest_scheme': '[0.4, 0.4, 0.2]',
    'sl_clamped': 'False',
    'tp_minimum_enforced': 'False'
}
```

### 4. **Query Stream with Redis CLI**
```bash
# Get last 10 entries
redis-cli XREVRANGE quantum:stream:adaptive_levels + - COUNT 10

# Get all entries since timestamp
redis-cli XREAD COUNT 100 STREAMS quantum:stream:adaptive_levels 1734603045000-0
```

---

## 🔧 Configuration Tuning

### Edit Configuration
```bash
vim microservices/exitbrain_v3_5/adaptive_leverage_config.py
```

### Key Parameters

#### Base Percentages (Most Common Adjustments)
```python
BASE_TP_PCT = 0.01    # 1% - Increase for more aggressive TP
BASE_SL_PCT = 0.005   # 0.5% - Increase for wider stops
```

#### Scaling Factors
```python
FUNDING_TP_SCALE = 0.8      # How much funding affects TP
DIVERGENCE_SL_SCALE = 0.4   # How much divergence widens SL
VOLATILITY_SL_SCALE = 0.2   # How much volatility widens SL
```

#### Harvest Schemes
```python
HARVEST_LOW_LEVERAGE = [0.3, 0.3, 0.4]    # ≤10x
HARVEST_MID_LEVERAGE = [0.4, 0.4, 0.2]    # ≤30x
HARVEST_HIGH_LEVERAGE = [0.5, 0.3, 0.2]   # >30x
```

### Validate Configuration
```bash
python microservices/exitbrain_v3_5/adaptive_leverage_config.py
```

---

## 🧪 Testing

### Run Unit Tests
```bash
# All tests
cd c:/quantum_trader
python -m pytest microservices/exitbrain_v3_5/tests/test_adaptive_leverage_engine.py -v

# Specific test
python -m pytest microservices/exitbrain_v3_5/tests/test_adaptive_leverage_engine.py::test_lsf_decreases_with_leverage -v
```

### VPS Validation
```bash
# On VPS
cd ~/quantum_trader
bash validate_adaptive_leverage.sh
```

**Expected Output:**
```
[1] ✅ AdaptiveLeverageEngine class found
[2] ✅ compute_levels integrated in ExitBrain
[3] ✅ Import successful! 20x Leverage: TP1=0.85%, TP2=1.33%, TP3=1.87%, SL=0.80%
[4] ✅ ALL TESTS PASSED
```

---

## 📊 Example Calculations

### Low Leverage (5x)
```
LSF: 0.5531
TP1: 1.15% (0.6 + 0.5531 = 1.1531x base)
TP2: 1.48% (1.2 + 0.2765 = 1.4765x base)
TP3: 1.94% (1.8 + 0.1382 = 1.9382x base)
SL: 0.68% (1 + (1-0.5531)×0.8 = 1.3575x base)
Harvest: [0.3, 0.3, 0.4] (Conservative)
```

### Medium Leverage (20x)
```
LSF: 0.2472
TP1: 0.85% (0.6 + 0.2472 = 0.8472x base)
TP2: 1.32% (1.2 + 0.1236 = 1.3236x base)
TP3: 1.86% (1.8 + 0.0618 = 1.8618x base)
SL: 0.80% (1 + (1-0.2472)×0.8 = 1.6022x base)
Harvest: [0.4, 0.4, 0.2] (Aggressive)
```

### High Leverage (50x)
```
LSF: 0.2085
TP1: 0.81% (0.6 + 0.2085 = 0.8085x base)
TP2: 1.30% (1.2 + 0.1042 = 1.3042x base)
TP3: 1.85% (1.8 + 0.0521 = 1.8521x base)
SL: 0.82% (1 + (1-0.2085)×0.8 = 1.6332x base)
Harvest: [0.5, 0.3, 0.2] (Ultra-aggressive)
```

---

## 🚨 Monitoring Alerts

### When to Tune Parameters

#### SL Clamps Triggered (>10% of calculations)
```
⚠️ SL clamps active in 15% of calculations
→ Increase BASE_SL_PCT from 0.005 to 0.007
```

#### TP Minimums Enforced (>5% of calculations)
```
⚠️ TP minimums enforced in 8% of calculations
→ Increase BASE_TP_PCT from 0.01 to 0.012
```

#### Frequent Funding Adjustments
```
⚠️ Large funding rate swings observed
→ Reduce FUNDING_TP_SCALE from 0.8 to 0.6
```

#### High Volatility Environments
```
⚠️ Volatility consistently >0.5
→ Increase VOLATILITY_SL_SCALE from 0.2 to 0.3
```

---

## 📚 Integration Points

### Position Monitor
```python
# backend/services/monitoring/position_monitor.py
from backend.domains.exits.exit_brain_v3.v35_integration import get_v35_integration

v35 = get_v35_integration()
adaptive_levels = v35.compute_adaptive_levels(leverage, volatility_factor)
# Use adaptive_levels for dynamic TP/SL updates
```

### Event-Driven Executor
```python
# backend/services/execution/event_driven_executor.py
from backend.domains.exits.exit_brain_v3.exit_router import ExitRouter

exit_router = ExitRouter(redis_client)
exit_plan = exit_router.build_exit_plan(signal_data)
# exit_plan includes adaptive TP/SL levels
```

---

## 🔗 Related Documentation

- [AI_EXIT_BRAIN_DYNAMIC_TP_IMPLEMENTATION.md](AI_EXIT_BRAIN_DYNAMIC_TP_IMPLEMENTATION.md) - Exit Brain v3 Overview
- [AI_EXIT_BRAIN_V3_TP_PROFILES.md](AI_EXIT_BRAIN_V3_TP_PROFILES.md) - TP Profile System
- [AI_LEVERAGE_AWARE_RISK_MANAGEMENT.md](AI_LEVERAGE_AWARE_RISK_MANAGEMENT.md) - Risk Management System

---

## 💡 Tips & Best Practices

1. **Start Conservative:** Begin with default config, tune based on data
2. **Monitor First Week:** Run `watch` mode continuously during initial rollout
3. **Gradual Tuning:** Change one parameter at a time, monitor for 24h
4. **Test in Staging:** Always validate changes in test environment first
5. **Backup Config:** Git commit config changes with clear messages
6. **Document Tuning:** Add comments explaining why parameters were changed

---

## 🆘 Troubleshooting

### Issue: SL too tight, positions stopped out frequently
**Solution:** Increase `BASE_SL_PCT` or `SL_LEVERAGE_FACTOR`

### Issue: TP levels not being reached
**Solution:** Decrease `BASE_TP_PCT` or adjust `TP_PROGRESSION_FACTOR`

### Issue: High leverage positions too risky
**Solution:** Adjust `HARVEST_HIGH_LEVERAGE` to front-load harvesting

### Issue: Redis stream not populating
**Solution:** Check `ENABLE_ADAPTIVE_STREAM = True` in config

### Issue: Tests failing after config change
**Solution:** Run `validate_config()` to check parameter validity

---

## 📞 Support

For issues or questions:
1. Check monitoring logs: `grep "Adaptive Levels" logs/exitbrain_v3.log`
2. Run validation: `bash validate_adaptive_leverage.sh`
3. Check Redis stream: `redis-cli XINFO STREAM quantum:stream:adaptive_levels`
4. Review test output: `pytest -v test_adaptive_leverage_engine.py`

---

**Last Updated:** 2024-12-19  
**Version:** v3.5  
**Status:** Production Ready ✅
