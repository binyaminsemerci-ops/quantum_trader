# Strategy Runtime Engine - Quick Reference

## One-Line Summary
**Bridges AI-generated strategies from SG AI to live trading execution.**

## Status Check
```bash
# Quick health check
python -c "from backend.services.strategy_runtime_integration import check_strategy_runtime_health; import json; print(json.dumps(check_strategy_runtime_health(), indent=2))"
```

## Log Markers
Look for these in Event-Driven Executor logs:

```
[OK] Strategy Runtime Engine initialized: X active strategies
[STRATEGY] Generated Y signals from Strategy Runtime Engine
[STRATEGY] Converted to Z signals: BTCUSDT: LONG @ 85% ...
[SIGNAL] Merged signals: N total (M AI + Z strategy)
```

## Control Policies

### Set Risk Mode
```python
from backend.services.strategy_runtime_integration import QuantumPolicyStore
store = QuantumPolicyStore()

store.set_risk_mode("NORMAL")      # Default
store.set_risk_mode("AGGRESSIVE")  # Higher risk
store.set_risk_mode("DEFENSIVE")   # Lower risk
```

### Set Confidence Threshold
```python
store.set_global_min_confidence(0.50)  # Default
store.set_global_min_confidence(0.65)  # Higher quality signals
store.set_global_min_confidence(0.40)  # More signals
```

### Enable Specific Strategies Only
```python
# Empty list = all strategies allowed
store.set_allowed_strategies([])

# Only allow specific strategies
store.set_allowed_strategies([
    "rsi-oversold-123",
    "macd-cross-456"
])
```

## Prometheus Metrics

Access at: `http://localhost:8000/metrics`

```promql
# Total signals generated
strategy_runtime_signals_generated_total

# Active strategy count
strategy_runtime_active_strategies

# Signal confidence distribution
strategy_runtime_signal_confidence

# Evaluation performance
strategy_runtime_evaluation_duration_seconds
```

## Common Operations

### Check Active Strategies
```sql
SELECT strategy_id, name, status 
FROM sg_strategies 
WHERE status = 'LIVE';
```

### Promote Strategy to LIVE
```sql
UPDATE sg_strategies 
SET status = 'LIVE' 
WHERE strategy_id = 'your-strategy-id';
```

### Pause Strategy
```sql
UPDATE sg_strategies 
SET status = 'PAUSED' 
WHERE strategy_id = 'your-strategy-id';
```

### Archive Strategy
```sql
UPDATE sg_strategies 
SET status = 'ARCHIVED' 
WHERE strategy_id = 'your-strategy-id';
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No signals | Check for LIVE strategies in database |
| Import error | `pip install redis` |
| Redis error | Normal - uses DB fallback |
| Binance error | Check API credentials |

## Emergency Actions

### Disable All Strategy Signals
```python
# Option 1: Pause all strategies in DB
UPDATE sg_strategies SET status = 'PAUSED' WHERE status = 'LIVE';

# Option 2: Set impossible confidence
from backend.services.strategy_runtime_integration import QuantumPolicyStore
QuantumPolicyStore().set_global_min_confidence(1.01)
```

### Re-enable
```python
# Restore confidence threshold
QuantumPolicyStore().set_global_min_confidence(0.50)

# Re-activate strategies
UPDATE sg_strategies SET status = 'LIVE' WHERE status = 'PAUSED';
```

## Files

| File | Purpose |
|------|---------|
| `backend/services/strategy_runtime_integration.py` | Main integration code |
| `backend/services/event_driven_executor.py` | Executor with integration |
| `migrations/add_policy_store_table.sql` | Database schema |
| `STRATEGY_RUNTIME_PRODUCTION_DEPLOYMENT.md` | Full documentation |
| `test_integration_simple.py` | Quick validation test |

## Support Contacts

- **Architecture Questions:** See `STRATEGY_RUNTIME_PRODUCTION_DEPLOYMENT.md`
- **Integration Issues:** Check `backend/services/strategy_runtime_integration.py` logs
- **Database Issues:** Review `migrations/add_policy_store_table.sql`

---

**Quick Start:** `python test_integration_simple.py`  
**Full Docs:** `STRATEGY_RUNTIME_PRODUCTION_DEPLOYMENT.md`  
**Status:** OPERATIONAL ✅
