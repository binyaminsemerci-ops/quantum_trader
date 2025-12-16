# PolicyStore Quick Reference

## Import

```python
from backend.services.policy_store import (
    InMemoryPolicyStore,
    PolicyDefaults,
    GlobalPolicy,
    PolicyValidationError,
)
```

## Initialization

```python
# Default policy
store = InMemoryPolicyStore()

# Custom initial policy
store = InMemoryPolicyStore(initial_policy=PolicyDefaults.create_aggressive())
```

## Basic Operations

### Read

```python
# As dict
policy = store.get()
risk_mode = policy['risk_mode']
max_pos = policy['max_positions']

# As typed object
policy_obj = store.get_policy_object()
risk_mode = policy_obj.risk_mode
max_pos = policy_obj.max_positions
```

### Write

```python
# Full update (replace everything)
store.update({
    "risk_mode": "NORMAL",
    "allowed_strategies": ["STRAT_1", "STRAT_2"],
    "max_risk_per_trade": 0.01,
    "max_positions": 10,
    "global_min_confidence": 0.65,
    "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
    "opp_rankings": {"BTCUSDT": 0.9},
    "model_versions": {"xgboost": "v14"},
})

# Partial update (patch specific fields)
store.patch({
    "risk_mode": "AGGRESSIVE",
    "max_risk_per_trade": 0.015,
})

# Reset to default
store.reset()
```

## Policy Fields

| Field | Type | Example | Who Writes |
|-------|------|---------|------------|
| `risk_mode` | str | "AGGRESSIVE" | MSC AI |
| `allowed_strategies` | list[str] | ["STRAT_1", "STRAT_2"] | MSC AI, SG AI |
| `allowed_symbols` | list[str] | ["BTCUSDT", "ETHUSDT"] | OppRank |
| `max_risk_per_trade` | float | 0.01 | MSC AI |
| `max_positions` | int | 10 | MSC AI |
| `global_min_confidence` | float | 0.65 | MSC AI |
| `opp_rankings` | dict[str, float] | {"BTCUSDT": 0.9} | OppRank |
| `model_versions` | dict[str, str] | {"xgboost": "v14"} | CLM |
| `system_health` | dict | {"status": "ok"} | Monitor |
| `custom_params` | dict | {"key": "value"} | Any |

## Common Patterns

### MSC AI: Change Risk Mode

```python
if market_regime == "TRENDING":
    store.patch({
        "risk_mode": "AGGRESSIVE",
        "max_risk_per_trade": 0.015,
        "max_positions": 12,
        "global_min_confidence": 0.60,
    })
elif market_regime == "CHOPPY":
    store.patch({
        "risk_mode": "DEFENSIVE",
        "max_risk_per_trade": 0.005,
        "max_positions": 5,
        "global_min_confidence": 0.75,
    })
```

### OppRank: Update Rankings

```python
rankings = {
    "BTCUSDT": 0.92,
    "ETHUSDT": 0.88,
    "SOLUSDT": 0.85,
}

store.patch({
    "opp_rankings": rankings,
    "allowed_symbols": list(rankings.keys()),
})
```

### CLM: Promote Model

```python
if new_model_better_than_old:
    store.patch({
        "model_versions": {
            "xgboost": "v15",
        }
    })
```

### Strategy Generator: Add/Remove Strategy

```python
# Add strategy
policy = store.get()
new_strategies = policy['allowed_strategies'] + ["STRAT_42"]
store.patch({"allowed_strategies": new_strategies})

# Remove strategy
policy = store.get()
updated = [s for s in policy['allowed_strategies'] if s != "STRAT_7"]
store.patch({"allowed_strategies": updated})
```

### Orchestrator: Check Signal

```python
policy = store.get()

# Check strategy
if signal.strategy_id not in policy['allowed_strategies']:
    return False

# Check symbol
if signal.symbol not in policy['allowed_symbols']:
    return False

# Check confidence
if signal.confidence < policy['global_min_confidence']:
    return False

return True
```

### RiskGuard: Validate Risk

```python
policy = store.get()
risk_fraction = position_size / account_balance

if risk_fraction > policy['max_risk_per_trade']:
    return "REJECT"

return "OK"
```

### Portfolio Balancer: Check Capacity

```python
policy = store.get()

if len(open_positions) >= policy['max_positions']:
    return "AT_CAPACITY"

return "OK"
```

## Validation Rules

```python
# Risk mode
risk_mode ∈ {"AGGRESSIVE", "NORMAL", "DEFENSIVE"}

# Numeric ranges
0 < max_risk_per_trade ≤ 1
1 ≤ max_positions ≤ 100
0 ≤ global_min_confidence ≤ 1

# Opp rankings
∀ symbol: 0 ≤ opp_rankings[symbol] ≤ 1

# Types
allowed_strategies: list[str]
allowed_symbols: list[str]
opp_rankings: dict[str, float]
model_versions: dict[str, str]
```

## Error Handling

```python
from policy_store import PolicyValidationError

try:
    store.update(untrusted_policy)
except PolicyValidationError as e:
    logger.error(f"Invalid policy: {e}")
    store.reset()  # Fallback to safe defaults
```

## Default Presets

```python
from policy_store import PolicyDefaults

# Balanced (default)
default = PolicyDefaults.create_default()
# risk_mode="NORMAL", max_risk=0.01, max_pos=10

# Conservative
conservative = PolicyDefaults.create_conservative()
# risk_mode="DEFENSIVE", max_risk=0.005, max_pos=5

# Aggressive
aggressive = PolicyDefaults.create_aggressive()
# risk_mode="AGGRESSIVE", max_risk=0.02, max_pos=15
```

## Component Initialization

```python
# Initialize store once at startup
policy_store = InMemoryPolicyStore()

# Load initial policy (from config or MSC AI)
policy_store.update(initial_policy_dict)

# Pass to all components
orchestrator = OrchestratorPolicy(policy_store)
risk_guard = RiskGuard(policy_store, account_balance)
portfolio = PortfolioBalancer(policy_store)
ensemble = EnsembleManager(policy_store)
safety = SafetyGovernor(policy_store)
```

## Thread Safety

```python
# Safe: Concurrent reads
threads = [Thread(target=lambda: store.get()) for _ in range(100)]

# Safe: Concurrent writes (atomic)
def update_mode(mode):
    store.patch({"risk_mode": mode})

threads = [Thread(target=update_mode, args=(m,)) for m in modes]
```

## Best Practices

✅ **Use `patch()` for incremental updates**  
✅ **Use `update()` only for full replacement**  
✅ **Cache policy for batch operations**  
✅ **Let timestamps auto-update**  
✅ **Validate before expensive operations**  
✅ **Use typed access (`get_policy_object()`) when possible**  

❌ **Don't read policy on every iteration**  
❌ **Don't manually set `last_updated`**  
❌ **Don't mutate returned dicts (they're copies anyway)**  

## Testing

```python
def test_my_component():
    # Use in-memory store for tests
    store = InMemoryPolicyStore()
    
    # Set test policy
    store.update({
        "risk_mode": "NORMAL",
        "max_risk_per_trade": 0.01,
    })
    
    # Test component
    component = MyComponent(store)
    assert component.do_something() == expected
```

## Monitoring

```python
# Log policy changes
old = store.get()
store.patch(changes)
new = store.get()

if old['risk_mode'] != new['risk_mode']:
    logger.info(f"Risk mode: {old['risk_mode']} → {new['risk_mode']}")
```

## File Locations

```
backend/services/
├── policy_store.py                    # Core implementation
├── test_policy_store.py               # Test suite
├── policy_store_examples.py           # Usage examples
├── policy_store_integration_demo.py   # Integration demo
└── POLICY_STORE_README.md            # Full documentation
```

## Quick Start

```python
# 1. Import
from backend.services.policy_store import InMemoryPolicyStore

# 2. Initialize
store = InMemoryPolicyStore()

# 3. Set policy
store.update({
    "risk_mode": "NORMAL",
    "allowed_strategies": ["STRAT_1"],
    "max_risk_per_trade": 0.01,
    "max_positions": 10,
    "global_min_confidence": 0.65,
})

# 4. Read policy
policy = store.get()
print(policy['risk_mode'])  # "NORMAL"

# 5. Update policy
store.patch({"risk_mode": "AGGRESSIVE"})
```

---

**Need help?** See `POLICY_STORE_README.md` for complete documentation.
