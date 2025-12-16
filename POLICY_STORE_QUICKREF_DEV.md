# PolicyStore Developer Quick Reference 🚀

## 5-Second Overview
**PolicyStore** = Central config hub for all AI components in Quantum Trader. Thread-safe, validated, always available.

---

## Essential Code Snippets

### 1. Read Policy (Any Route Handler)
```python
from fastapi import Request

@app.get("/my-endpoint")
async def handler(request: Request):
    policy_store = request.app.state.policy_store
    policy = policy_store.get()
    
    risk_mode = policy['risk_mode']              # "AGGRESSIVE" | "NORMAL" | "DEFENSIVE"
    max_risk = policy['max_risk_per_trade']      # 0.01 - 0.05
    max_pos = policy['max_positions']            # 3 - 10
    min_conf = policy['global_min_confidence']   # 0.6 - 0.8
    
    return {"risk_mode": risk_mode}
```

### 2. Update Policy (Single Field)
```python
# Simple patch
policy_store.patch({'risk_mode': 'AGGRESSIVE'})

# Multiple fields
policy_store.patch({
    'max_risk_per_trade': 0.02,
    'max_positions': 8,
    'global_min_confidence': 0.75
})
```

### 3. Update Nested Data (Rankings, Model Versions)
```python
# Update opportunity rankings (deep merge)
policy_store.patch({
    'opp_rankings': {
        'BTCUSDT': 0.95,
        'ETHUSDT': 0.87,
        'SOLUSDT': 0.82
    }
})

# Update model versions
policy_store.patch({
    'model_versions': {
        'lstm_v1': '2024.01.15',
        'transformer_v2': '2024.01.10'
    }
})
```

### 4. Read Specific Field
```python
policy = policy_store.get()

# Risk parameters
if policy['risk_mode'] == 'AGGRESSIVE':
    # Use aggressive settings
    pass

# Symbol filtering
allowed = policy['allowed_symbols']
if not allowed or 'BTCUSDT' in allowed:
    # Symbol is allowed
    pass

# Strategy filtering
strategies = policy['allowed_strategies']
if 'momentum' in strategies:
    # Use momentum strategy
    pass
```

### 5. Full Policy Replacement
```python
# Replace entire policy (use sparingly)
policy_store.update({
    'risk_mode': 'DEFENSIVE',
    'allowed_strategies': ['mean_reversion'],
    'allowed_symbols': ['BTCUSDT', 'ETHUSDT'],
    'max_risk_per_trade': 0.005,
    'max_positions': 3,
    'global_min_confidence': 0.8,
    'opp_rankings': {},
    'model_versions': {}
})
```

### 6. Reset to Defaults
```python
# Nuclear option: restore defaults from environment
policy_store.reset()
```

---

## HTTP API Cheat Sheet

```bash
# Status check
curl http://localhost:8000/api/policy/status

# Get full policy
curl http://localhost:8000/api/policy

# Change risk mode
curl -X POST http://localhost:8000/api/policy/risk_mode/AGGRESSIVE

# Update fields
curl -X PATCH http://localhost:8000/api/policy \
  -H "Content-Type: application/json" \
  -d '{"max_risk_per_trade": 0.02, "max_positions": 8}'

# Reset
curl -X POST http://localhost:8000/api/policy/reset

# Get risk mode only
curl http://localhost:8000/api/policy/risk_mode

# Get allowed symbols
curl http://localhost:8000/api/policy/allowed_symbols

# Get model versions
curl http://localhost:8000/api/policy/model_versions
```

---

## Policy Fields Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_mode` | str | "NORMAL" | AGGRESSIVE, NORMAL, or DEFENSIVE |
| `allowed_strategies` | list[str] | [] | Whitelist of strategies (empty = all) |
| `allowed_symbols` | list[str] | [] | Whitelist of symbols (empty = all) |
| `max_risk_per_trade` | float | 0.01 | 0.005 (DEFENSIVE) to 0.02 (AGGRESSIVE) |
| `max_positions` | int | 5 | 3 (DEFENSIVE) to 10 (AGGRESSIVE) |
| `global_min_confidence` | float | 0.7 | 0.75 (DEFENSIVE) to 0.65 (AGGRESSIVE) |
| `opp_rankings` | dict | {} | {symbol: score} from OpportunityRanker |
| `model_versions` | dict | {} | {model_name: version} from CLM |
| `last_updated` | str | (auto) | ISO timestamp of last update |

---

## Risk Mode Presets

| Mode | Max Risk | Max Pos | Min Conf | Use Case |
|------|----------|---------|----------|----------|
| **AGGRESSIVE** | 0.020 | 10 | 0.65 | Bull market, high conviction |
| **NORMAL** | 0.010 | 5 | 0.70 | Balanced trading |
| **DEFENSIVE** | 0.005 | 3 | 0.75 | Bear market, uncertainty |

---

## Integration Patterns

### MSC AI → PolicyStore
```python
# After MSC AI determines new risk parameters
def update_from_msc_ai(policy_store, msc_decision):
    policy_store.patch({
        'risk_mode': msc_decision.risk_mode,
        'max_risk_per_trade': msc_decision.max_risk,
        'max_positions': msc_decision.max_positions,
        'global_min_confidence': msc_decision.min_confidence
    })
```

### OpportunityRanker → PolicyStore
```python
# After ranking opportunities
def update_rankings(policy_store, rankings):
    policy_store.patch({
        'opp_rankings': rankings  # {'BTCUSDT': 0.95, ...}
    })
```

### RiskGuard ← PolicyStore
```python
# Read risk limits
def check_risk(policy_store, proposed_trade):
    policy = policy_store.get()
    
    if proposed_trade.risk > policy['max_risk_per_trade']:
        return False  # Reject
    
    if current_positions >= policy['max_positions']:
        return False  # Reject
    
    return True  # Approve
```

### Orchestrator ← PolicyStore
```python
# Read confidence threshold and rankings
def select_signals(policy_store, signals):
    policy = policy_store.get()
    
    # Filter by confidence
    filtered = [s for s in signals 
                if s.confidence >= policy['global_min_confidence']]
    
    # Sort by opportunity ranking
    rankings = policy['opp_rankings']
    filtered.sort(key=lambda s: rankings.get(s.symbol, 0), reverse=True)
    
    return filtered[:policy['max_positions']]
```

---

## Thread Safety

✅ **All operations are thread-safe** (uses `threading.RLock()`)

```python
# Safe to call from multiple threads simultaneously
import threading

def worker(policy_store):
    policy_store.patch({'risk_mode': 'AGGRESSIVE'})

threads = [threading.Thread(target=worker, args=(policy_store,)) 
           for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# No race conditions!
```

---

## Validation Rules

**Automatic validation on every update:**

- `risk_mode`: Must be "AGGRESSIVE", "NORMAL", or "DEFENSIVE"
- `max_risk_per_trade`: Must be 0 ≤ value ≤ 1
- `max_positions`: Must be integer ≥ 1
- `global_min_confidence`: Must be 0 ≤ value ≤ 1
- `opp_rankings`: Values must be 0 ≤ score ≤ 1
- `allowed_strategies`: Must be list of strings
- `allowed_symbols`: Must be list of strings
- `model_versions`: Must be dict[str, str]

**Invalid updates are rejected immediately with clear error messages.**

---

## Common Mistakes

### ❌ DON'T: Mutate returned dict
```python
policy = policy_store.get()
policy['risk_mode'] = 'AGGRESSIVE'  # ❌ No effect!
```

### ✅ DO: Use patch()
```python
policy_store.patch({'risk_mode': 'AGGRESSIVE'})  # ✅ Works!
```

### ❌ DON'T: Partial nested updates with update()
```python
policy_store.update({'opp_rankings': {'BTCUSDT': 0.95}})  # ❌ Replaces entire dict!
```

### ✅ DO: Use patch() for nested updates
```python
policy_store.patch({'opp_rankings': {'BTCUSDT': 0.95}})  # ✅ Merges!
```

---

## Testing

```bash
# Unit tests
cd backend/services
python -m pytest test_policy_store.py -v

# Integration test
python test_policy_api.py
```

---

## Where Is It?

| What | Where |
|------|-------|
| **Core Implementation** | `backend/services/policy_store.py` |
| **Unit Tests** | `backend/services/test_policy_store.py` |
| **HTTP API** | `backend/routes/policy.py` |
| **Integration Test** | `test_policy_api.py` |
| **Documentation** | `POLICY_STORE_README.md` |
| **Examples** | `backend/services/policy_store_examples.py` |
| **This File** | `POLICY_STORE_QUICKREF_DEV.md` |

---

## Getting Started in 30 Seconds

1. **PolicyStore is already running** (initialized in `backend/main.py`)

2. **Access it anywhere:**
   ```python
   policy_store = request.app.state.policy_store
   policy = policy_store.get()
   ```

3. **Update it:**
   ```python
   policy_store.patch({'risk_mode': 'AGGRESSIVE'})
   ```

4. **Test it:**
   ```bash
   curl http://localhost:8000/api/policy
   ```

**Done!** 🎉

---

## Key Principles

1. **Single Source of Truth** - All config in one place
2. **Thread-Safe** - No race conditions
3. **Validated** - Invalid updates rejected immediately
4. **Atomic** - Updates are all-or-nothing
5. **Timestamped** - Every change tracked
6. **Deep Merge** - Nested dicts merge intelligently
7. **Always Available** - Initialized at startup

---

## Need Help?

- **Full Documentation**: `POLICY_STORE_README.md`
- **Examples**: `backend/services/policy_store_examples.py`
- **Integration Demo**: `backend/services/policy_store_integration_demo.py`
- **Architecture**: `POLICY_STORE_ARCHITECTURE_DIAGRAM.md`
- **Complete Guide**: `POLICY_STORE_INTEGRATION_COMPLETE.md`

---

**Last Updated**: 2024-01-15  
**Version**: 1.0.0  
**Status**: ✅ Production Ready
