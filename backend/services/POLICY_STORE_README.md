# PolicyStore Documentation

## Overview

**PolicyStore** is the central configuration and state management system for Quantum Trader AI. It acts as the single source of truth for all global trading parameters, ensuring coordinated decision-making across all AI components.

## Architecture

### Core Components

```
PolicyStore (Protocol)
├── GlobalPolicy (dataclass)
├── PolicyValidator
├── PolicySerializer
├── PolicyMerger
├── PolicyDefaults
└── Implementations
    ├── InMemoryPolicyStore
    ├── PostgresPolicyStore
    ├── RedisPolicyStore
    └── SQLitePolicyStore
```

### Design Principles

- **Single Source of Truth**: All components read from one authoritative source
- **Atomic Operations**: Thread-safe updates with consistency guarantees
- **Backend Agnostic**: Clean separation between interface and storage
- **Type Safe**: Structured data with validation
- **Extensible**: Easy to add new fields or backends

## GlobalPolicy Structure

```python
@dataclass
class GlobalPolicy:
    risk_mode: str                        # "AGGRESSIVE" | "NORMAL" | "DEFENSIVE"
    allowed_strategies: list[str]          # Strategy IDs permitted to trade
    allowed_symbols: list[str]             # Tradeable symbols
    max_risk_per_trade: float             # Max capital fraction per trade (0-1)
    max_positions: int                     # Max concurrent positions (1-100)
    global_min_confidence: float          # Min confidence threshold (0-1)
    opp_rankings: dict[str, float]        # Symbol scores from OppRank
    model_versions: dict[str, str]        # Active model versions
    system_health: dict[str, Any]         # Health indicators
    custom_params: dict[str, Any]         # Extension point
    last_updated: str                      # ISO timestamp (auto-managed)
```

## Usage

### Basic Setup

```python
from policy_store import InMemoryPolicyStore, PolicyDefaults

# Create store with defaults
store = InMemoryPolicyStore()

# Or with custom initial policy
store = InMemoryPolicyStore(initial_policy=PolicyDefaults.create_aggressive())
```

### Reading Policy

```python
# Get as dictionary
policy_dict = store.get()
print(policy_dict['risk_mode'])

# Get as typed object
policy_obj = store.get_policy_object()
print(policy_obj.risk_mode)
```

### Updating Policy

```python
# Full update (replaces entire policy)
store.update({
    "risk_mode": "NORMAL",
    "allowed_strategies": ["STRAT_1", "STRAT_2"],
    "max_risk_per_trade": 0.01,
    "max_positions": 10,
    "global_min_confidence": 0.65,
    "allowed_symbols": ["BTCUSDT", "ETHUSDT"],
    "opp_rankings": {"BTCUSDT": 0.9, "ETHUSDT": 0.85},
    "model_versions": {"xgboost": "v14"},
})

# Partial update (patches specific fields)
store.patch({
    "risk_mode": "AGGRESSIVE",
    "max_risk_per_trade": 0.015,
})

# Nested dict merge (for opp_rankings, model_versions)
store.patch({
    "opp_rankings": {"SOLUSDT": 0.88},  # Adds to existing rankings
    "model_versions": {"lightgbm": "v11"},  # Adds to existing versions
})
```

### Reset

```python
# Reset to default empty policy
store.reset()
```

## Component Integration

### MSC AI (Meta Strategy Controller)

```python
# Read current state
policy = store.get()
current_mode = policy['risk_mode']

# Change risk mode based on market regime
if market_regime == "TRENDING" and volatility == "LOW":
    store.patch({
        "risk_mode": "AGGRESSIVE",
        "max_risk_per_trade": 0.015,
        "max_positions": 12,
        "global_min_confidence": 0.60,
    })
elif market_regime == "CHOPPY" or volatility == "HIGH":
    store.patch({
        "risk_mode": "DEFENSIVE",
        "max_risk_per_trade": 0.005,
        "max_positions": 5,
        "global_min_confidence": 0.75,
    })
```

### Opportunity Ranker

```python
# Update symbol rankings and allowed symbols
top_symbols = ranker.compute_rankings(all_symbols)

store.patch({
    "opp_rankings": top_symbols,  # Dict[str, float]
    "allowed_symbols": list(top_symbols.keys())[:10],  # Top 10
})
```

### Continuous Learning Manager

```python
# Promote new model version after shadow evaluation
if shadow_metrics['sharpe'] > production_metrics['sharpe']:
    store.patch({
        "model_versions": {
            "xgboost": "v15",  # New version
        }
    })
```

### Strategy Generator

```python
# Read allowed strategies
policy = store.get()
current_strategies = policy['allowed_strategies']

# Promote new strategy
if new_strategy_performance_ok:
    store.patch({
        "allowed_strategies": current_strategies + ["STRAT_42"],
    })

# Demote underperforming strategy
store.patch({
    "allowed_strategies": [s for s in current_strategies if s != "STRAT_7"],
})
```

### RiskGuard

```python
# Check proposed trade against policy
policy = store.get()

if proposed_risk > policy['max_risk_per_trade']:
    return "REJECT: Risk too high"

if signal_confidence < policy['global_min_confidence']:
    return "REJECT: Confidence too low"

if symbol not in policy['allowed_symbols']:
    return "REJECT: Symbol not allowed"

return "APPROVE"
```

### Orchestrator Policy

```python
# Validate signal against global policy
policy = store.get()

if signal.strategy_id not in policy['allowed_strategies']:
    return False  # Strategy not allowed

if signal.symbol not in policy['allowed_symbols']:
    return False  # Symbol not allowed

if signal.confidence < policy['global_min_confidence']:
    return False  # Confidence too low

return True
```

### Portfolio Balancer

```python
# Check position capacity
policy = store.get()
current_positions = len(get_open_positions())

if current_positions >= policy['max_positions']:
    return "REJECT: At max capacity"

return "OK"
```

## Validation

All updates are validated before being stored:

```python
# Valid risk modes
risk_mode ∈ {"AGGRESSIVE", "NORMAL", "DEFENSIVE"}

# Numeric ranges
0 < max_risk_per_trade ≤ 1
1 ≤ max_positions ≤ 100
0 ≤ global_min_confidence ≤ 1
0 ≤ opp_rankings[symbol] ≤ 1 (for all symbols)

# Type checks
allowed_strategies: list[str]
allowed_symbols: list[str]
opp_rankings: dict[str, float]
model_versions: dict[str, str]
```

Validation errors raise `PolicyValidationError`:

```python
try:
    store.update({"risk_mode": "INVALID"})
except PolicyValidationError as e:
    print(f"Validation failed: {e}")
```

## Thread Safety

All operations are thread-safe:

```python
# Concurrent reads (safe)
threads = [Thread(target=lambda: store.get()) for _ in range(100)]

# Concurrent writes (safe, atomic)
def update_risk(mode):
    store.patch({"risk_mode": mode})

threads = [
    Thread(target=update_risk, args=("AGGRESSIVE",)),
    Thread(target=update_risk, args=("DEFENSIVE",)),
]
```

Internally uses `threading.RLock()` for atomicity.

## Storage Backends

### In-Memory (Development/Testing)

```python
store = InMemoryPolicyStore()
```

**Pros**: Fast, no dependencies, perfect for tests  
**Cons**: Not persistent, single-process only

### PostgreSQL (Production)

```python
from psycopg2.pool import SimpleConnectionPool

pool = SimpleConnectionPool(1, 20, dsn="postgresql://...")
store = PostgresPolicyStore(connection_pool=pool)
```

**Pros**: ACID, high concurrency, JSONB queries  
**Cons**: Requires DB setup, network latency

**Schema**:
```sql
CREATE TABLE policy_store (
    id INTEGER PRIMARY KEY DEFAULT 1,
    policy_json JSONB NOT NULL,
    last_updated TIMESTAMP NOT NULL DEFAULT NOW(),
    CONSTRAINT single_row CHECK (id = 1)
);

CREATE INDEX idx_policy_last_updated ON policy_store(last_updated);
```

### Redis (High-Performance)

```python
import redis

client = redis.Redis(host='localhost', port=6379)
store = RedisPolicyStore(redis_client=client)
```

**Pros**: Sub-ms latency, pub/sub for notifications  
**Cons**: Requires Redis server, no complex queries

### SQLite (Embedded)

```python
store = SQLitePolicyStore(db_path="policy.db")
```

**Pros**: File-based, no external dependencies  
**Cons**: Single writer, not for high concurrency

### Factory Pattern

```python
from policy_store import PolicyStoreFactory

# Select backend via config
backend = config.get("policy_backend")  # "memory", "postgres", "redis", "sqlite"

if backend == "postgres":
    store = PolicyStoreFactory.create("postgres", connection_pool=pool)
elif backend == "redis":
    store = PolicyStoreFactory.create("redis", redis_client=client)
else:
    store = PolicyStoreFactory.create("memory")
```

## Default Policies

Three preset configurations:

```python
from policy_store import PolicyDefaults

# Standard balanced approach
default = PolicyDefaults.create_default()
# risk_mode="NORMAL", max_risk=0.01, max_pos=10, min_conf=0.65

# Conservative risk management
conservative = PolicyDefaults.create_conservative()
# risk_mode="DEFENSIVE", max_risk=0.005, max_pos=5, min_conf=0.75

# Aggressive trading
aggressive = PolicyDefaults.create_aggressive()
# risk_mode="AGGRESSIVE", max_risk=0.02, max_pos=15, min_conf=0.55

# Use in store initialization
store = InMemoryPolicyStore(initial_policy=conservative)
```

## Best Practices

### 1. Read Often, Write Sparingly

```python
# Good: Cache policy for batch operations
policy = store.get()
for signal in signals:
    if signal.confidence >= policy['global_min_confidence']:
        process(signal)

# Bad: Read on every iteration
for signal in signals:
    if signal.confidence >= store.get()['global_min_confidence']:
        process(signal)
```

### 2. Use Patch for Partial Updates

```python
# Good: Only update what changed
store.patch({"risk_mode": "AGGRESSIVE"})

# Bad: Full update when only one field changed
policy = store.get()
policy['risk_mode'] = "AGGRESSIVE"
store.update(policy)
```

### 3. Let Timestamp Auto-Update

```python
# Good: Timestamp managed automatically
store.patch({"max_positions": 12})

# Bad: Manual timestamp (will be overwritten anyway)
store.patch({
    "max_positions": 12,
    "last_updated": datetime.now().isoformat(),  # Ignored
})
```

### 4. Validate Before Complex Operations

```python
from policy_store import PolicyValidator

# Validate before expensive operations
try:
    PolicyValidator.validate(new_policy)
except PolicyValidationError:
    return  # Don't proceed with invalid policy

# Now safe to update
store.update(new_policy)
```

### 5. Use Typed Access When Possible

```python
# Good: Type safety
policy_obj = store.get_policy_object()
risk_mode: str = policy_obj.risk_mode
max_pos: int = policy_obj.max_positions

# Less safe: Dict access
policy_dict = store.get()
risk_mode = policy_dict['risk_mode']  # Could typo key
```

## Error Handling

```python
from policy_store import PolicyValidationError

try:
    store.update(untrusted_policy)
except PolicyValidationError as e:
    logger.error(f"Invalid policy: {e}")
    # Fallback to safe defaults
    store.reset()
except Exception as e:
    logger.critical(f"Store error: {e}")
    # Alert operations team
```

## Monitoring & Observability

### Track Policy Changes

```python
def log_policy_change(old_policy, new_policy):
    """Log significant policy changes."""
    if old_policy['risk_mode'] != new_policy['risk_mode']:
        logger.info(f"Risk mode changed: {old_policy['risk_mode']} -> {new_policy['risk_mode']}")
    
    if len(new_policy['allowed_strategies']) != len(old_policy['allowed_strategies']):
        logger.info(f"Active strategies: {len(new_policy['allowed_strategies'])}")

# Use in update flow
old = store.get()
store.patch(changes)
new = store.get()
log_policy_change(old, new)
```

### Metrics to Track

- Policy update frequency
- Risk mode distribution (time in each mode)
- Number of allowed strategies over time
- Policy validation failures
- Read/write latency

## Testing

### Unit Tests

```python
def test_risk_guard_respects_policy():
    store = InMemoryPolicyStore()
    store.update({"max_risk_per_trade": 0.01})
    
    risk_guard = RiskGuard(policy_store=store)
    
    assert risk_guard.validate_trade(risk=0.008) == True
    assert risk_guard.validate_trade(risk=0.015) == False
```

### Integration Tests

```python
def test_full_policy_workflow():
    store = InMemoryPolicyStore()
    
    # MSC AI sets initial policy
    store.update({"risk_mode": "NORMAL", "allowed_strategies": ["S1"]})
    
    # OppRank updates rankings
    store.patch({"opp_rankings": {"BTC": 0.9}})
    
    # CLM updates models
    store.patch({"model_versions": {"xgb": "v2"}})
    
    # Verify all updates applied
    policy = store.get()
    assert policy['risk_mode'] == "NORMAL"
    assert policy['opp_rankings']['BTC'] == 0.9
    assert policy['model_versions']['xgb'] == "v2"
```

### Concurrency Tests

```python
def test_concurrent_updates():
    store = InMemoryPolicyStore()
    
    def updater(risk_mode):
        for _ in range(100):
            store.patch({"risk_mode": risk_mode})
    
    threads = [Thread(target=updater, args=(mode,)) 
               for mode in ["AGGRESSIVE", "NORMAL", "DEFENSIVE"]]
    
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Should complete without errors or corruption
    policy = store.get()
    assert policy['risk_mode'] in ["AGGRESSIVE", "NORMAL", "DEFENSIVE"]
```

## Migration Guide

### From Config Files

```python
# Old: Static config file
with open("config.yaml") as f:
    config = yaml.load(f)

# New: Dynamic PolicyStore
store = InMemoryPolicyStore()
store.update({
    "risk_mode": config['risk_mode'],
    "max_positions": config['max_positions'],
    # ... map other fields
})

# Now components read from store instead of config
```

### From Environment Variables

```python
# Old: Environment variables
MAX_RISK = float(os.getenv("MAX_RISK", "0.01"))

# New: PolicyStore
store.update({"max_risk_per_trade": float(os.getenv("MAX_RISK", "0.01"))})
policy = store.get()
max_risk = policy['max_risk_per_trade']
```

## FAQ

**Q: Can multiple processes share the same PolicyStore?**  
A: Yes, if using PostgreSQL or Redis backend. In-memory store is single-process only.

**Q: How often should I read from PolicyStore?**  
A: Read when needed, but cache for batch operations. Policy doesn't change frequently (minutes to hours).

**Q: Can I add custom fields to GlobalPolicy?**  
A: Yes, use the `custom_params` dict for extension without modifying core structure.

**Q: What happens if validation fails mid-operation?**  
A: Update is rejected atomically. Store remains in previous valid state.

**Q: Should I use update() or patch()?**  
A: Use `patch()` for incremental changes (preferred). Use `update()` for full replacements.

**Q: How do I handle policy in distributed systems?**  
A: Use PostgreSQL or Redis backend with proper connection pooling. Consider pub/sub for change notifications.

## See Also

- `policy_store.py` - Core implementation
- `test_policy_store.py` - Comprehensive test suite
- `policy_store_examples.py` - Usage examples
- MSC AI documentation - How risk modes are decided
- Opportunity Ranker documentation - Symbol ranking computation
