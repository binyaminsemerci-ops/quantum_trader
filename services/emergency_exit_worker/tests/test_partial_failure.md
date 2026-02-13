# Test: Partial Failure Handling

## Objective
Verify that if ONE position fails to close, the others are still closed.

## Setup
1. Create 3 test positions
2. Mock one symbol to fail (or use invalid precision)
3. Trigger panic_close
4. Verify other positions closed

## Test Scenario

### Scenario A: API Error on One Symbol

```python
# Setup: Create positions including one with a potential issue
positions = [
    ("BTCUSDT", "BUY", 0.001),   # Should succeed
    ("ETHUSDT", "BUY", 0.01),    # Should succeed
    ("INVALIDUSDT", "BUY", 1.0)   # Will fail (doesn't exist)
]
```

Note: On real testnet, use a symbol that might hit rate limits or has unusual precision requirements.

### Scenario B: Quantity Precision Error

```python
# Create position with odd quantity that might cause precision issues
client.futures_create_order(
    symbol="BTCUSDT",
    side="BUY",
    type="MARKET",
    quantity=0.00123456789  # Unusual precision
)
```

## Test Steps

### Step 1: Create Mixed Positions
```python
# Create some valid positions
valid_positions = [
    ("BTCUSDT", "BUY", 0.001),
    ("ETHUSDT", "BUY", 0.01)
]

for symbol, side, qty in valid_positions:
    client.futures_create_order(symbol=symbol, side=side, type="MARKET", quantity=qty)
```

### Step 2: Trigger Panic Close
```python
import redis
import time

r = redis.Redis()
r.xadd("quantum:stream:system.panic_close", {
    "source": "risk_kernel",
    "reason": "TEST - partial failure",
    "timestamp": str(time.time())
})
```

### Step 3: Verify Behavior
```python
time.sleep(5)

# Check completion event
messages = r.xread({"quantum:stream:panic_close.completed": "0"}, count=10)
latest = messages[-1] if messages else None

if latest:
    _, data = latest[1][0]
    closed = int(data[b'positions_closed'])
    failed = int(data[b'positions_failed'])
    failed_symbols = json.loads(data[b'failed_symbols'])
    
    print(f"Closed: {closed}")
    print(f"Failed: {failed}")
    print(f"Failed symbols: {failed_symbols}")
```

## Expected Results
- ✅ Valid positions are closed
- ✅ Worker continues after failure
- ✅ Failed symbols logged
- ✅ Completion event includes failure count
- ✅ System still halted (even with partial failure)

## Key Assertion
```python
# The worker must NOT stop when one position fails
assert closed > 0, "At least some positions should close"
```

## Verification Query
```bash
# Check EEW logs for error handling
journalctl -u quantum-emergency-exit-worker --since "5 minutes ago" | grep -E "FAILED|closed"
```

## Cleanup
```python
# Manual close any remaining positions
# Reset halt state
r.delete("quantum:state:trading_halted")
```
