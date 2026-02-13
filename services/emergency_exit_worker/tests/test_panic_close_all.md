# Test: Panic Close All Positions

## Objective
Verify that when `system.panic_close` is triggered, ALL open positions are closed.

## Setup
1. Create 3 test positions (different symbols)
2. Start Emergency Exit Worker
3. Trigger panic_close event

## Test Steps

### Step 1: Create Test Positions
```python
# Create test positions on testnet
positions = [
    ("BTCUSDT", "BUY", 0.001),
    ("ETHUSDT", "BUY", 0.01),
    ("SOLUSDT", "SELL", 1.0)
]

for symbol, side, qty in positions:
    client.futures_create_order(
        symbol=symbol,
        side=side,
        type="MARKET",
        quantity=qty
    )
```

### Step 2: Verify Positions Open
```python
positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]
assert len(open_positions) == 3, f"Expected 3 positions, got {len(open_positions)}"
```

### Step 3: Trigger Panic Close
```python
import redis
import time
import json

r = redis.Redis()
r.xadd("quantum:stream:system.panic_close", {
    "source": "ops",
    "reason": "TEST - panic_close_all",
    "timestamp": str(time.time())
})
```

### Step 4: Verify All Closed
```python
import time
time.sleep(5)  # Wait for execution

positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]
assert len(open_positions) == 0, f"Expected 0 positions, got {len(open_positions)}"
```

### Step 5: Verify Completion Event
```python
# Check panic_close.completed stream
messages = r.xread({"quantum:stream:panic_close.completed": "0"}, count=1)
assert len(messages) > 0, "No completion event found"

_, data = messages[0][1][0]
assert data[b'positions_closed'] == b'3'
assert data[b'positions_failed'] == b'0'
```

## Expected Results
- ✅ All 3 positions closed
- ✅ panic_close.completed published
- ✅ positions_failed = 0
- ✅ Execution time < 5 seconds

## Cleanup
```python
# Reset trading halt state for next test
r.delete("quantum:state:trading_halted")
```
