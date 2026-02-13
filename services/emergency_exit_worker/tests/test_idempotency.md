# Test: Idempotency

## Objective
Verify that sending `panic_close` twice does NOT cause errors.

## Why This Matters
In emergency situations, the trigger might be sent multiple times:
- Network issues cause retry
- Multiple services detect same issue
- Operator sends manual trigger while automatic one in progress

The system MUST handle this gracefully.

## Test Steps

### Step 1: Create Test Position
```python
client.futures_create_order(
    symbol="BTCUSDT",
    side="BUY",
    type="MARKET",
    quantity=0.001
)
```

### Step 2: Send panic_close TWICE (rapidly)
```python
import redis
import time

r = redis.Redis()

# First trigger
r.xadd("quantum:stream:system.panic_close", {
    "source": "risk_kernel",
    "reason": "TEST - idempotency check 1",
    "timestamp": str(time.time())
})

# Second trigger - immediately after
time.sleep(0.1)  # Small delay
r.xadd("quantum:stream:system.panic_close", {
    "source": "risk_kernel",
    "reason": "TEST - idempotency check 2",
    "timestamp": str(time.time())
})
```

### Step 3: Wait and Verify
```python
time.sleep(5)

# Check positions
positions = client.futures_position_information()
open_positions = [p for p in positions if float(p['positionAmt']) != 0]
print(f"Open positions: {len(open_positions)}")

# Check completion events
messages = r.xrange("quantum:stream:panic_close.completed", "-", "+")
print(f"Completion events: {len(messages)}")

for msg_id, data in messages[-3:]:  # Last 3
    print(f"  {msg_id}: closed={data.get(b'positions_closed')}, failed={data.get(b'positions_failed')}")
```

## Expected Results

### Scenario A: First trigger closes, second sees 0 positions
- ✅ First panic_close: positions_closed = 1
- ✅ Second panic_close: positions_found = 0, positions_closed = 0
- ✅ No errors in logs

### Scenario B: Race condition (both see positions)
- ✅ First closes positions
- ✅ Second gets "Order would immediately trigger" or similar
- ✅ Logged as failed but worker continues
- ✅ Final state: 0 open positions

## Key Assertions
```python
# After both triggers complete:
assert len(open_positions) == 0, "All positions must be closed"

# No crashes
# Check systemctl status quantum-emergency-exit-worker
# Should be "active (running)"
```

## Error Log Check
```bash
# Should NOT see crashes or unhandled exceptions
journalctl -u quantum-emergency-exit-worker --since "5 minutes ago" | grep -iE "error|exception|crash"

# May see these (acceptable):
# - "Order would immediately trigger"
# - "Position already closed"
# - "No position to reduce"
```

## Race Condition Handling
The worker handles this by:
1. Each trigger fetches FRESH positions
2. If position already closed, close attempt fails gracefully
3. Worker continues to next position
4. Both triggers complete without crash

## Cleanup
```python
r.delete("quantum:state:trading_halted")
```
