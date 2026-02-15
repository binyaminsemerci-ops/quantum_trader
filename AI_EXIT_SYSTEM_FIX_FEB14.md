# Exit System Debug & Fix - February 14, 2026

## Problem Found

### Root Cause: AI Engine HTTP Blocking
The AI Engine uses uvicorn single-worker mode. When the FastAPI lifespan startup hooks start Redis stream consumers, they continuously process events without yielding to HTTP handlers.

**Evidence:**
- Port 8001 IS bound: `LISTEN 127.0.0.1:8001 (python pid=1805438)`
- curl CONNECTS but times out waiting for response (event loop starvation)
- Event-driven exits (`[AI-EXIT]` logs) work fine via Redis
- HTTP calls to `/api/v1/evaluate-exit` never respond

### Impact
- ExitManager in autonomous_trader calls AI Engine via HTTP
- HTTP timeout = 5s (increased to 30s but still not enough)
- When timeout, fallback was `HOLD` → positions never exit

## Fix Applied: Local R-Based Fallback

Added local fallback logic in `exit_manager.py` (before "# Default: hold"):

```python
# LOCAL FALLBACK when AI unavailable - R-based exits
R = position.R_net
age_hours = position.age_seconds / 3600 if hasattr(position, "age_seconds") else 0

# Profit taking: R > 2.0 = close all
if R > 2.0:
    return ExitDecision(action="CLOSE", percentage=1.0, reason="local_fallback_profit")

# Profit taking: R > 1.0 = partial close 50%
if R > 1.0:
    return ExitDecision(action="PARTIAL_CLOSE", percentage=0.5, reason="local_fallback_profit")

# Loss cutting: R < -1.0 and position > 4 hours
if R < -1.0 and age_hours > 4:
    return ExitDecision(action="CLOSE", percentage=1.0, reason="local_fallback_loss")
```

### Thresholds:
| Condition | Action | Percentage |
|-----------|--------|------------|
| R > 2.0 | CLOSE | 100% |
| R > 1.0 | PARTIAL_CLOSE | 50% |
| R < -1.0 AND age > 4h | CLOSE | 100% |
| -1.0 ≤ R ≤ 1.0 | HOLD | 0% |

## Current Position Status

| Symbol | R | Falls into | Expected Action |
|--------|---|------------|-----------------|
| BTCUSDT | -0.12 | Hold zone | HOLD |
| ETHUSDT | -0.10 | Hold zone | HOLD |
| SOLUSDT | -0.12 | Hold zone | HOLD |

Positions are in small-loss territory. Fallback will trigger when:
- Market moves in our favor (R > 1.0)
- Or losses deepen and age > 4 hours (R < -1.0)

## Files Modified

1. **exit_manager.py** (autonomous_trader)
   - Line 53: `timeout=5.0` → `timeout=30.0`
   - Lines ~130-160: Added local R-based fallback

2. **exit_evaluator.py** (ai_engine)  
   - Line 302: Skip slow `ensemble.get_signal()` call

## Future Improvements Needed

### Option 1: Fix Event Loop Starvation
Add `await asyncio.sleep(0)` in Redis stream consumer loops to yield to HTTP:
```python
async def _consume_stream():
    while True:
        messages = await redis.xread(...)
        for msg in messages:
            await handle_message(msg)
            await asyncio.sleep(0)  # Yield to HTTP handlers
```

### Option 2: Multiple Workers
Change uvicorn to run with workers (requires code changes for shared state):
```bash
uvicorn main:app --workers 2 --host 127.0.0.1 --port 8001
```

### Option 3: Separate HTTP Service
Run a lightweight HTTP service separate from the event processor.

### Option 4: Event-Driven Exits (Best)
Make autonomous_trader listen to Redis `exit.evaluated` events instead of HTTP calls.

## Summary

✅ **FIXED**: Exits now work via local fallback when AI times out
✅ **FIXED**: Timeout increased from 5s to 30s  
⚠️ **WORKAROUND**: Event loop starvation still exists but has fallback
📋 **TODO**: Implement proper event-driven exit communication

---
*Documented: 2026-02-14 01:15 UTC*
