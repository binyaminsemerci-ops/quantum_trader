# Chaos Exit Safety Tests

This directory contains behavioral tests for the Emergency Exit System.

## What This Tests

The chaos test validates **system behavior**, not unit logic:

1. **Exit Brain dies** (simulated by killing mock)
2. **Watchdog detects** within 5 seconds
3. **panic_close published** to Redis stream
4. **EEW executes** and closes all positions
5. **System is FLAT** (0 positions, trading halted)

## Directory Structure

```
mocks/
├── exit_brain_mock.py              # Publishes heartbeats
├── exit_brain_watchdog_mock.py     # Monitors heartbeat, triggers panic_close
└── emergency_exit_worker_mock.py   # Handles panic_close, closes positions

assertions/
├── assert_heartbeat_flowing.py      # Verify heartbeat was active
├── assert_panic_close.py            # Verify panic_close published
├── assert_panic_close_completed.py  # Verify EEW executed
└── assert_no_open_positions.py      # Verify system is FLAT
```

## Running Locally

```bash
# Start Redis
docker run -d --name redis-test -p 6379:6379 redis:7

# Terminal 1: Start Exit Brain mock
python tests/mocks/exit_brain_mock.py

# Terminal 2: Start Watchdog mock  
python tests/mocks/exit_brain_watchdog_mock.py

# Terminal 3: Start EEW mock
python tests/mocks/emergency_exit_worker_mock.py

# Terminal 4: Kill Exit Brain and run assertions
kill -9 $(pgrep -f exit_brain_mock.py)
sleep 8
python tests/assertions/assert_panic_close.py
python tests/assertions/assert_panic_close_completed.py
python tests/assertions/assert_no_open_positions.py
```

## CI Integration

See `.github/workflows/chaos_exit_safety.yml`

**POLICY**: No merge to `main` if this workflow fails.
