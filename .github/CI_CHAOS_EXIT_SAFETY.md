# CI: Chaos Exit Safety

**Status**: Active  
**Created**: 2026-02-14  

---

## Overview

GitHub Actions workflow that validates the Emergency Exit System's fail-closed behavior.

## Workflow Location

```
.github/workflows/chaos_exit_safety.yml
```

## Triggers

- Pull requests to `main`
- Push to `main` branch
- Push to `shadow/*` branches
- Manual dispatch

## What It Tests

| Step | Description | Pass Criteria |
|------|-------------|---------------|
| 1 | Exit Brain mock publishes heartbeats | Stream has entries |
| 2 | Watchdog mock monitors heartbeats | Running |
| 3 | EEW mock listens for panic_close | Running |
| 4 | **CHAOS**: Kill Exit Brain | Process killed |
| 5 | Watchdog detects and triggers panic_close | Event in stream within 6s |
| 6 | EEW receives and executes | Completion event published |
| 7 | System is FLAT | positions=0, trading=halted |

## Policy

### 🔒 MANDATORY

**No code can be merged to `main` if Chaos Exit Safety CI fails.**

This applies to ALL changes, including:
- "Small" fixes
- Documentation updates (if they touch services/)
- Dependency updates

### Rationale

The exit safety system is the last line of defense. If it doesn't work:
- Positions remain open during system failure
- Losses can accumulate unchecked
- Capital is at risk

CI verification ensures this path is **always tested**.

## Files

### Mocks (CI-only)

| File | Purpose |
|------|---------|
| `tests/mocks/exit_brain_mock.py` | Publishes heartbeats |
| `tests/mocks/exit_brain_watchdog_mock.py` | Monitors, triggers panic_close |
| `tests/mocks/emergency_exit_worker_mock.py` | Handles panic_close |

### Assertions

| File | Validates |
|------|-----------|
| `tests/assertions/assert_heartbeat_flowing.py` | Heartbeat was active |
| `tests/assertions/assert_panic_close.py` | panic_close was published |
| `tests/assertions/assert_panic_close_completed.py` | EEW executed |
| `tests/assertions/assert_no_open_positions.py` | System is FLAT |

## Thresholds

| Metric | Value | Notes |
|--------|-------|-------|
| Heartbeat missing | 5s | Triggers panic_close |
| Total test time | 5 min | Workflow timeout |
| Max execution time | 10s | EEW must complete quickly |

## Local Testing

```bash
# Run the full chaos test locally
docker run -d --name redis-test -p 6379:6379 redis:7

# Start all mocks in background
python tests/mocks/exit_brain_mock.py &
python tests/mocks/exit_brain_watchdog_mock.py &
python tests/mocks/emergency_exit_worker_mock.py &

# Wait for startup
sleep 5

# CHAOS: Kill Exit Brain
pkill -f exit_brain_mock.py

# Wait for detection + execution
sleep 8

# Run assertions
python tests/assertions/assert_panic_close.py
python tests/assertions/assert_panic_close_completed.py
python tests/assertions/assert_no_open_positions.py

# Cleanup
pkill -f watchdog_mock.py
pkill -f emergency_exit_worker_mock.py
docker rm -f redis-test
```

## Troubleshooting

### CI Fails: "No panic_close events found"

- Check watchdog logs
- Verify heartbeat was flowing before kill
- Ensure consumer group was created

### CI Fails: "No completion events found"

- Check EEW logs
- Verify EEW received the panic_close event
- Check consumer group acknowledgment

### CI Fails: "positions != 0"

- EEW may not have closed all positions
- Check for failed symbols in completion event

## Related Docs

- [REDIS_STREAMS_SCHEMA.md](../services/REDIS_STREAMS_SCHEMA.md)
- [CHAOS_TEST_RUNBOOK.md](../services/emergency_exit_worker/CHAOS_TEST_RUNBOOK.md)
- [AI_EMERGENCY_EXIT_OPERATIONAL_FINALIZATION_FEB14.md](../AI_EMERGENCY_EXIT_OPERATIONAL_FINALIZATION_FEB14.md)
