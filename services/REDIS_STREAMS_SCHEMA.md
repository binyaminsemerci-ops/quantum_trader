# Redis Streams Schema - Emergency Exit System

> **Konkret, operativ spesifikasjon for fail-closed sikkerhet**

---

## Stream Overview

| Stream | Purpose | Producers | Consumers |
|--------|---------|-----------|-----------|
| `system:panic_close` | Global nødkanal | risk_kernel, exit_brain, ops | EEW, Watchdog |
| `system:panic_close:completed` | EEW completion | EEW | Audit, Ops |
| `exit_brain:heartbeat` | Health monitoring | Exit Brain | Watchdog |

---

## 1. `system:panic_close` — Global Nødkanal

### Purpose
Trigger unconditional close of ALL positions.

### Producers (EXHAUSTIVE LIST)
- `risk_kernel` — Automatic triggers
- `exit_brain` — Fatal health failure only
- `ops` — Manual emergency key

**NO OTHER PRODUCER ALLOWED**

### Schema
```
Field           Type        Description
─────────────────────────────────────────────────────────
event_id        uuid        Unique event identifier
reason          string      e.g. "EXIT_BRAIN_HEARTBEAT_LOST"
severity        enum        CRITICAL (always)
issued_by       enum        risk_kernel | exit_brain | ops
ts              int64       Epoch milliseconds
```

### Example Message
```redis
XADD system:panic_close *
  event_id "550e8400-e29b-41d4-a716-446655440000"
  reason "EXIT_BRAIN_HEARTBEAT_LOST"
  severity "CRITICAL"
  issued_by "watchdog"
  ts "1707840000123"
```

### Semantics
- **Delivery**: At-least-once
- **Idempotency**: EEW MUST handle duplicates gracefully
- **TTL**: NONE (audit requirement - keep forever)
- **Consumer Groups**: `emergency_exit_worker`, `audit_logger`

### Consumer Group Setup
```redis
XGROUP CREATE system:panic_close emergency_exit_worker $ MKSTREAM
XGROUP CREATE system:panic_close audit_logger $ MKSTREAM
```

---

## 2. `system:panic_close:completed` — EEW Output

### Purpose
Report completion of panic close execution.

### Producer
- `emergency_exit_worker` ONLY

### Schema
```
Field               Type        Description
─────────────────────────────────────────────────────────
event_id            uuid        Original panic_close event_id
positions_total     int         Positions found open
positions_closed    int         Successfully closed
positions_failed    int         Failed to close
failed_symbols      json        List of failed symbols
ts_started          int64       Epoch ms when started
ts_completed        int64       Epoch ms when completed
execution_time_ms   int         Total execution time
```

### Example Message
```redis
XADD system:panic_close:completed *
  event_id "550e8400-e29b-41d4-a716-446655440000"
  positions_total "5"
  positions_closed "5"
  positions_failed "0"
  failed_symbols "[]"
  ts_started "1707840000123"
  ts_completed "1707840002456"
  execution_time_ms "2333"
```

### Semantics
- **TTL**: NONE (audit requirement)
- **Max Length**: Unlimited (or set high limit like 100000)

---

## 3. `exit_brain:heartbeat` — Health Monitoring

### Purpose
Real-time Exit Brain health signal for watchdog.

### Producer
- `exit_brain` ONLY

### Schema
```
Field               Type        Description
─────────────────────────────────────────────────────────
service_id          string      Instance identifier
status              enum        OK | DEGRADED
active_positions    int         Currently monitored positions
last_decision_ts    int64       Epoch ms of last exit decision
latency_ms          int         Last decision loop duration
ts                  int64       Epoch ms of this heartbeat
```

### Example Message
```redis
XADD exit_brain:heartbeat MAXLEN ~ 1000 *
  service_id "exit_brain_main"
  status "OK"
  active_positions "3"
  last_decision_ts "1707839999456"
  latency_ms "245"
  ts "1707840000123"
```

### Frequency
- **Hard requirement**: 1-2 seconds
- **Never slower than**: 2 seconds

### Semantics
- **Max Length**: ~1000 messages (trim)
- **TTL**: N/A (stream auto-trims)

---

## 4. Watchdog Trigger Rules (Hard)

The watchdog MUST trigger `system:panic_close` when ANY of these conditions are met:

### Trigger Conditions

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Heartbeat mangler | > 5 seconds | TRIGGER |
| status = DEGRADED | > 5 seconds consecutive | TRIGGER |
| active_positions > 0 AND last_decision_ts stagnerer | > 30 seconds | TRIGGER |
| active_positions > 0 AND heartbeat mangler | > 3 seconds | TRIGGER (most critical) |

### Trigger Logic (Pseudocode)
```python
def should_trigger_panic_close(state: WatchdogState) -> str | None:
    now = current_time_ms()
    
    # Rule 1: Heartbeat missing
    if now - state.last_heartbeat_ts > 5000:
        return "EXIT_BRAIN_HEARTBEAT_LOST"
    
    # Rule 2: Degraded too long
    if state.degraded_since and (now - state.degraded_since > 5000):
        return "EXIT_BRAIN_DEGRADED_TOO_LONG"
    
    # Rule 3: Decision stagnant with positions
    if state.active_positions > 0:
        if now - state.last_decision_ts > 30000:
            return "EXIT_BRAIN_DECISION_STAGNANT"
    
    # Rule 4: CRITICAL - Positions unguarded
    if state.active_positions > 0:
        if now - state.last_heartbeat_ts > 3000:
            return "POSITIONS_UNGUARDED"
    
    return None  # Healthy
```

---

## 5. State Keys

### Trading State
```redis
# Trading halt state (set by EEW after panic_close)
HSET system:state:trading
  halted "true"
  reason "EXIT_BRAIN_HEARTBEAT_LOST"
  halted_at "1707840002456"
  halted_by "emergency_exit_worker"
  requires_manual_ack "true"
```

### Manual Reset
```redis
# To clear halt state (requires ops authorization)
DEL system:state:trading
# OR
HSET system:state:trading
  halted "false"
  cleared_by "ops_manual"
  cleared_at "1707840312456"
```

---

## 6. Consumer Examples

### EEW Consumer
```python
async def consume_panic_close():
    while True:
        messages = await redis.xreadgroup(
            "emergency_exit_worker",
            "eew_instance_1",
            {"system:panic_close": ">"},
            count=1,
            block=1000
        )
        
        for stream, entries in messages:
            for msg_id, fields in entries:
                event_id = fields[b'event_id'].decode()
                
                # Check idempotency
                if await already_processed(event_id):
                    await redis.xack("system:panic_close", "emergency_exit_worker", msg_id)
                    continue
                
                # Execute panic close
                result = await execute_panic_close(fields)
                
                # Publish completion
                await publish_completion(event_id, result)
                
                # Acknowledge
                await redis.xack("system:panic_close", "emergency_exit_worker", msg_id)
```

### Watchdog Consumer
```python
async def monitor_exit_brain():
    last_id = "$"
    
    while True:
        messages = await redis.xread(
            {"exit_brain:heartbeat": last_id},
            block=2000  # 2 second timeout
        )
        
        if not messages:
            # No heartbeat in 2 seconds - potential issue
            check_and_alert()
            continue
        
        for stream, entries in messages:
            for msg_id, fields in entries:
                last_id = msg_id
                update_watchdog_state(fields)
        
        # Check trigger conditions
        reason = should_trigger_panic_close(state)
        if reason:
            await publish_panic_close(reason)
```

---

## 7. Audit Requirements

All streams are audit-critical:

1. **No deletion** of `system:panic_close` messages
2. **No deletion** of `system:panic_close:completed` messages
3. **Heartbeat stream** may trim to ~1000 for performance
4. **Event correlation** via `event_id` field

### Audit Query Example
```redis
# Find all panic_close events in last 24 hours
XRANGE system:panic_close - + COUNT 100

# Find completion for specific event
XREAD STREAMS system:panic_close:completed 0
# Then filter by event_id
```

---

*Schema version: 1.0*
*Last updated: 14. Februar 2026*
