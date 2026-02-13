# Exit Brain Heartbeat Specification

## Purpose
Detect Exit Brain failure FASTER than the market can damage you.

## Heartbeat Event

### Stream
```
quantum:stream:exit_brain.heartbeat
```

### Message Format
```json
{
  "timestamp": 1707840000.123,
  "status": "OK",
  "active_positions_count": 5,
  "last_decision_ts": 1707839999.456,
  "loop_cycle_ms": 245,
  "memory_mb": 128,
  "pending_exits": 2
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | float | Unix timestamp of this heartbeat |
| `status` | string | `OK` or `DEGRADED` |
| `active_positions_count` | int | Current open positions being monitored |
| `last_decision_ts` | float | Timestamp of last exit decision |
| `loop_cycle_ms` | int | How long last decision cycle took |
| `memory_mb` | int | Memory usage (optional) |
| `pending_exits` | int | Exits queued but not confirmed |

## Status Values

### OK
- Exit Brain is healthy
- Decision loop running normally
- All systems nominal

### DEGRADED
- Exit Brain is running but impaired
- May trigger watchdog if persists
- Possible causes:
  - High latency
  - Memory pressure
  - External API issues
  - Processing backlog

## Frequency

| Condition | Interval |
|-----------|----------|
| Normal operation | 1-2 seconds |
| Under load | 1 second (more frequent) |
| Degraded | 500ms (even more frequent) |

## Implementation

### In Exit Brain
```python
class ExitBrain:
    def __init__(self):
        self.heartbeat_interval = 1.0  # seconds
        self.last_decision_ts = 0.0
        
    async def _heartbeat_loop(self):
        while self.running:
            await self._publish_heartbeat()
            await asyncio.sleep(self.heartbeat_interval)
    
    async def _publish_heartbeat(self):
        status = "OK"
        
        # Check for degraded conditions
        if self._is_degraded():
            status = "DEGRADED"
        
        await self.redis.xadd(
            "quantum:stream:exit_brain.heartbeat",
            {
                "timestamp": str(time.time()),
                "status": status,
                "active_positions_count": str(len(self.active_positions)),
                "last_decision_ts": str(self.last_decision_ts),
                "loop_cycle_ms": str(self.last_cycle_ms),
                "pending_exits": str(len(self.pending_exits))
            },
            maxlen=1000  # Keep last 1000 heartbeats
        )
    
    def _is_degraded(self) -> bool:
        # Check various health indicators
        if self.last_cycle_ms > 500:  # Decision loop too slow
            return True
        if time.time() - self.last_decision_ts > 10:  # No decisions in 10s
            if len(self.active_positions) > 0:  # But has positions
                return True
        return False
```

## Consumer Example
```python
async def read_heartbeats():
    while True:
        messages = await redis.xread(
            {"quantum:stream:exit_brain.heartbeat": "$"},
            block=2000  # 2 second timeout
        )
        
        if not messages:
            # No heartbeat received in 2 seconds!
            logger.warning("Exit Brain heartbeat MISSING")
        else:
            for _, data in messages[0][1]:
                status = data[b'status'].decode()
                positions = int(data[b'active_positions_count'])
                logger.debug(f"Heartbeat: {status}, positions={positions}")
```

## Retention
- Stream maxlen: 1000 messages (~15 minutes at 1/sec)
- No archival needed (real-time only)

## Related
- [watchdog_rules.md](watchdog_rules.md) - How watchdog responds to heartbeats
- [recovery_flow.md](recovery_flow.md) - What happens after failure
