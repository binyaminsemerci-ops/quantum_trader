# Exit Brain Watchdog Rules

## Purpose
Monitor Exit Brain health and trigger `system.panic_close` when Exit Brain fails.

## Watchdog Monitors

### Who Monitors Exit Brain?
1. **Risk Kernel** (primary)
2. **Ops Watchdog** (backup)
3. **Capital Allocation** (optional, for awareness)

## Trigger Conditions

### Trigger `system.panic_close` If:

| Condition | Threshold | Rationale |
|-----------|-----------|-----------|
| Heartbeat missing | > 5 seconds | Exit Brain crashed |
| Status = DEGRADED | > 10 seconds consecutive | Exit Brain impaired |
| last_decision_ts stagnant | > 30 seconds with active positions | Decision loop stuck |
| active_positions ≠ 0 AND heartbeat missing | > 3 seconds | Most critical: positions unguarded |

### Logic
```python
class ExitBrainWatchdog:
    def __init__(self):
        self.last_heartbeat_ts = 0
        self.degraded_since = None
        self.last_decision_ts = 0
        self.active_positions = 0
        
    async def check_health(self) -> bool:
        """Returns False if panic_close should trigger"""
        now = time.time()
        
        # Rule 1: Heartbeat missing
        heartbeat_age = now - self.last_heartbeat_ts
        if heartbeat_age > 5:
            logger.error(f"HEARTBEAT MISSING: {heartbeat_age:.1f}s")
            return False
        
        # Rule 2: Degraded too long
        if self.degraded_since:
            degraded_duration = now - self.degraded_since
            if degraded_duration > 10:
                logger.error(f"DEGRADED TOO LONG: {degraded_duration:.1f}s")
                return False
        
        # Rule 3: Decision loop stuck
        if self.active_positions > 0:
            decision_age = now - self.last_decision_ts
            if decision_age > 30:
                logger.error(f"DECISION LOOP STUCK: {decision_age:.1f}s with {self.active_positions} positions")
                return False
        
        # Rule 4: Critical - positions unguarded
        if self.active_positions > 0 and heartbeat_age > 3:
            logger.error(f"POSITIONS UNGUARDED: {self.active_positions} positions, heartbeat {heartbeat_age:.1f}s old")
            return False
        
        return True
```

## NO Grace Periods

> **CRITICAL**: No grace periods during volatility.

If Exit Brain is dead and you have open positions, you have seconds — not minutes — before the market damages you.

### Wrong Approach
```python
# WRONG - DO NOT DO THIS
if heartbeat_missing and volatility > threshold:
    grace_period = 30  # Give it more time because volatile
```

### Correct Approach
```python
# CORRECT - Same threshold regardless of market
if heartbeat_missing > 5:
    trigger_panic_close()  # No exceptions
```

## Acceptable Outcomes

| Scenario | Result | Acceptable? |
|----------|--------|-------------|
| False positive (Exit Brain was fine) | System stops, manual restart | ✅ Yes |
| True positive (Exit Brain crashed) | Positions closed | ✅ Yes |
| False negative (missed failure) | Positions unguarded | ❌ NO |

> **False positives are OK. False negatives are NOT.**

## Implementation

### In Risk Kernel
```python
class RiskKernel:
    async def _monitor_exit_brain(self):
        watchdog = ExitBrainWatchdog()
        
        while self.running:
            # Read latest heartbeat
            heartbeats = await self.redis.xread(
                {"quantum:stream:exit_brain.heartbeat": "$"},
                block=2000
            )
            
            if heartbeats:
                for _, data in heartbeats[0][1]:
                    watchdog.update_from_heartbeat(data)
            else:
                # No heartbeat received
                pass
            
            # Check health
            if not await watchdog.check_health():
                await self._trigger_panic_close(
                    source="risk_kernel",
                    reason="Exit Brain watchdog failure"
                )
                break
```

## Logging

Every watchdog check should log:
```
[WATCHDOG] OK - heartbeat_age=0.8s, status=OK, positions=3, last_decision=2.1s ago
[WATCHDOG] WARNING - heartbeat_age=3.2s, status=DEGRADED
[WATCHDOG] CRITICAL - heartbeat_age=5.1s - TRIGGERING PANIC CLOSE
```

## Testing

See [tests/](tests/) for:
- `test_kill_exit_brain.md` - Kill process, verify panic_close < 5s
- `test_artificial_latency.md` - Inject latency, verify forced exit
- `test_false_positive.md` - Verify system stops (acceptable)

## Related
- [heartbeat.md](heartbeat.md) - Heartbeat specification
- [recovery_flow.md](recovery_flow.md) - Recovery after failure
- [../emergency_exit_worker/](../emergency_exit_worker/) - What happens when panic_close triggers
