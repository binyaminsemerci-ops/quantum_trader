# Exit Brain Recovery Flow

## After Panic Close

When `panic_close.completed` is received, the system enters recovery mode.

## Recovery Sequence

```
┌─────────────────────────────────────────────────────────────┐
│               EXIT BRAIN RECOVERY FLOW                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Exit Brain STOPPED (crashed or killed)                 │
│     │                                                       │
│  2. Watchdog detects heartbeat missing                     │
│     │                                                       │
│  3. panic_close TRIGGERED                                  │
│     │                                                       │
│  4. Emergency Exit Worker closes ALL positions             │
│     │                                                       │
│  5. System is now FLAT (no positions)                      │
│     │                                                       │
│  6. Exit Brain RESTARTED                                   │
│     │                                                       │
│  7. Exit Brain enters SHADOW MODE                          │
│     │                                                       │
│  8. Manual approval required to go live                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Shadow Mode

After restart, Exit Brain operates in shadow mode:
- **Monitors positions** (if any new ones opened)
- **Computes decisions** (what it would do)
- **Logs decisions** (for analysis)
- **Does NOT execute** (no actual closes)

### Shadow Mode Duration
- Minimum: 5 minutes
- Until: Manual approval

### Shadow Mode Exit
```python
# Ops command to exit shadow mode
await redis.publish("quantum:command:exit_brain", json.dumps({
    "command": "enable_live_mode",
    "operator": "ops_team",
    "timestamp": time.time(),
    "confirmation_code": "EXIT_LIVE_CONFIRMED"
}))
```

## State Transitions

```
HEALTHY → (watchdog fail) → STOPPED
STOPPED → (restart) → SHADOW
SHADOW → (manual approval) → HEALTHY
SHADOW → (5 min without approval) → remains SHADOW
```

## Verification Before Live

Before exiting shadow mode, verify:

### 1. Exit Brain Health
```bash
# Check heartbeats are flowing
redis-cli XLEN quantum:stream:exit_brain.heartbeat
# Should show increasing count

# Check status is OK
redis-cli XREVRANGE quantum:stream:exit_brain.heartbeat + - COUNT 1
# status should be "OK"
```

### 2. Decision Quality
```bash
# Review shadow decisions
cat /var/log/quantum/exit_brain_shadow.log | tail -50

# Look for reasonable decisions
# - Not excessive exits
# - Proper confidence levels
# - Matching market conditions
```

### 3. Position Sync
```python
# Verify Exit Brain's view matches exchange
eb_positions = exit_brain.get_monitored_positions()
exchange_positions = binance.futures_position_information()

# They should match
assert len(eb_positions) == len(exchange_positions)
```

### 4. Watchdog Test
```bash
# Verify watchdog is monitoring
journalctl -u quantum-exit-brain-watchdog | tail -10
# Should show recent OK checks
```

## Manual Reset Procedure

```bash
# 1. Check current state
redis-cli HGETALL quantum:state:trading_halted

# 2. Verify system is FLAT
python check_positions.py  # Should show 0

# 3. Verify Exit Brain healthy
systemctl status quantum-exit-brain
redis-cli XREVRANGE quantum:stream:exit_brain.heartbeat + - COUNT 1

# 4. Clear halt state
redis-cli DEL quantum:state:trading_halted

# 5. Enable live mode
redis-cli PUBLISH quantum:command:exit_brain '{"command":"enable_live_mode","operator":"ops","timestamp":1707840000}'

# 6. Re-enable trading
redis-cli SET quantum:state:trading_enabled true
```

## Audit Log

Every recovery creates audit entry:
```json
{
  "event": "exit_brain_recovery",
  "timestamp": 1707840000.123,
  "panic_close_trigger": "risk_kernel",
  "panic_close_reason": "Exit Brain watchdog failure",
  "positions_closed": 5,
  "downtime_seconds": 45,
  "shadow_mode_duration": 312,
  "approved_by": "ops_team",
  "live_enabled_at": 1707840312.456
}
```

## Failure During Recovery

If Exit Brain fails AGAIN during shadow mode:
1. Do NOT trigger panic_close (already flat)
2. Log failure
3. Alert ops
4. Remain in shadow mode
5. Investigate root cause

## Related
- [heartbeat.md](heartbeat.md) - Heartbeat specification
- [watchdog_rules.md](watchdog_rules.md) - Watchdog trigger rules
- [../emergency_exit_worker/](../emergency_exit_worker/) - Panic close execution
