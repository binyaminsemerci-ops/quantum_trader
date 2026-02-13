# Restart Protocol Module

**Grunnlov**: §7 (Fail-Closed)  
**Purpose**: Safe system restart procedures  

## Restart Types

### Graceful Restart

Used for: Planned maintenance, updates, deployments

```
1. Pause new entries (no new trades)
2. Wait for pending orders to resolve
3. Optionally close positions OR preserve
4. Save complete state
5. Shutdown services in order
6. Start services in order
7. Run pre-flight check
8. Resume trading
```

### Emergency Restart

Used for: System crash, data corruption, unknown state

```
1. Kill-switch FIRST (close all positions)
2. Force stop all services
3. Restore from last known good state
4. Clear any pending operations
5. Start services in safe mode
6. Run full pre-flight
7. Manual review before trading
8. Start in Proof Mode (Level 1)
```

### Hot Restart

Used for: Single service failure, failover

```
1. Detect failed service
2. Switch to standby instance
3. Sync state from primary
4. Continue without interruption
5. Log failover event
6. Alert for review
```

## Service Start Order

```
1. audit_ledger (logging first)
2. data_integrity (data quality)
3. human_override_lock (kill-switch ready)
4. risk_kernel (risk monitoring)
5. policy_engine (rules active)
6. capital_allocation (sizing ready)
7. market_regime (context)
8. exit_brain (can close positions)
9. entry_gate (entries blocked until last)
10. execution (can execute orders)
11. signal_advisory (AI signals last)
```

## Service Stop Order

Reverse of start order.
