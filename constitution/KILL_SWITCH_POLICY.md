# 🔴 KILL-SWITCH POLICY - Emergency Procedures

**Document**: Kill-Switch & Emergency Protocol  
**Authority**: CONSTITUTIONAL - HIGHEST PRIORITY  
**Version**: 1.0  
**Grunnlov Reference**: §10, §11, §12, §13  

---

## The Kill-Switch Promise

> **"When activated, ALL trading STOPS. No exceptions. No delays. No overrides."**

The Kill-Switch is the ultimate safeguard. It sits at Level 0 in the Decision Hierarchy, above all other systems including Risk Kernel and Policy Engine.

---

## 1. Kill-Switch Definition

### What Kill-Switch Does

1. **CLOSE** all open positions at market price
2. **CANCEL** all pending orders
3. **BLOCK** any new order submission
4. **DISABLE** all automated trading
5. **ALERT** all channels (Telegram, Email, Dashboard)
6. **LOG** complete system state
7. **PERSIST** until manual human reset

### Irreversibility

Kill-Switch CANNOT be:
- Overridden by AI
- Auto-reset by time
- Bypassed by backdoor
- Partially disabled

---

## 2. Kill-Switch Triggers

### Automatic Triggers (System-Initiated)

| Trigger | Threshold | Response Time |
|---------|-----------|---------------|
| **Daily Loss Limit** | -5% of equity | < 100ms |
| **Catastrophic Drawdown** | -20% of equity | < 100ms |
| **Data Integrity Failure** | < 90% data quality | < 1 second |
| **Exchange Connection Lost** | > 30 seconds | < 100ms |
| **Position Sync Mismatch** | Any discrepancy | < 1 second |
| **10 Consecutive Losses** | Count = 10 | Immediate |
| **Flash Crash Detection** | > 10% in 1 hour | < 1 second |
| **Black Swan Event** | > 20% in 24 hours | < 1 second |
| **Liquidation Risk** | Margin < 50% | < 100ms |

### Manual Triggers (Human-Initiated)

| Method | Authority | Verification |
|--------|-----------|--------------|
| Dashboard Button | Founder only | Password required |
| Telegram Command | `/killswitch` | 2FA confirmation |
| API Endpoint | Authenticated | API key + signature |
| Phone Call | Emergency | Voice verification |

---

## 3. Kill-Switch Execution Sequence

### Step-by-Step Execution

```
T+0ms:    Trigger detected
T+10ms:   Kill-switch flag set to TRUE
T+20ms:   All order submission blocked
T+30ms:   Market close orders queued
T+50ms:   Orders sent to exchange
T+100ms:  Alert sent to all channels
T+200ms:  System state logged
T+500ms:  Confirmation received (or retry)
T+1sec:   All positions confirmed closed
T+2sec:   Final state persisted
T+5sec:   Dashboard updated
```

### Execution Code

```python
async def execute_kill_switch(reason: str, trigger: str):
    """
    Level 0 Emergency Protocol.
    No code may override this function.
    """
    timestamp = datetime.utcnow()
    
    # 1. Set flag FIRST (atomic operation)
    await set_kill_switch_active(True)
    
    # 2. Block ALL new orders
    await block_order_submission()
    
    # 3. Cancel pending orders
    cancelled = await cancel_all_orders()
    
    # 4. Close all positions at market
    closed = await close_all_positions_market()
    
    # 5. Send alerts
    await send_emergency_alert(
        message=f"🔴 KILL-SWITCH ACTIVATED\n"
                f"Reason: {reason}\n"
                f"Trigger: {trigger}\n"
                f"Positions closed: {len(closed)}\n"
                f"Time: {timestamp}",
        channels=["telegram", "email", "dashboard", "sms"]
    )
    
    # 6. Log complete state
    await log_system_state(
        event="KILL_SWITCH",
        reason=reason,
        trigger=trigger,
        cancelled_orders=cancelled,
        closed_positions=closed,
        timestamp=timestamp
    )
    
    # 7. Persist kill-switch state
    await persist_kill_switch_state({
        "active": True,
        "activated_at": timestamp,
        "reason": reason,
        "trigger": trigger,
        "requires_manual_reset": True
    })
    
    return {
        "success": True,
        "positions_closed": len(closed),
        "orders_cancelled": len(cancelled),
        "timestamp": timestamp
    }
```

---

## 4. Kill-Switch Reset Protocol

### Prerequisites for Reset

1. **Minimum 24-hour cooling period** (automatic)
2. **Root cause analysis completed** (documented)
3. **Corrective action implemented** (if applicable)
4. **System health check passed** (automated)
5. **Human verification** (manual)

### Reset Procedure

```
1. 24h MUST have passed since activation
2. Complete incident report filed
3. Run full system health check
4. Run shadow-mode test (min 10 trades)
5. Verify all risk limits unchanged
6. Two-factor authentication
7. Manual reset button pressed
8. System enters Proof Mode (Level 1)
9. Gradual restoration over 7 days
```

### Reset Authorization

| Scenario | Who Can Reset |
|----------|---------------|
| Daily Loss Trigger | Any founder |
| Drawdown > 15% | Founder + 24h wait |
| Drawdown > 20% | Founder + 7-day wait |
| Data Integrity | After data restored |
| Black Swan | Founder + full review |

---

## 5. Kill-Switch Testing

### Regular Testing Schedule

| Test | Frequency | Method |
|------|-----------|--------|
| Smoke test | Weekly | Shadow mode trigger |
| Full test | Monthly | Testnet live trigger |
| Failover test | Quarterly | Simulated exchange failure |

### Test Protocol

```python
def test_kill_switch():
    """
    Monthly kill-switch test - TESTNET ONLY
    """
    # 1. Create test positions (testnet)
    positions = create_test_positions(3)
    
    # 2. Trigger kill-switch
    result = execute_kill_switch(
        reason="MONTHLY_TEST",
        trigger="manual_test"
    )
    
    # 3. Verify all closed
    assert len(result["positions_closed"]) == 3
    
    # 4. Verify flag set
    assert is_kill_switch_active() == True
    
    # 5. Verify new orders blocked
    try:
        create_order(...)
        assert False, "Should have been blocked"
    except OrderBlockedError:
        pass  # Expected
    
    # 6. Reset for test
    reset_kill_switch(test_mode=True)
    
    return "KILL_SWITCH_TEST_PASSED"
```

---

## 6. Black Swan Protocol

### Definition

A Black Swan event is:
- BTC/ETH moves > 20% in 24 hours
- Or > 10% in 1 hour
- Or exchange declares force majeure
- Or major counterparty failure

### Black Swan Response

```
IMMEDIATE (T+0 to T+1 minute):
├── Kill-switch activated
├── All positions closed at market
├── Alert all channels
└── Log final state

SHORT-TERM (T+1 minute to T+24 hours):
├── System enters lockdown
├── Human assessment required
├── Monitor external news
└── No automated decisions

MEDIUM-TERM (T+24h to T+7 days):
├── Complete incident analysis
├── Risk model review
├── Strategy adjustment if needed
└── Gradual restart planning

RECOVERY (T+7 days+):
├── Shadow-mode testing
├── Proof Mode restart (Level 1)
├── Gradual scaling back
└── Weekly reviews for 30 days
```

---

## 7. Flash Crash Protocol

### Detection

```python
def detect_flash_crash(price_data):
    """
    Flash crash: >10% move in <1 hour that partially reverses
    """
    one_hour_change = abs(
        (price_data.current - price_data.one_hour_ago) / 
        price_data.one_hour_ago
    )
    
    if one_hour_change > 0.10:
        return FlashCrashEvent(
            detected=True,
            magnitude=one_hour_change,
            direction="down" if price_data.current < price_data.one_hour_ago else "up"
        )
    
    return FlashCrashEvent(detected=False)
```

### Response

1. Immediate kill-switch if positions open
2. 4-hour mandatory pause
3. Assess if event is ongoing
4. No re-entry until volatility normalizes (ATR < 2x average)

---

## 8. Disaster Recovery

### System Failure Scenarios

| Scenario | Recovery |
|----------|----------|
| Server crash | Hot standby takes over, kill-switch if doubt |
| Exchange API down | Close via backup API or manual |
| Database corruption | Restore from replica, kill-switch |
| Network partition | Kill-switch all segments |

### Data Recovery

```
Priority during disaster:
1. Close positions (safety)
2. Save state to persistent storage
3. Alert humans
4. Log everything
5. Prepare for manual intervention
```

---

## 9. Kill-Switch Logging

### Required Log Fields

```json
{
    "kill_switch_id": "uuid",
    "activated_at": "ISO8601",
    "trigger_type": "automatic|manual",
    "trigger_reason": "daily_loss|drawdown|data_integrity|...",
    "trigger_threshold": 0.05,
    "actual_value": 0.051,
    "positions_at_activation": [...],
    "orders_at_activation": [...],
    "positions_closed": [...],
    "orders_cancelled": [...],
    "close_slippage": {...},
    "total_realized_loss": -500.00,
    "alerts_sent": ["telegram", "email", "sms"],
    "reset_at": null,
    "reset_by": null,
    "incident_report_id": null
}
```

---

## 10. Kill-Switch Guarantees

### System Guarantees

1. **Kill-Switch is always reachable** - Separate service, multiple paths
2. **Kill-Switch has priority** - Overrides all other systems
3. **Kill-Switch is tested** - Regular verification
4. **Kill-Switch is logged** - Complete audit trail
5. **Kill-Switch requires human to reset** - No auto-recovery

### What We Accept

By using this system, we accept:
- Positions may be closed at unfavorable prices during kill-switch
- Some slippage is expected during emergency closes
- False positives may occur (better safe than sorry)
- Manual reset takes time (by design)

---

## 11. Emergency Contact Protocol

### Escalation Chain

```
Level 1: Dashboard alert + Telegram
Level 2: SMS to primary founder
Level 3: Phone call to secondary founder
Level 4: All-hands notification

Response requirements:
- Level 1-2: Acknowledge within 15 minutes
- Level 3-4: Immediate action required
```

---

**END OF KILL-SWITCH POLICY DOCUMENT**
