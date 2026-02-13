# RK-FT-05: Kill-Switch Persistence Test

**Test ID**: RK-FT-05  
**Category**: Safety-Critical  
**Priority**: P0 (Must Pass)  

---

## Test Objective

Verify that once kill-switch is triggered, it PERSISTS until manual reset.

---

## Test Cases

### TC-05.1: Kill-Switch Persists After Trigger

**Given**: Kill-switch triggered by daily loss  
**When**: Time passes (any duration)  
**Then**: Kill-switch remains active

```python
def test_kill_switch_persistence():
    # Trigger
    trigger_kill_switch("daily_loss", "Test trigger")
    
    # Wait
    time.sleep(10)
    
    # Still active
    assert risk_kernel.is_kill_switch_active() == True
```

### TC-05.2: System Restart Does Not Clear Kill-Switch

**Given**: Kill-switch is active  
**When**: System restarts  
**Then**: Kill-switch is STILL active after restart

```python
def test_kill_switch_survives_restart():
    # Trigger
    trigger_kill_switch("drawdown", "Test")
    
    # Simulate restart
    risk_kernel.shutdown()
    risk_kernel.startup()
    
    # Still active
    assert risk_kernel.is_kill_switch_active() == True
```

### TC-05.3: All Trades Rejected When Kill-Switch Active

**Given**: Kill-switch is active  
**When**: Valid trade is submitted  
**Then**: Result = `risk.rejected`

```python
def test_all_trades_rejected_during_kill_switch():
    # Activate
    trigger_kill_switch("test", "Test")
    
    # Try valid trade
    result = risk_kernel.evaluate(perfectly_valid_request())
    
    assert result.type == "risk.rejected"
    assert "kill-switch" in result.reason.lower()
```

### TC-05.4: No Automatic Reset

**Given**: Kill-switch triggered  
**When**: Conditions that triggered it no longer exist  
**Then**: Kill-switch is STILL active

```python
def test_no_automatic_reset():
    # Trigger due to daily loss
    capital = over_daily_limit_capital()
    risk_kernel.check_daily_loss(capital)
    
    # "Fix" the condition
    capital.daily_pnl = 0  # Back to positive
    
    # Kill-switch still active
    assert risk_kernel.is_kill_switch_active() == True
```

### TC-05.5: Manual Reset Requires Authentication

**Given**: Kill-switch is active  
**When**: Reset attempted without authentication  
**Then**: Reset FAILS, kill-switch remains active

```python
def test_reset_requires_auth():
    trigger_kill_switch("test", "Test")
    
    # Try reset without auth
    result = risk_kernel.reset_kill_switch(auth_token=None)
    
    assert result.success == False
    assert risk_kernel.is_kill_switch_active() == True
```

### TC-05.6: Reset After Cooldown Only

**Given**: Kill-switch triggered  
**When**: Reset attempted before 24h cooldown  
**Then**: Reset FAILS

```python
def test_reset_requires_cooldown():
    trigger_kill_switch("test", "Test")
    
    # Try immediate reset (with valid auth)
    result = risk_kernel.reset_kill_switch(
        auth_token=valid_auth(),
        review_id="review_123"
    )
    
    assert result.success == False
    assert "cooldown" in result.reason.lower()
```

### TC-05.7: Successful Reset After Requirements Met

**Given**: Kill-switch active for 24h+  
**And**: Review ID exists  
**And**: Valid authentication  
**When**: Reset attempted  
**Then**: Reset succeeds

```python
def test_successful_reset():
    trigger_kill_switch("test", "Test")
    
    # Simulate 24h passage
    mock_time_advance(hours=25)
    
    # Reset with all requirements
    result = risk_kernel.reset_kill_switch(
        auth_token=valid_auth(),
        review_id="review_123"
    )
    
    assert result.success == True
    assert risk_kernel.is_kill_switch_active() == False
```

### TC-05.8: Post-Reset State is Level 1

**Given**: Kill-switch reset successfully  
**When**: System resumes trading  
**Then**: Scaling level = 1 (Proof Mode)

```python
def test_post_reset_level():
    # ... reset kill-switch ...
    
    state = get_system_state()
    assert state.scaling_level == 1
```

---

## Kill-Switch State Storage

```python
class KillSwitchState:
    active: bool
    triggered_at: datetime
    trigger_reason: str
    trigger_type: str
    positions_at_trigger: List[Position]
    reset_at: Optional[datetime]
    reset_by: Optional[str]
    
    # Persisted to:
    # - Redis (fast access)
    # - PostgreSQL (durability)
    # - File backup (disaster recovery)
```

---

## Acceptance Criteria

- [ ] Kill-switch persists indefinitely
- [ ] Kill-switch survives restart
- [ ] All trades rejected when active
- [ ] No automatic reset
- [ ] Reset requires auth + cooldown + review
- [ ] Post-reset returns to Level 1
