# Kill-Switch Tests

**Purpose**: Verify kill-switch always works  

## Critical Tests

### 1. Activation Tests

```python
class KillSwitchActivationTests:
    def test_manual_kill_switch(self):
        """Manual trigger closes all positions"""
        
    def test_daily_loss_auto_trigger(self):
        """5% loss auto-triggers kill-switch"""
        
    def test_drawdown_auto_trigger(self):
        """20% drawdown auto-triggers"""
        
    def test_data_failure_auto_trigger(self):
        """Data integrity fail triggers kill-switch"""
        
    def test_10_loss_series_trigger(self):
        """10 consecutive losses triggers"""
```

### 2. Execution Tests

```python
class KillSwitchExecutionTests:
    def test_all_positions_closed(self):
        """Every open position is closed"""
        
    def test_all_orders_cancelled(self):
        """Every pending order is cancelled"""
        
    def test_new_orders_blocked(self):
        """No new orders can be placed"""
        
    def test_execution_time(self):
        """Complete execution < 5 seconds"""
```

### 3. State Tests

```python
class KillSwitchStateTests:
    def test_flag_persists_restart(self):
        """Kill-switch survives system restart"""
        
    def test_no_auto_reset(self):
        """System cannot auto-reset kill-switch"""
        
    def test_state_logged(self):
        """Complete state snapshot saved"""
```

### 4. Reset Tests

```python
class KillSwitchResetTests:
    def test_requires_authentication(self):
        """Reset requires valid auth token"""
        
    def test_requires_cooldown(self):
        """Reset only after 24h cooldown"""
        
    def test_requires_review(self):
        """Reset requires incident review ID"""
        
    def test_resets_to_proof_mode(self):
        """After reset, system in Level 1"""
```

## Test Frequency

- Smoke test: Weekly (shadow mode)
- Full test: Monthly (testnet)
- Failover test: Quarterly (production-like)

## Zero Tolerance

Kill-switch tests have ZERO tolerance for failure.
If any test fails, system is not production-ready.
