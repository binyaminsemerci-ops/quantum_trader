# Stress Tests

**Purpose**: Test system under extreme conditions  

## Test Categories

### 1. Loss Series Tests

```python
class LossSeriesTests:
    def test_3_consecutive_losses(self):
        """System reduces size by 25%"""
        
    def test_5_consecutive_losses(self):
        """System reduces size by 50%, cooldown"""
        
    def test_7_consecutive_losses(self):
        """System enters observer mode"""
        
    def test_10_consecutive_losses(self):
        """Full stop, shadow-only mode"""
```

### 2. Volatility Tests

```python
class VolatilityTests:
    def test_double_volatility_response(self):
        """Size reduction to 50% on 2x vol"""
        
    def test_extreme_volatility_flat(self):
        """System goes flat on extreme vol"""
        
    def test_flash_crash_detection(self):
        """10% move in 1h triggers kill-switch"""
```

### 3. Load Tests

```python
class LoadTests:
    def test_100_signals_per_second(self):
        """System handles signal burst"""
        
    def test_concurrent_position_updates(self):
        """Multiple position updates don't conflict"""
        
    def test_audit_log_throughput(self):
        """Logging keeps up with event rate"""
```

### 4. Recovery Tests

```python
class RecoveryTests:
    def test_recovery_from_5_loss_series(self):
        """Gradual size restoration after series"""
        
    def test_recovery_from_drawdown(self):
        """Proper scaling back after drawdown"""
        
    def test_recovery_from_kill_switch(self):
        """System restart after kill-switch"""
```

## Stress Test Execution

```bash
pytest tests/stress_tests/ -v --timeout=300
```

Run weekly or before major changes.
