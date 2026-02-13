# Shadow Mode Tests

**Purpose**: Test trading logic without real money  

## What is Shadow Mode?

Shadow mode runs the complete trading system but:
- Does NOT execute real orders
- DOES make all decisions as if live
- DOES log everything
- DOES track hypothetical P&L

## Use Cases

### 1. Strategy Validation

Before deploying any change:
1. Run shadow mode minimum 50 trades
2. Compare results to backtest
3. Verify no Grunnlov violations
4. Check decision paths are correct

### 2. Post-Kill-Switch Recovery

After kill-switch:
1. Run shadow mode minimum 10 trades
2. Verify system health
3. Confirm risk calculations correct
4. Only then proceed to live

### 3. New Feature Testing

For any new feature:
1. Deploy to shadow mode first
2. Run parallel to production
3. Compare decisions
4. Verify no regressions

## Interface

```python
class ShadowMode:
    def enable(self):
        """Enable shadow mode"""
        pass
    
    def disable(self):
        """Disable shadow mode"""
        pass
    
    def get_shadow_trades(self) -> List[ShadowTrade]:
        """Get all shadow trades"""
        pass
    
    def get_shadow_performance(self) -> Performance:
        """Calculate hypothetical performance"""
        pass
```

## Shadow vs Live Comparison

```python
def compare_shadow_to_live():
    """
    After running both, compare:
    - Same signals received?
    - Same decisions made?
    - Same risk limits applied?
    - Same exit logic triggered?
    """
```

## Shadow Mode Tests

```python
class ShadowModeTests:
    def test_no_real_orders(self):
        """Verify no orders sent to exchange"""
        
    def test_complete_decision_path(self):
        """All services involved in decisions"""
        
    def test_accurate_pnl_tracking(self):
        """P&L calculation correct"""
        
    def test_grunnlov_enforcement(self):
        """All Grunnlover checked"""
```
