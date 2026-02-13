# Risk Kernel Service

**Port**: 8002  
**Grunnlov**: §1 (2% rule), §2 (5% daily), Drawdown  
**Authority**: Level 1 (VETO)  

## Purpose

The Risk Kernel is the last line of defense. It has VETO power over any trade that would breach risk limits.

## Interface

```python
class RiskKernel:
    async def evaluate_risk(self, trade: Trade) -> RiskResult:
        """PASS or VETO with risk metrics"""
        pass
    
    async def get_current_exposure(self) -> ExposureReport:
        """Current portfolio risk state"""
        pass
    
    async def check_drawdown_status(self) -> DrawdownStatus:
        """Current drawdown level and size multiplier"""
        pass
```

## Events

**Listens to**: `policy.approved`, `position.pnl.update`, `market.volatility`  
**Emits**: `risk.approved`, `risk.veto`, `risk.alert`, `circuit_breaker.triggered`
