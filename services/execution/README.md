# Execution Service

**Port**: 8009  
**Grunnlov**: §12 (Execution Quality), §13 (Slippage)  
**Authority**: Execution only (no decision power)  

## Purpose

Handles all order execution. Receives approved trades and executes them on the exchange. No decision-making, pure execution.

## Interface

```python
class Execution:
    async def execute_order(self, order: Order) -> ExecutionResult:
        """Execute order on exchange"""
        pass
    
    async def cancel_order(self, order_id: str) -> CancelResult:
        """Cancel pending order"""
        pass
    
    async def get_execution_quality(self, period: str) -> QualityReport:
        """Slippage and fill rate metrics"""
        pass
    
    async def emergency_close_all(self) -> CloseAllResult:
        """Kill-switch execution"""
        pass
```

## Execution Rules

1. Never retry more than 3 times
2. Log all execution attempts
3. Report slippage immediately
4. Support kill-switch at any time

## Events

**Listens to**: `entry.approved`, `exit.decision`, `kill_switch.activated`  
**Emits**: `order.submitted`, `order.filled`, `order.failed`, `slippage.report`
