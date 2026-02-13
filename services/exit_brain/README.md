# Exit Brain Service

**Port**: 8008  
**Grunnlov**: §3 (Exits), §4 (Trailing), §5 (Stop-Loss)  
**Authority**: Level 4 (Autonomous for exits)  

## Purpose

Manages all position exits. AI signals entries, but ExitBrain controls exits autonomously using the 5 Exit Formulas.

## The 5 Exit Formulas

1. **Stop-Loss** - Mandatory, never removed
2. **Take-Profit** - Tiered R-multiple scaling
3. **Time-Based** - Max hold time 72h
4. **Regime-Change** - Exit on condition change
5. **Circuit-Breaker** - Emergency exit

## Interface

```python
class ExitBrain:
    async def evaluate_position(self, position: Position) -> ExitDecision:
        """Should we exit? Which formula?"""
        pass
    
    async def update_trailing_stop(self, position: Position) -> StopUpdate:
        """Adjust trailing stop if conditions met"""
        pass
    
    async def force_exit(self, position_id: str, reason: str) -> ExitResult:
        """Emergency or time-based forced exit"""
        pass
```

## Events

**Listens to**: `position.opened`, `position.pnl.update`, `regime.change`, `circuit_breaker.triggered`  
**Emits**: `exit.decision`, `exit.executed`, `stop.updated`
