# Entry Gate Service

**Port**: 8007  
**Grunnlov**: §5 (Stop Required), §9 (Pre-Flight)  
**Authority**: Level 5  

## Purpose

The final gatekeeper for new position entries. Ensures all conditions are met before allowing a trade to proceed to execution.

## Interface

```python
class EntryGate:
    async def evaluate_entry(self, trade: Trade) -> EntryResult:
        """APPROVED or DENIED with reason"""
        pass
    
    async def check_pre_conditions(self, trade: Trade) -> PreConditionResult:
        """Verifies all entry prerequisites"""
        pass
    
    async def get_entry_queue(self) -> List[PendingEntry]:
        """Trades waiting for entry approval"""
        pass
```

## Pre-Entry Checklist

1. Stop-loss defined? (Grunnlov §5)
2. Pre-flight passed? (Grunnlov §9)
3. Not in no-trade period? (Grunnlov §8)
4. Risk approved?
5. Capital allocated?
6. No conflicting positions?

## Events

**Listens to**: `capital.allocated`, `pre_flight.passed`  
**Emits**: `entry.approved`, `entry.denied`, `entry.queued`
