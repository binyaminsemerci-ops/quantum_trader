# Audit Ledger Service

**Port**: 8010  
**Grunnlov**: §11 (Logging)  
**Authority**: Read-only (observes, never modifies)  

## Purpose

Immutable log of all system events. Every decision, every trade, every state change is recorded here.

## Interface

```python
class AuditLedger:
    async def log_event(self, event: AuditEvent) -> LogResult:
        """Record event to immutable ledger"""
        pass
    
    async def query_events(self, filter: EventFilter) -> List[AuditEvent]:
        """Query historical events"""
        pass
    
    async def get_trade_history(self, trade_id: str) -> TradeAuditTrail:
        """Complete audit trail for a specific trade"""
        pass
    
    async def export_report(self, period: str) -> Report:
        """Generate audit report for period"""
        pass
```

## What Gets Logged

- Every trade decision (with reasoning)
- Every VETO (with reason)
- Every position change
- Every risk event
- Every kill-switch activation
- Every policy violation
- Every system state change

## Events

**Listens to**: ALL events  
**Emits**: `audit.logged`, `audit.alert`

## Immutability

Once logged, events cannot be modified or deleted. This is the source of truth.
