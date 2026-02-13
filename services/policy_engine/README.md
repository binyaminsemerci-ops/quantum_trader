# Policy Engine Service

**Port**: 8001  
**Grunnlov**: Enforces ALL 15 Grunnlover  
**Authority**: Level 2 (VETO)  

## Purpose

The Policy Engine is the guardian of all constitutional rules. Every trade decision passes through this service to verify compliance with the 15 Grunnlover.

## Interface

```python
class PolicyEngine:
    async def validate_trade(self, trade: Trade) -> PolicyResult:
        """Returns APPROVED, REJECTED, or VETO with reason"""
        pass
    
    async def check_grunnlov(self, grunnlov_id: int, context: dict) -> bool:
        """Check specific Grunnlov compliance"""
        pass
```

## Events

**Listens to**: `trade.proposed`, `position.change`, `risk.alert`  
**Emits**: `policy.approved`, `policy.rejected`, `policy.veto`
