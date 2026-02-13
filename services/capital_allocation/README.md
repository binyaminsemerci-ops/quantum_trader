# Capital Allocation Service

**Port**: 8005  
**Grunnlov**: §1 (2% Rule), Scaling Levels  
**Authority**: Level 3  

## Purpose

Determines position sizes, tracks capital allocation, and enforces scaling level rules.

## Interface

```python
class CapitalAllocation:
    async def calculate_position_size(self, trade: Trade) -> PositionSize:
        """Kelly-modified position sizing"""
        pass
    
    async def get_current_allocation(self) -> AllocationReport:
        """Current capital distribution"""
        pass
    
    async def get_scaling_level(self) -> ScalingLevel:
        """Current scaling level (1-4)"""
        pass
    
    async def check_allocation_available(self, amount: float) -> bool:
        """Can we allocate this much?"""
        pass
```

## Events

**Listens to**: `risk.approved`, `position.opened`, `position.closed`  
**Emits**: `capital.allocated`, `capital.released`, `scaling.change`
