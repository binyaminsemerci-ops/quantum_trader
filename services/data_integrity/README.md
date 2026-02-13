# Data Integrity Service

**Port**: 8004  
**Grunnlov**: §6 (95% Data Quality), §7 (Fail-Closed)  
**Authority**: VETO (can halt trading on data issues)  

## Purpose

Ensures all data feeding the system is accurate, timely, and complete. No trading occurs on bad data.

## Interface

```python
class DataIntegrity:
    async def validate_market_data(self, data: MarketData) -> IntegrityResult:
        """Validates incoming market data"""
        pass
    
    async def get_data_quality_score(self) -> float:
        """Returns 0.0-1.0 quality score. <0.95 = halt"""
        pass
    
    async def check_data_freshness(self) -> FreshnessResult:
        """Ensures data is not stale"""
        pass
```

## Events

**Listens to**: `data.received`, `exchange.heartbeat`  
**Emits**: `data.valid`, `data.invalid`, `data.stale`, `integrity.fail`
