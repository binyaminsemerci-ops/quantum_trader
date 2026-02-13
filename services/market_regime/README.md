# Market Regime Service

**Port**: 8003  
**Grunnlov**: §14 (Regime Changes), §6 (Data Quality)  
**Authority**: Advisory (provides context)  

## Purpose

Detects current market regime (trending, ranging, volatile, calm) and provides context for other services.

## Interface

```python
class MarketRegime:
    async def get_current_regime(self, symbol: str) -> Regime:
        """Returns current market regime classification"""
        pass
    
    async def detect_regime_change(self) -> Optional[RegimeChange]:
        """Detects if regime has changed since last check"""
        pass
    
    async def get_volatility_state(self) -> VolatilityState:
        """Current volatility classification"""
        pass
```

## Events

**Listens to**: `market.price`, `market.volume`, `indicator.update`  
**Emits**: `regime.update`, `regime.change`, `volatility.alert`
