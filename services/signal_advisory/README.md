# Signal Advisory Service

**Port**: 8006  
**Grunnlov**: §15 (AI is Advisory)  
**Authority**: Advisory ONLY (no decision power)  

## Purpose

Receives and processes AI/ML signals. These are ADVISORY ONLY - they inform but do not decide.

## Interface

```python
class SignalAdvisory:
    async def receive_signal(self, signal: Signal) -> SignalAck:
        """Receives AI signal for logging and forwarding"""
        pass
    
    async def get_signal_confidence(self, signal_id: str) -> float:
        """Returns confidence score 0.0-1.0"""
        pass
    
    async def get_active_signals(self) -> List[Signal]:
        """Returns all active advisory signals"""
        pass
```

## Events

**Listens to**: `ai.signal.generated`, `model.prediction`  
**Emits**: `signal.advisory`, `signal.expired`

## Important

Signals from this service are ALWAYS labeled as advisory. No downstream service should treat them as commands.
