# No-Trade Calendar Module

**Grunnlov**: §8 (Ingen Holiday-Trading)  
**Purpose**: Tracks when NOT to trade  

## Calendar Types

### Absolute (No override possible)

- CME Bitcoin Futures Expiry (±4h)
- CME Options Expiry (±4h)
- Kill-switch active
- Daily loss limit hit
- Data integrity failure

### Conditional (Override with justification)

- FOMC announcements
- Major economic data
- High funding rate periods
- Low liquidity weekends

### Human (Warning + acknowledgment)

- Minor economic events
- Off-hours trading
- Low confidence signals

## Interface

```python
class NoTradeCalendar:
    def is_no_trade_period(self, timestamp: datetime) -> NoTradeResult:
        """Check if current time is a no-trade period"""
        pass
    
    def get_upcoming_no_trade_events(self, days: int = 7) -> List[NoTradeEvent]:
        """Get no-trade events in next N days"""
        pass
    
    def can_override(self, event: NoTradeEvent) -> bool:
        """Check if event can be overridden"""
        pass
```

## Data Sources

- CME calendar (futures expiry)
- Economic calendar (FOMC, CPI, NFP)
- Exchange maintenance schedules
- Custom blackout periods
