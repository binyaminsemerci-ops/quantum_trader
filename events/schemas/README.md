# Event Schemas

All events follow strict schemas for validation.

## Trade Proposed Event

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["symbol", "direction", "entry_price", "stop_loss", "confidence"],
    "properties": {
        "symbol": {"type": "string", "pattern": "^[A-Z]+USDT$"},
        "direction": {"enum": ["LONG", "SHORT"]},
        "entry_price": {"type": "number", "minimum": 0},
        "stop_loss": {"type": "number", "minimum": 0},
        "take_profit": {"type": "number", "minimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "signal_source": {"type": "string"},
        "signal_id": {"type": "string", "format": "uuid"}
    }
}
```

## Policy Result Event

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["result", "trade_id", "grunnlover_checked"],
    "properties": {
        "result": {"enum": ["APPROVED", "REJECTED", "VETO"]},
        "trade_id": {"type": "string", "format": "uuid"},
        "grunnlover_checked": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "grunnlov_id": {"type": "integer", "minimum": 1, "maximum": 15},
                    "passed": {"type": "boolean"},
                    "detail": {"type": "string"}
                }
            }
        },
        "rejection_reason": {"type": "string"}
    }
}
```

## Position Update Event

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["position_id", "symbol", "side", "size", "entry_price", "unrealized_pnl"],
    "properties": {
        "position_id": {"type": "string", "format": "uuid"},
        "symbol": {"type": "string"},
        "side": {"enum": ["LONG", "SHORT"]},
        "size": {"type": "number", "minimum": 0},
        "entry_price": {"type": "number", "minimum": 0},
        "current_price": {"type": "number", "minimum": 0},
        "unrealized_pnl": {"type": "number"},
        "unrealized_pnl_pct": {"type": "number"},
        "stop_loss": {"type": "number"},
        "take_profit": {"type": "number"},
        "opened_at": {"type": "string", "format": "date-time"}
    }
}
```

## Kill Switch Event

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["trigger", "reason", "positions_closed"],
    "properties": {
        "trigger": {"enum": ["manual", "daily_loss", "drawdown", "data_integrity", "loss_series", "black_swan"]},
        "reason": {"type": "string"},
        "triggered_by": {"type": "string"},
        "positions_closed": {"type": "integer", "minimum": 0},
        "orders_cancelled": {"type": "integer", "minimum": 0},
        "total_loss_usd": {"type": "number"},
        "system_state_snapshot_id": {"type": "string", "format": "uuid"}
    }
}
```
