# Trade Intent Schema Contract v1.1 BRIDGE-PATCH

**Status**: Production Standard with AI Sizing Bridge (2026-01-21)  
**Enforcement**: Required for all trade.intent publishers  
**Enhancement**: Supports AI-injected position sizing, leverage, and harvest policy

## What Changed (v1.0 → v1.1)

**BACKWARDS COMPATIBLE**: v1.0 payloads still accepted.

New capabilities:
- **Optional AI fields**: `ai_size_usd`, `ai_leverage`, `ai_harvest_policy`
- **Harvest policy**: Exit strategy control (scalper/swing/trend_runner)
- **Fail-closed**: Risk Governor enforces safety bounds in execution service

## Redis Stream Structure

### Stream Field Names
```python
{
    "event_type": "trade.intent",       # Required: Event type identifier
    "payload": "<JSON_STRING>",         # Required: JSON-encoded trade data
    "timestamp": "<ISO8601>",           # Required: Event timestamp
    "source": "<source_id>"             # Required: Publisher identifier
}
```

**CRITICAL**: The JSON data MUST be in the `"payload"` field, not `"data"`.

## Payload JSON Schema v1.1

### Required Core Fields
```python
{
    "symbol": str,              # Trading pair (e.g., "BTCUSDT")
    "action": str,              # "BUY", "SELL", or "CLOSE"
    "confidence": float,        # 0.0 - 1.0
    "timestamp": str,           # ISO8601 timestamp
}
```

### Optional: Size/Leverage (v1.0 style)
```python
{
    "position_size_usd": float, # USD value of position
    "leverage": float,          # Leverage multiplier (5..80)
}
```

### Optional: AI-Injected Fields (v1.1)
```python
{
    "ai_size_usd": float,               # AI-recommended position size
    "ai_leverage": float,               # AI-recommended leverage
    "ai_harvest_policy": dict,          # AI-recommended exit policy
    "risk_budget_usd": float,           # Risk budget constraint
}
```

AI fields take precedence if both v1.0 and AI fields present:
- If `ai_size_usd` present → use it (AI mode)
- Else if `position_size_usd` present → use it (legacy mode)
- Else → Risk Governor applies defaults

### Optional: Harvest Policy
```python
{
    "harvest_policy": {
        "mode": "scalper|swing|trend_runner",    # Exit strategy
        "trail_pct": float,                       # Trailing stop %
        "max_time_sec": int,                      # Max hold time (seconds)
        "partial_close_pct": float                # Partial close threshold
    }
}
```

**Harvest Modes**:
- **scalper**: Tight trailing (0.5%), short max_time (30 min)
- **swing**: Balanced trailing (1.0%), medium max_time (1 hour)
- **trend_runner**: Aggressive trailing (2.0%), long max_time (2 hours)

### Optional: Legacy/Metadata Fields
```python
{
    "entry_price": float,       # Target entry price
    "stop_loss": float,         # Stop loss price
    "take_profit": float,       # Take profit price
    "stop_loss_pct": float,     # Stop loss percentage
    "take_profit_pct": float,   # Take profit percentage
    "source": str,              # Signal source identifier
    "model": str,               # Model identifier
    "meta_strategy": str,       # Strategy name
    "regime": str,              # Market regime
}
```

## Schema Evolution Rules

1. **Adding fields**: Always optional, never break consumers
2. **Removing fields**: Deprecate for 30 days before removal
3. **Renaming fields**: Add new field, deprecate old field
4. **Changing types**: Create new field with `_v2` suffix

## Backwards Compatibility

### Deprecated Fields (removed after 2026-02-20)
- `"side"`: Use `"action"` instead (auto-converted for now)

## Validation Rules

1. **symbol**: Must match `/^[A-Z]{3,10}USDT$/`
2. **action**: Must be in ["BUY", "SELL", "CLOSE"]
3. **confidence**: Must be in range [0.0, 1.0]
4. **position_size_usd**: Must be > 0
5. **leverage**: Must be in range [1, 125]
6. **timestamp**: Must be ISO8601 format

## Publisher Implementation

### Python Example
```python
from ai_engine.services.eventbus_bridge import EventBusClient, validate_trade_intent

async def publish_trade_signal(signal: dict):
    # Validate before publishing
    errors = validate_trade_intent(signal)
    if errors:
        raise ValueError(f"Invalid trade intent: {errors}")
    
    async with EventBusClient() as bus:
        await bus.publish(
            stream_name="quantum:stream:trade.intent",
            event_type="trade.intent",
            payload=signal,
            source="ai-engine"
        )
```

## Consumer Implementation

### Python Example
```python
async for message in eventbus.subscribe_with_group(
    stream_name="quantum:stream:trade.intent",
    group_name="quantum:group:execution:trade.intent",
    consumer_name=f"execution-{hostname}-{pid}"
):
    # message["payload"] contains parsed JSON
    # message["_message_id"] contains Redis stream ID
    symbol = message["symbol"]
    action = message["action"]  # NOT "side"
    # ...
```

## Failure Modes

### Producer Failures
- **Missing required field**: Fail immediately, don't publish
- **Invalid field type**: Fail immediately, don't publish
- **Invalid field value**: Fail immediately, don't publish

### Consumer Failures
- **Missing "payload" field**: Log error, skip message, ACK to prevent retry loop
- **JSON decode error**: Log error, skip message, ACK to prevent retry loop
- **Unknown field**: Ignore field, continue processing
- **Missing optional field**: Use default value

## Monitoring

Track these metrics in production:
- `trade_intent_validation_failures`: Counter of validation failures
- `trade_intent_schema_mismatches`: Counter of deprecated field usage
- `trade_intent_parse_errors`: Counter of JSON decode errors

## Testing

Run schema validation tests before deployment:
```bash
python -m pytest tests/test_trade_intent_schema.py -v
```

## Version History

- **v1.0** (2026-01-21): Initial standard after P0.D.4 investigation
  - Standardized `"payload"` field name
  - Standardized `"action"` field (deprecated `"side"`)
  - Required field validation
  - Fail-closed producer validation
