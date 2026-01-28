# HarvestBrain: Profit Harvesting Microservice

**Version:** 1.0.0  
**Mode:** Shadow/Live  
**Input:** `quantum:stream:execution.result`  
**Output:** `quantum:stream:harvest.suggestions` (shadow) or `quantum:stream:trade.intent` (live)

## Overview

HarvestBrain implements **deterministic profit harvesting** based on R (return on risk):

- **R-based ladder:** Close partials at 0.5R, 1.0R, 1.5R
- **Break-even moves:** Move SL to entry price when R >= threshold
- **Trailing stops:** ATR-based trailing updates (v1+)
- **Fail-closed:** Kill-switch, dedup, shadow mode default
- **Idempotent:** All actions deduplicated via Redis TTL keys

## Configuration

**Location:** `/etc/quantum/harvest-brain.env`

**Key Settings:**
```bash
HARVEST_MODE=shadow          # or 'live'
HARVEST_MIN_R=0.5            # Don't harvest unless R >= 0.5
HARVEST_LADDER=0.5:0.25,... # R levels and position fractions to close
HARVEST_SET_BE_AT_R=0.5      # Move SL to break-even at this R
HARVEST_KILL_SWITCH_KEY=quantum:kill  # Fail-closed
```

## Operation

### Shadow Mode (Default)

```bash
HARVEST_MODE=shadow
```

- Publishes proposals to `quantum:stream:harvest.suggestions`
- No live orders created
- Safe for testing and validation
- Useful for backtesting and metrics

### Live Mode

```bash
HARVEST_MODE=live
```

- Publishes reduce-only intents to `quantum:stream:trade.intent`
- Execution service processes and submits orders
- Requires validation before switching from shadow
- Kill-switch (`quantum:kill=1`) stops all output

## Input Streams

| Stream | Purpose |
|--------|---------|
| `quantum:stream:execution.result` | Fills/executions → position updates |
| `quantum:stream:position.snapshot` | Fresh position snapshots (when available) |
| `quantum:stream:pnl.snapshot` | PnL snapshots (when available) |

**Note:** v1 derives position from execution fills only (position.snapshot not yet populated).

## Output Events

### Shadow Mode: `harvest.suggestions`

```json
{
  "intent_type": "HARVEST_PARTIAL",
  "symbol": "ETHUSDT",
  "side": "SELL",
  "qty": 0.25,
  "reason": "R=0.52 >= 0.5",
  "r_level": 0.52,
  "unrealized_pnl": 52.5,
  "dry_run": true,
  "source": "harvest_brain",
  "correlation_id": "harvest:ETHUSDT:0.5:...",
  "trace_id": "harvest:ETHUSDT:partial:...",
  "timestamp": "2026-01-18T..."
}
```

### Live Mode: `trade.intent`

Reduce-only intents sent to execution service:

```json
{
  "symbol": "ETHUSDT",
  "side": "SELL",
  "qty": 0.25,
  "intent_type": "REDUCE_ONLY",
  "reason": "R=0.52 >= 0.5",
  "reduce_only": true,
  "source": "harvest_brain",
  "correlation_id": "harvest:ETHUSDT:0.5:...",
  "trace_id": "harvest:ETHUSDT:partial:...",
  "timestamp": "2026-01-18T..."
}
```

## Idempotency

All harvesting actions are deduplicated via Redis keys:

```
quantum:dedup:harvest:<symbol>:<action>:<r_level>:<pnl_cents>
```

**TTL:** Configurable (`HARVEST_DEDUP_TTL_SEC`, default 900 seconds)

If same action submitted within TTL → skipped with debug log.

## Fail-Closed Safety

1. **Default mode:** Shadow (no live orders)
2. **Kill-switch:** If `quantum:kill=1` → no publishing
3. **Fresh data:** Skip harvesting if position data > 30s old
4. **Rate limiting:** Max 30 actions/minute
5. **Validation:** Symbol, side, qty must be provable

## systemd Integration

**Enable:**
```bash
sudo systemctl enable quantum-harvest-brain.service
sudo systemctl start quantum-harvest-brain.service
```

**Logs:**
```bash
sudo journalctl -u quantum-harvest-brain -f
```

**Status:**
```bash
sudo systemctl status quantum-harvest-brain
```

## Proof of Operation

**Service active:**
```bash
systemctl is-active quantum-harvest-brain
```

**Shadow suggestions being published:**
```bash
redis-cli XLEN quantum:stream:harvest.suggestions
redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 3
```

**Dedup working (no duplicates):**
```bash
# Same event → only one entry in stream
redis-cli KEYS "quantum:dedup:harvest:*" | wc -l
```

**Kill-switch verified:**
```bash
redis-cli SET quantum:kill 1
# Check logs - should see "Kill-switch active"
journalctl -u quantum-harvest-brain -n 5
redis-cli SET quantum:kill 0
```

## Rollback

Stop and disable service:
```bash
sudo systemctl stop quantum-harvest-brain
sudo systemctl disable quantum-harvest-brain
```

No cleanup needed (service-only, no database changes).

## Future Enhancements

- [ ] Integrate `position.snapshot` stream for fresh position data
- [ ] Implement ATR-based trailing stops
- [ ] Add profit-locking (move SL as position profits)
- [ ] Dynamic ladder adjustment based on volatility
- [ ] Metrics/Prometheus integration
