# P3 Apply Layer - Architecture & Operations

## Overview

The Apply Layer (P3) consumes Harvest Proposals from P2 and creates executable Apply Plans. It's the bridge between calc-only phases (P0-P2) and actual execution.

## Architecture

```
P2 Harvest Proposals (Redis) → Apply Layer → Apply Plans → Execution → Results
                                    ↓                           ↓
                              quantum:stream:        quantum:stream:
                              apply.plan             apply.result
```

### Components

1. **Apply Layer Service** (`microservices/apply_layer/main.py`)
   - Poll loop (5s default)
   - Reads harvest proposals from Redis
   - Creates deterministic apply plans
   - Publishes plans to stream
   - Executes plans (mode-dependent)
   - Publishes results to stream

2. **Redis Streams**
   - `quantum:stream:apply.plan`: All plans (EXECUTE/SKIP/BLOCKED)
   - `quantum:stream:apply.result`: Execution outcomes

3. **Prometheus Metrics** (port 8043)
   - `quantum_apply_plan_total{symbol,decision}`: Plans created
   - `quantum_apply_execute_total{symbol,step,status}`: Executions
   - `quantum_apply_dedupe_hits_total`: Duplicate detections
   - `quantum_apply_last_success_epoch{symbol}`: Last execution timestamp

## Modes

### P3.0 - Dry Run (Default)
- **APPLY_MODE=dry_run**
- Plans created and published
- NO execution (would_execute=true in results)
- Full audit trail
- Safe to run in production

### P3.1 - Testnet
- **APPLY_MODE=testnet**
- Plans executed against Binance testnet
- Requires `BINANCE_TESTNET_API_KEY` and `BINANCE_TESTNET_API_SECRET`
- Full execution with order tracking

## Idempotency Model

Every apply plan has a stable `plan_id` (SHA256 hash of: symbol, action, kill_score, new_sl_proposed, computed_at_utc).

**Dedupe mechanism:**
1. Plan created → `plan_id` computed
2. Redis SETNX `quantum:apply:dedupe:<plan_id>` with TTL (6h default)
3. If key exists → SKIP (duplicate)
4. If new → EXECUTE (or other decision based on gates)

**Guarantees:**
- Same proposal → same plan_id
- Prevents double execution
- TTL ensures keys don't accumulate forever

## Safety Gates

### 1. Kill Switch
- **APPLY_KILL_SWITCH=true** → blocks ALL executions
- Emergency stop for system-wide halt

### 2. Allowlist
- **APPLY_ALLOWLIST=BTCUSDT** (default)
- Only allowlisted symbols can execute
- Others marked SKIP (not_in_allowlist)

### 3. Kill Score Thresholds
- **K >= 0.80** (K_BLOCK_CRITICAL) → block ALL actions
- **K >= 0.60** (K_BLOCK_WARNING) → allow CLOSE/tighten only, block risk increase
- **K < 0.60** → all actions allowed (subject to other gates)

### 4. Idempotency
- Prevents duplicate execution via Redis dedupe keys

## Apply Plan Schema

```python
{
  "plan_id": "abc123",           # Stable hash
  "symbol": "BTCUSDT",
  "action": "PARTIAL_75",        # From harvest_action
  "kill_score": 0.527,
  "k_components": {              # K breakdown
    "regime_flip": 0.0,
    "sigma_spike": 0.14,
    "ts_drop": 0.054,
    "age_penalty": 0.042
  },
  "new_sl_proposed": 100.2,
  "R_net": 7.01,
  "decision": "EXECUTE",         # EXECUTE/SKIP/BLOCKED/ERROR
  "reason_codes": [...],         # Why this decision
  "steps": [                     # Execution steps
    {
      "step": "CLOSE_PARTIAL_75",
      "type": "market_reduce_only",
      "side": "close",
      "pct": 75.0
    }
  ],
  "timestamp": 1769130500
}
```

## Apply Result Schema

```python
{
  "plan_id": "abc123",
  "symbol": "BTCUSDT",
  "decision": "EXECUTE",
  "executed": true,              # Actually executed?
  "would_execute": false,        # Would execute in dry_run?
  "steps_results": [
    {
      "step": "CLOSE_PARTIAL_75",
      "status": "success",
      "details": "...",
      "order_id": "12345"
    }
  ],
  "error": null,
  "timestamp": 1769130505
}
```

## Harvest Action Mapping

| Harvest Action | Apply Steps |
|----------------|-------------|
| `FULL_CLOSE_PROPOSED` | CLOSE_FULL (market reduceOnly 100%) |
| `PARTIAL_75` | CLOSE_PARTIAL_75 (market reduceOnly 75%) |
| `PARTIAL_50` | CLOSE_PARTIAL_50 (market reduceOnly 50%) |
| `UPDATE_SL` | UPDATE_SL (modify stop loss order) |
| `HOLD` | No action (SKIP) |

## Configuration

See `deployment/config/apply-layer.env` for all settings.

**Critical settings:**
- `APPLY_MODE`: dry_run or testnet
- `APPLY_ALLOWLIST`: Symbols allowed to execute (default: BTCUSDT)
- `K_BLOCK_CRITICAL`: Kill score threshold for blocking all actions (default: 0.80)
- `K_BLOCK_WARNING`: Kill score threshold for blocking risk increase (default: 0.60)

## Operator Runbook

### Start Service

```bash
# Copy env if missing
sudo cp deployment/config/apply-layer.env /etc/quantum/apply-layer.env

# Install systemd unit
sudo cp deployment/systemd/quantum-apply-layer.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable quantum-apply-layer
sudo systemctl start quantum-apply-layer

# Check status
sudo systemctl status quantum-apply-layer
journalctl -u quantum-apply-layer -f
```

### Verify Operation

```bash
# Check plans being created
redis-cli XLEN quantum:stream:apply.plan

# Read recent plans
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3

# Check results
redis-cli XLEN quantum:stream:apply.result
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3

# Check metrics
curl http://localhost:8043/metrics | grep quantum_apply
```

### Enable Testnet Mode

1. **Add Binance credentials** to `/etc/quantum/apply-layer.env`:
   ```bash
   BINANCE_TESTNET_API_KEY=your_key
   BINANCE_TESTNET_API_SECRET=your_secret
   ```

2. **Change mode**:
   ```bash
   sudo sed -i 's/APPLY_MODE=dry_run/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env
   ```

3. **Restart service**:
   ```bash
   sudo systemctl restart quantum-apply-layer
   ```

4. **Verify testnet execution**:
   ```bash
   journalctl -u quantum-apply-layer -f | grep TESTNET
   redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 1
   ```

### Emergency Stop

```bash
# Set kill switch
sudo sed -i 's/APPLY_KILL_SWITCH=false/APPLY_KILL_SWITCH=true/' /etc/quantum/apply-layer.env

# Restart
sudo systemctl restart quantum-apply-layer

# Verify (all plans should be BLOCKED)
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3
```

### Adjust Allowlist

```bash
# Add ETHUSDT to allowlist
sudo sed -i 's/APPLY_ALLOWLIST=BTCUSDT/APPLY_ALLOWLIST=BTCUSDT,ETHUSDT/' /etc/quantum/apply-layer.env

# Restart
sudo systemctl restart quantum-apply-layer
```

### Monitor Dedupe Hits

```bash
# Check for duplicate plans
curl -s http://localhost:8043/metrics | grep dedupe_hits

# List dedupe keys in Redis
redis-cli KEYS "quantum:apply:dedupe:*" | wc -l
```

## Troubleshooting

### No plans created
- Check harvest proposal exists: `redis-cli HGETALL quantum:harvest:proposal:BTCUSDT`
- Check service logs: `journalctl -u quantum-apply-layer -n 50`
- Verify symbols config: `grep SYMBOLS /etc/quantum/apply-layer.env`

### Plans created but all SKIP
- Check allowlist: `grep ALLOWLIST /etc/quantum/apply-layer.env`
- Check kill switch: `grep KILL_SWITCH /etc/quantum/apply-layer.env`
- Review reason_codes in apply.plan stream

### Testnet execution fails
- Verify credentials: `grep BINANCE /etc/quantum/apply-layer.env`
- Check Binance testnet status
- Review error field in apply.result stream

### High dedupe_hits
- Normal if harvest proposals unchanged
- Indicates idempotency working correctly
- If excessive, check harvest proposal update frequency

## Integration with P0-P2

**No changes required to existing services:**
- MarketState (P0.5) continues as-is
- Risk Proposal (P1.5) continues as-is
- Harvest Proposal (P2.5-P2.7) continues as-is

Apply Layer is purely additive - it reads harvest proposals but never modifies them.

## Next Steps (P3.2+)

1. **Governor integration**: Daily/hourly execution limits per symbol
2. **Position state tracking**: Verify position exists before execution
3. **Advanced order types**: Limit orders, trailing stops
4. **Multi-exchange**: Add FTX/Bybit testnet support
5. **ML-based execution**: Optimal fill strategies

## Security Considerations

1. **Credentials**: Never commit API keys; use systemd EnvironmentFile or secrets manager
2. **Allowlist**: Start with 1 symbol (BTCUSDT), expand gradually
3. **Kill switch**: Test emergency stop procedure regularly
4. **Audit trail**: Retain apply.plan and apply.result streams for compliance
5. **Position size**: Implement max position limits (future P3.2)

## Testing

See proof pack scripts:
- `ops/p3_proof_dry_run.sh`: Verify dry_run mode
- `ops/p3_proof_testnet.sh`: Verify testnet execution
- `docs/P3_PROOF_PACK.md`: Proof pack template
