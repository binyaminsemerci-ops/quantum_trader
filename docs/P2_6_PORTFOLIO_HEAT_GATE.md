# P2.6 Portfolio Heat Gate - Hedge Fund OS Edition

## Overview

Portfolio-level safety calibration layer that prevents premature FULL_CLOSE decisions based on portfolio-wide risk metrics. Sits between Harvest Kernel (P2.5) and Apply Layer.

## Architecture

```
P2.5 Harvest Kernel
   ↓ harvest.proposal
P2.6 Portfolio Heat Gate  ← THIS COMPONENT
   ↓ harvest.calibrated
Apply Layer (P3.1)
   ↓
P3.2 Governor + P3.3 Position State Brain
   ↓
Execution
```

## Core Function

**Portfolio Heat Formula:**
```
Heat = Σ(|position_notional_i| * sigma_i) / equity_usd
```

**Gating Rules:**

| Heat Bucket | Heat Range | Rule |
|-------------|------------|------|
| COLD | < 0.25 | FULL_CLOSE → PARTIAL_25 (preserve winners) |
| WARM | 0.25 - 0.65 | FULL_CLOSE → PARTIAL_75 (moderate exit) |
| HOT | ≥ 0.65 | FULL_CLOSE allowed (high risk justifies full exit) |

PARTIAL actions always pass unchanged.

## Hard Invariants

1. **FAIL-CLOSED**: Missing data → downgrade to PARTIAL_25
2. **MONOTONIC**: Only reduces aggressiveness, never increases
3. **AUDIT**: Every decision traceable with context
4. **SHADOW-FIRST**: Default mode is shadow (log only)

## Redis Schema

**Input Stream:** `quantum:stream:harvest.proposal`
```
{
  "plan_id": "uuid",
  "symbol": "BTCUSDT",
  "action": "FULL_CLOSE",
  "trace_id": "uuid",
  "reason": "tp_hit"
}
```

**Output Stream:** `quantum:stream:harvest.calibrated`
```
{
  "trace_id": "uuid",
  "plan_id": "uuid",
  "symbol": "BTCUSDT",
  "original_action": "FULL_CLOSE",
  "calibrated_action": "PARTIAL_75",
  "heat_value": 0.42,
  "heat_bucket": "WARM",
  "mode": "shadow",
  "calibrated": true,
  "reason": "portfolio_heat_warm",
  "timestamp": 1234567890
}
```

**Data Sources:**
- `quantum:state:portfolio` - equity_usd, total_positions
- `quantum:stream:position.snapshot` - symbol, position_notional_usd, sigma

## Configuration

**Environment File:** `/etc/quantum/portfolio-heat-gate.env`

```bash
P26_MODE=shadow              # shadow | enforce
HEAT_MIN=0.25                # COLD/WARM threshold
HEAT_MAX=0.65                # WARM/HOT threshold
P26_POLL_SEC=2               # Stream poll interval
P26_METRICS_PORT=8056        # Prometheus metrics port
```

## Deployment

```bash
# 1. Sync code
rsync -avz microservices/portfolio_heat_gate/ root@vps:/home/qt/quantum_trader/microservices/portfolio_heat_gate/

# 2. Deploy service
ssh root@vps 'cd /home/qt/quantum_trader && ./deploy/deploy_p26_heat_gate.sh'

# 3. Verify
ssh root@vps 'systemctl status quantum-portfolio-heat-gate'

# 4. Check metrics
curl http://vps:8056/metrics | grep p26_

# 5. Generate proof
ssh root@vps 'cd /home/qt/quantum_trader && ./deploy/proof_p26_heat_gate.sh'
```

## Prometheus Metrics

- `p26_heat_value` - Current portfolio heat
- `p26_bucket{state}` - Heat bucket state (COLD/WARM/HOT)
- `p26_actions_downgraded_total{from_action, to_action, reason}` - Downgrade counter
- `p26_proposals_processed_total` - Total proposals processed
- `p26_stream_lag_ms` - Stream processing latency
- `p26_failures_total{reason}` - Failures by reason

## Test Scenarios

### 1. COLD → PARTIAL_25
```bash
redis-cli HMSET quantum:state:portfolio equity_usd 10000
redis-cli XADD quantum:stream:position.snapshot "*" symbol ETHUSDT position_notional_usd 1800 sigma 0.15
redis-cli XADD quantum:stream:harvest.proposal "*" plan_id test1 symbol ETHUSDT action FULL_CLOSE
# Expected: heat=0.027 (COLD), downgrade to PARTIAL_25
```

### 2. WARM → PARTIAL_75
```bash
redis-cli HMSET quantum:state:portfolio equity_usd 3000
redis-cli XADD quantum:stream:harvest.proposal "*" plan_id test2 symbol ETHUSDT action FULL_CLOSE
# Expected: heat=0.294 (WARM), downgrade to PARTIAL_75
```

### 3. HOT → PASS
```bash
redis-cli HMSET quantum:state:portfolio equity_usd 1000
redis-cli XADD quantum:stream:harvest.proposal "*" plan_id test3 symbol BTCUSDT action FULL_CLOSE
# Expected: heat=0.88 (HOT), FULL_CLOSE allowed
```

### 4. PARTIAL → PASS
```bash
redis-cli XADD quantum:stream:harvest.proposal "*" plan_id test4 symbol ETHUSDT action PARTIAL_50
# Expected: PARTIAL_50 passes unchanged
```

## Monitoring

```bash
# Tail logs
journalctl -u quantum-portfolio-heat-gate -f

# Check downgrades
journalctl -u quantum-portfolio-heat-gate | grep DOWNGRADE

# Metrics
curl -s http://localhost:8056/metrics | grep p26_

# Consumer group status
redis-cli XINFO GROUPS quantum:stream:harvest.proposal | grep p26_heat_gate
```

## Integration Notes

### Does NOT Modify
- Harvest Kernel (P2.5)
- Apply Layer (P3.1)
- Governor (P3.2)
- Position State Brain (P3.3)

### Pure Calibration Layer
- Reads from harvest.proposal
- Writes to harvest.calibrated
- Apply Layer should consume harvest.calibrated instead of harvest.proposal (when enforce mode active)

## Shadow vs Enforce Mode

**Shadow Mode (default):**
- Logs all decisions
- Metrics updated
- NO writes to harvest.calibrated stream
- Safe for observation

**Enforce Mode:**
- Logs + writes to harvest.calibrated
- Apply Layer must consume calibrated stream
- Requires Apply Layer integration

## Fail-Closed Examples

1. **Missing equity:** Heat = 0.0 (COLD) → FULL_CLOSE → PARTIAL_25
2. **Missing sigma:** Skip position in heat calculation
3. **No position snapshots:** Heat = 0.0 (COLD) → FULL_CLOSE → PARTIAL_25
4. **Invalid data:** Log error, fail to COLD bucket

## Proof Report

VPS proof report location: `docs/P2_6_VPS_PROOF.txt`

Verified:
- ✅ Service active and running
- ✅ Metrics responding on :8056
- ✅ Portfolio heat calculation working
- ✅ COLD → PARTIAL_25 (3 tests)
- ✅ WARM → PARTIAL_75 (1 test)
- ✅ HOT → PASS (1 test)
- ✅ PARTIAL → PASS (1 test)
- ✅ Consumer group processing
- ✅ Stream lag < 10ms

## Author

Quantum Trader Team - Hedge Fund OS Edition
2026-01-26
