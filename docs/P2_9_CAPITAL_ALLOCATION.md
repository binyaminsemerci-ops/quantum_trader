# P2.9 Capital Allocation Brain (Fund-Grade)

**Component ID**: P2.9  
**Service**: quantum-capital-allocation  
**Port**: 8059  
**Status**: PRODUCTION  
**Created**: 2026-01-28

---

## Overview

P2.9 Capital Allocation Brain dynamically allocates portfolio budget to per-symbol & per-cluster exposure targets using regime detection, cluster performance, drawdown zones, and risk state. This sits between P2.8 Budget Engine and Governor, providing intelligent capital distribution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    P2.9 ALLOCATION BRAIN                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Inputs:                                                    │
│  • quantum:portfolio:budget:{symbol}  (P2.8 budgets)      │
│  • quantum:state:portfolio            (PSP state)         │
│  • quantum:cluster:stress:{cluster}   (Cluster risk)      │
│  • quantum:stream:regime.state        (Market regime)     │
│                                                             │
│  Processing:                                                │
│  • Regime factor (TREND/MR/CHOP)                           │
│  • Cluster factor (1 - stress)                             │
│  • Drawdown factor (LOW/MID/HIGH zones)                    │
│  • Performance factor (sigmoid(alpha))                     │
│                                                             │
│  Outputs:                                                   │
│  • quantum:allocation:target:{symbol} (Target hashes)     │
│  • quantum:stream:allocation.decision (Event stream)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Allocation Formula

```
target_usd = base_budget × regime_factor × cluster_factor × drawdown_factor × performance_factor
```

### Regime Factors

| Regime         | Factor | Description                    |
|----------------|--------|--------------------------------|
| TREND          | 1.2    | Trending market, increase size |
| MEAN_REVERSION | 1.0    | Neutral, use base budget       |
| CHOP           | 0.6    | Choppy market, reduce size     |

### Cluster Factor

```
cluster_factor = clamp(1 - cluster_stress, 0.3, 1.0)
```

- Low stress → high allocation (up to 1.0)
- High stress → reduced allocation (down to 0.3)

### Drawdown Factors

| Zone      | Drawdown Range | Factor | Description                |
|-----------|----------------|--------|----------------------------|
| LOW       | < 5%           | 1.0    | Full allocation            |
| MID       | 5% - 15%       | 0.7    | Reduced allocation         |
| HIGH      | > 15%          | 0.4    | Significantly reduced      |

### Performance Factor

```
performance_factor = sigmoid(cluster_alpha) = 1 / (1 + exp(-alpha))
```

- Positive alpha → factor > 0.5 (increase allocation)
- Negative alpha → factor < 0.5 (decrease allocation)

## Safety Gates

### Per-Symbol Cap

**Max 40% of portfolio equity per symbol**

```python
max_symbol_usd = portfolio.equity_usd × 0.40

if target_usd > max_symbol_usd:
    target_usd = max_symbol_usd
```

### Per-Cluster Cap

**Max 60% of portfolio equity per cluster**

```python
cluster_total = sum(target_usd for symbol in cluster)
max_cluster_usd = portfolio.equity_usd × 0.60

if cluster_total > max_cluster_usd:
    scale = max_cluster_usd / cluster_total
    for symbol in cluster:
        symbol.target_usd *= scale
```

### Stale Data Fallback

**If input data > 60s old → use base budget (no multipliers)**

```python
if timestamp_age > 60:
    target_usd = base_budget  # Fail-safe: pass-through
    log_fallback()
```

## Operation Modes

### Shadow Mode (Default)

- Computes allocations
- Logs targets
- Streams decisions
- **Does NOT write** quantum:allocation:target:{symbol}
- Zero impact on trading

### Enforce Mode

- Computes allocations
- **Writes** quantum:allocation:target:{symbol}
- Streams decisions
- Governor reads allocation targets

**Activation**:
```bash
sed -i 's/P29_MODE=shadow/P29_MODE=enforce/' /etc/quantum/capital-allocation.env
systemctl restart quantum-capital-allocation
```

## Redis Keys

### Inputs

| Key Pattern                         | Source | TTL   | Description            |
|-------------------------------------|--------|-------|------------------------|
| quantum:portfolio:budget:{symbol}   | P2.8   | 300s  | Base budget per symbol |
| quantum:state:portfolio             | PSP    | 120s  | Portfolio equity/DD    |
| quantum:cluster:stress:{cluster}    | CLM    | 60s   | Cluster stress/alpha   |
| quantum:stream:regime.state         | Regime | N/A   | Market regime stream   |

### Outputs

| Key Pattern                         | Type   | TTL   | Fields                                    |
|-------------------------------------|--------|-------|-------------------------------------------|
| quantum:allocation:target:{symbol}  | Hash   | 300s  | target_usd, weight, cluster_id, regime,   |
|                                     |        |       | drawdown_zone, confidence, timestamp, mode|
| quantum:stream:allocation.decision  | Stream | N/A   | All target fields + regime/cluster/       |
|                                     |        |       | drawdown/performance factors              |

## Metrics (Prometheus)

**Endpoint**: http://localhost:8059/metrics

### Counters

- `p29_targets_computed_total{symbol}` - Total targets computed per symbol
- `p29_shadow_pass_total` - Total shadow mode passes (no override)
- `p29_enforce_overrides_total{symbol}` - Total enforce mode writes
- `p29_stale_fallback_total{data_source}` - Total stale data fallbacks

### Gauges

- `p29_allocation_confidence{symbol}` - Allocation confidence [0, 1]
- `p29_target_usd{symbol}` - Computed target in USD
- `p29_regime_factor` - Current regime multiplier
- `p29_cluster_factor{cluster}` - Current cluster multiplier
- `p29_drawdown_factor` - Current drawdown multiplier

### Histograms

- `p29_loop_duration_seconds` - Allocation loop duration

## Configuration

**File**: `/etc/quantum/capital-allocation.env`

```bash
# Mode
P29_MODE=shadow                # shadow | enforce

# Timing
P29_INTERVAL_SEC=5             # Loop interval
P29_METRICS_PORT=8059          # Metrics port

# Safety caps
MAX_SYMBOL_PCT=0.40            # 40% max per symbol
MAX_CLUSTER_PCT=0.60           # 60% max per cluster

# Stale threshold
STALE_DATA_SEC=60              # Stale data threshold

# Regime factors
REGIME_FACTOR_TREND=1.2        # TREND multiplier
REGIME_FACTOR_MR=1.0           # MEAN_REVERSION multiplier
REGIME_FACTOR_CHOP=0.6         # CHOP multiplier

# Drawdown factors
DD_LOW_THRESHOLD=0.05          # 5% DD threshold
DD_HIGH_THRESHOLD=0.15         # 15% DD threshold
DD_FACTOR_LOW=1.0              # LOW zone multiplier
DD_FACTOR_MID=0.7              # MID zone multiplier
DD_FACTOR_HIGH=0.4             # HIGH zone multiplier

# Cluster stress
CLUSTER_STRESS_MIN=0.3         # Min cluster factor
CLUSTER_STRESS_MAX=1.0         # Max cluster factor

# Target TTL
TARGET_TTL_SEC=300             # Target hash expiry
```

## Deployment

### Install

```bash
# Copy files
scp microservices/capital_allocation/main.py root@VPS:/home/qt/quantum_trader/microservices/capital_allocation/
scp deploy/quantum-capital-allocation.env root@VPS:/etc/quantum/capital-allocation.env
scp deploy/systemd/quantum-capital-allocation.service root@VPS:/etc/systemd/system/

# Set permissions
ssh root@VPS 'chown -R qt:qt /home/qt/quantum_trader/microservices/capital_allocation'
ssh root@VPS 'chmod 644 /etc/quantum/capital-allocation.env'

# Enable service
ssh root@VPS 'systemctl daemon-reload'
ssh root@VPS 'systemctl enable quantum-capital-allocation'
ssh root@VPS 'systemctl start quantum-capital-allocation'
```

### Verify

```bash
# Service status
systemctl status quantum-capital-allocation

# Logs
journalctl -u quantum-capital-allocation -f

# Metrics
curl http://localhost:8059/metrics | grep p29_

# Redis outputs
redis-cli KEYS "quantum:allocation:target:*"
redis-cli XLEN quantum:stream:allocation.decision
redis-cli XREVRANGE quantum:stream:allocation.decision + - COUNT 1
```

### Proof Script

```bash
bash scripts/proof_p29_allocation.sh
```

**Expected**: Exit 0, SUMMARY: PASS

## Dependencies

- **Upstream**: 
  - Portfolio State Publisher (PSP) - quantum:state:portfolio
  - Portfolio Risk Governor (P2.8) - quantum:portfolio:budget:{symbol}
  - Cluster Learning Module (CLM) - quantum:cluster:stress:{cluster}
  - Regime Detector - quantum:stream:regime.state

- **Downstream**:
  - Governor (reads quantum:allocation:target:{symbol} in enforce mode)

## Fail-Safe Design

1. **Missing portfolio state** → skip cycle, no targets written
2. **Stale portfolio state (>60s)** → skip cycle
3. **Missing budget data** → skip symbol
4. **Stale budget data (>60s)** → use base budget, log fallback
5. **Missing regime** → default factor 1.0, log fallback
6. **Missing cluster stress** → default factor 1.0, log fallback
7. **Stale regime/cluster (>60s)** → default factors, log fallback
8. **Shadow mode** → compute + log, never write targets
9. **Enforce mode** → write targets, Governor reads

**Philosophy**: Always fail-open. Missing/stale data → use conservative defaults, never block execution.

## Integration with Governor

In **enforce mode**, Governor checks allocation targets at Gate 0:

```python
# Governor pseudocode
allocation_key = f"quantum:allocation:target:{symbol}"
allocation = redis.hgetall(allocation_key)

if allocation:
    target_usd = float(allocation["target_usd"])
    # Use target_usd instead of P2.8 budget
else:
    # Fallback to P2.8 budget
    budget = get_p28_budget(symbol)
    target_usd = budget["budget_usd"]
```

## Monitoring

### Health Checks

```bash
# Service active
systemctl is-active quantum-capital-allocation

# Metrics responding
curl -f http://localhost:8059/metrics > /dev/null

# Recent cycles
journalctl -u quantum-capital-allocation --since "1 minute ago" | grep "Allocation cycle complete"
```

### Alert Thresholds

- **p29_stale_fallback_total** > 10/min → investigate data freshness
- **p29_loop_duration_seconds** > 2s → performance degradation
- Service restart > 3x/hour → investigate crashes
- Zero p29_targets_computed_total for 5 min → no symbols processed

## OPS Procedures

### Shadow → Enforce Activation

1. Monitor shadow mode for 24-48h
2. Verify:
   - p29_targets_computed_total incrementing
   - p29_stale_fallback_total low (<1%)
   - Allocation confidence > 0.5
   - No errors in logs
3. Enable enforce mode:
   ```bash
   sed -i 's/P29_MODE=shadow/P29_MODE=enforce/' /etc/quantum/capital-allocation.env
   systemctl restart quantum-capital-allocation
   ```
4. Monitor Governor for allocation target usage
5. Record in OPS ledger via ops_ledger_append.py

### Emergency Rollback

```bash
# Revert to shadow
sed -i 's/P29_MODE=enforce/P29_MODE=shadow/' /etc/quantum/capital-allocation.env
systemctl restart quantum-capital-allocation

# Confirm targets not written
redis-cli KEYS "quantum:allocation:target:*" | wc -l  # Should be 0 after TTL

# Record in OPS ledger
```

## Development Notes

### Cluster Mapping

Currently, cluster_id assignment requires symbol→cluster mapping. Implement via:

1. **Static config**: JSON file mapping symbols to clusters
2. **Redis hash**: quantum:config:symbol.cluster
3. **CLM integration**: Auto-discover clusters from CLM streams

### Regime Integration

Regime state read from `quantum:stream:regime.state`. Ensure regime detector is running and publishing entries.

### Performance Metrics

Alpha values from cluster performance streams. If unavailable, performance_factor defaults to 1.0 (neutral).

## Testing

### Unit Tests

```bash
# TODO: Add pytest suite
pytest tests/test_capital_allocation.py
```

### Integration Tests

Run proof script:
```bash
bash scripts/proof_p29_allocation.sh
```

### Load Testing

Monitor loop duration under load:
```bash
watch -n1 'curl -s localhost:8059/metrics | grep p29_loop_duration_seconds'
```

## Version History

| Version | Date       | Changes                          |
|---------|------------|----------------------------------|
| 1.0.0   | 2026-01-28 | Initial production deployment    |

## References

- [P2.8 Portfolio Risk Governor](AI_P28_BUDGET_ENGINE_FINAL.md)
- [Portfolio State Publisher](AI_PSP_DEPLOYMENT_REPORT.md)
- [Emergency Stop System](AI_ESS_FINAL_REPORT.md)
- [Governor Architecture](AI_GOVERNOR_ARCHITECTURE.md)

---

**End of Documentation**
