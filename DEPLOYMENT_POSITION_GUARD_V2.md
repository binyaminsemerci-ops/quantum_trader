# Apply Layer Hardening - Deployment Guide

**BUILD_TAG:** apply-layer-position-guard-v2 (Principal Engineer Edition)

## Changes

### 1. Robust has_position() (SHORT Position Support)
- Handles SHORT positions: `abs(position_amt) > 1e-12`
- Handles empty string, missing field: treated as `0.0`
- Better error handling: ValueError, TypeError caught separately

### 2. Dedupe Bypass Dev Flag
- `APPLY_DEDUPE_BYPASS=false` (default): 5 min TTL (production)
- `APPLY_DEDUPE_BYPASS=true` (dev/QA): 5 sec TTL (rapid testing)
- No breaking changes to production behavior

### 3. Field-Aware Stream Proof Script
- `proof_position_guard_stream_parse.sh`: Robust awk parsing
- Handles Redis XREVRANGE field/value pairs correctly
- No false negatives from grep line splitting

## Deployment Steps

### Step 1: Deploy Updated Apply Layer
```bash
# From Windows (WSL)
wsl bash -c "scp -i ~/.ssh/hetzner_fresh microservices/apply_layer/main.py root@46.224.116.254:/home/qt/quantum_trader/microservices/apply_layer/main.py"
```

### Step 2: Clear Python Cache
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "find /home/qt/quantum_trader/microservices/apply_layer -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"
```

### Step 3: Restart Service
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "systemctl restart quantum-apply-layer && sleep 3 && journalctl -u quantum-apply-layer -n 20 --no-pager"
```

Expected log output:
```
Dedupe: TTL=300s (bypass=False)
ApplyLayer initialized
```

### Step 4: Deploy Proof Script
```bash
wsl bash -c "scp -i ~/.ssh/hetzner_fresh scripts/proof_position_guard_stream_parse.sh root@46.224.116.254:/home/qt/scripts/"
```

### Step 5: Run Stream Proof
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "bash /home/qt/scripts/proof_position_guard_stream_parse.sh"
```

Expected output:
```
BTCUSDT: X total plans, Y with position guard
ETHUSDT: X total plans, Y with position guard
✓ TEST 1: Position guard active in stream
✓ TEST 2: Three-layer contract chain verified
```

### Step 6: Verify Position Snapshots
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "redis-cli HGETALL quantum:position:snapshot:BTCUSDT && echo '' && redis-cli HGETALL quantum:position:snapshot:ZECUSDT"
```

Expected:
- BTCUSDT: `position_amt: 0.0` or `0` (no position)
- ZECUSDT: `position_amt: -0.353` or similar (has position, SHORT)

### Step 7: Verify Guard Logs
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "journalctl -u quantum-apply-layer --since '5 minutes ago' --no-pager | grep -E 'UPDATE_SL_SKIP_NO_POSITION|BTCUSDT|ETHUSDT' | tail -20"
```

Expected:
```
UPDATE_SL_SKIP_NO_POSITION BTCUSDT: proposed_sl=100.20 (no_position)
UPDATE_SL_SKIP_NO_POSITION ETHUSDT: proposed_sl=50.10 (no_position)
```

## Dev Mode Testing (Optional)

To enable rapid iteration with 5s dedupe TTL:

### Enable Bypass
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "echo 'APPLY_DEDUPE_BYPASS=true' >> /etc/systemd/system/quantum-apply-layer.service.d/override.conf && systemctl daemon-reload && systemctl restart quantum-apply-layer"
```

### Verify Bypass Active
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "journalctl -u quantum-apply-layer -n 30 --no-pager | grep 'Dedupe:'"
```

Expected:
```
Dedupe: TTL=5s (bypass=True)
```

### Disable Bypass (Return to Production)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "sed -i '/APPLY_DEDUPE_BYPASS/d' /etc/systemd/system/quantum-apply-layer.service.d/override.conf && systemctl daemon-reload && systemctl restart quantum-apply-layer"
```

## Verification Checklist

- [ ] Service restarted successfully
- [ ] Log shows `Dedupe: TTL=300s (bypass=False)`
- [ ] Stream proof shows position guard events
- [ ] BTC/ETH (no position): UPDATE_SL skipped
- [ ] ZEC/FIL (has position): UPDATE_SL allowed
- [ ] Three-layer reason codes: `kill_score_open_ok,update_sl_no_position_skip,action_hold`
- [ ] No errors in logs
- [ ] Position snapshots accessible

## Rollback (if needed)

```bash
# Restore previous version from git
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "cd /home/qt/quantum_trader && git checkout HEAD~1 microservices/apply_layer/main.py && systemctl restart quantum-apply-layer"
```

## Key Improvements

1. **SHORT Position Support**: `abs(position_amt) > 1e-12` catches both LONG and SHORT
2. **Better Error Handling**: ValueError/TypeError separated for clearer debugging
3. **Dev Mode**: `APPLY_DEDUPE_BYPASS=true` enables 5s TTL for rapid testing (no production impact)
4. **Robust Stream Parsing**: Field-aware awk parsing eliminates false negatives

## Production Impact

- **Zero Breaking Changes**: All defaults unchanged
- **Fail-Soft**: Missing/invalid position data → assume no position (conservative)
- **Observable**: All guard decisions logged with structured format
- **Testable**: Dev bypass flag allows rapid iteration without affecting production
