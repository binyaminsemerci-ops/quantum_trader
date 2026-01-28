# P3 Apply Layer - Quick Deployment Guide

## Overview
Deploy P3 Apply Layer in **3 commands** on VPS.

**Modes**:
- **P3.0 (dry_run)**: Plans created, NO execution (safe to deploy immediately)
- **P3.1 (testnet)**: Plans executed against Binance testnet (requires credentials)

---

## Deployment Steps

### 1. Deploy Service (dry_run mode - SAFE)

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Pull latest code
cd /root/quantum_trader
git pull origin main

# Deploy (idempotent, safe to re-run)
sudo bash ops/p3_deploy.sh
```

**What it does**:
- Syncs code to `/home/qt/quantum_trader`
- Installs config to `/etc/quantum/apply-layer.env` (APPLY_MODE=dry_run)
- Installs systemd unit
- Enables and starts service
- Waits 10s and shows initial status

**Expected output**:
```
=== P3 Apply Layer Deployment ===
✅ Repo synced
✅ Config installed: /etc/quantum/apply-layer.env
✅ Systemd unit installed
✅ Service active
  apply.plan: 3 entries
  apply.result: 3 entries
✅ Prometheus metrics available on :8043
✅ P3 Apply Layer deployed successfully
```

---

### 2. Verify dry_run Mode

```bash
# Run proof pack script
bash /home/qt/quantum_trader/ops/p3_proof_dry_run.sh
```

**Checks**:
- ✅ Service active
- ✅ APPLY_MODE=dry_run
- ✅ Plans created in `quantum:stream:apply.plan`
- ✅ Results in `quantum:stream:apply.result`
- ✅ Idempotency working (dedupe keys)
- ✅ NO Binance calls (executed=false in all results)
- ✅ Allowlist working (BTCUSDT only)
- ✅ Prometheus metrics on :8043

**Expected**: All ✅ green checks

---

### 3. Monitor Operation

```bash
# Watch logs
journalctl -u quantum-apply-layer -f

# Check plans
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3

# Check results
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3

# Check metrics
curl http://localhost:8043/metrics | grep quantum_apply

# Service status
systemctl status quantum-apply-layer
```

---

## Enable Testnet Mode (P3.1) - Optional

⚠️ **Only after dry_run verified!**

### Prerequisites
1. Binance testnet account: https://testnet.binancefuture.com/
2. API key + secret generated
3. Dry_run mode verified operational

### Steps

```bash
# 1. Add Binance credentials
sudo nano /etc/quantum/apply-layer.env

# Add these lines:
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_API_SECRET=your_testnet_secret_here

# 2. Change mode to testnet
sudo sed -i 's/APPLY_MODE=dry_run/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env

# 3. Restart service
sudo systemctl restart quantum-apply-layer

# 4. Verify mode changed
grep APPLY_MODE /etc/quantum/apply-layer.env

# 5. Wait 30s for first execution
sleep 30

# 6. Run testnet proof pack
bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh
```

**Expected**:
- ✅ APPLY_MODE=testnet
- ✅ Binance credentials configured
- ✅ Results with executed=true
- ✅ Order IDs in results
- ✅ Only BTCUSDT executed (allowlist)

---

## Emergency Stop

If anything goes wrong:

```bash
# Method 1: Enable kill switch (blocks all executions)
sudo sed -i 's/APPLY_KILL_SWITCH=false/APPLY_KILL_SWITCH=true/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer

# Method 2: Stop service completely
sudo systemctl stop quantum-apply-layer

# Method 3: Switch back to dry_run
sudo sed -i 's/APPLY_MODE=testnet/APPLY_MODE=dry_run/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer
```

---

## Configuration Tuning

Edit `/etc/quantum/apply-layer.env`:

```bash
# Expand allowlist (after BTCUSDT tested successfully)
APPLY_ALLOWLIST=BTCUSDT,ETHUSDT

# Adjust kill score thresholds
K_BLOCK_CRITICAL=0.85  # More aggressive (default 0.80)
K_BLOCK_WARNING=0.65   # More aggressive (default 0.60)

# Change poll interval
APPLY_POLL_SEC=10      # Slower polling (default 5s)

# Adjust dedupe TTL
APPLY_DEDUPE_TTL_SEC=7200  # 2 hours (default 6h)
```

Then restart:
```bash
sudo systemctl restart quantum-apply-layer
```

---

## Troubleshooting

### Service not starting
```bash
# Check logs
journalctl -u quantum-apply-layer -n 50

# Check Python path
which python3

# Check dependencies
python3 -c "import redis, prometheus_client; print('OK')"
```

### No plans created
```bash
# Check harvest proposals exist
redis-cli HGETALL quantum:harvest:proposal:BTCUSDT

# Check service logs
journalctl -u quantum-apply-layer -f

# Verify symbols config
grep SYMBOLS /etc/quantum/apply-layer.env
```

### Plans created but all SKIP
```bash
# Check allowlist
grep ALLOWLIST /etc/quantum/apply-layer.env

# Check kill switch
grep KILL_SWITCH /etc/quantum/apply-layer.env

# Review reason codes in plans
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 | grep reason_codes
```

### Testnet execution fails
```bash
# Verify credentials set
grep BINANCE_TESTNET /etc/quantum/apply-layer.env | grep -v "^#"

# Check Binance testnet status
curl https://testnet.binancefuture.com/fapi/v1/ping

# Review error in results
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 1
```

---

## Integration with Prometheus (Optional)

Add to `/etc/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'quantum-apply-layer'
    static_configs:
      - targets: ['localhost:8043']
        labels:
          service: 'apply-layer'
          component: 'quantum-trader'
    scrape_interval: 10s
```

Then reload:
```bash
curl -X POST http://localhost:9091/-/reload
```

Verify:
```bash
# Check target in Prometheus UI
# http://46.224.116.254:9091/targets

# Query metrics
curl 'http://localhost:9091/api/v1/query?query=quantum_apply_plan_total'
```

---

## Summary

| Step | Command | Expected |
|------|---------|----------|
| Deploy | `sudo bash ops/p3_deploy.sh` | Service active, dry_run mode |
| Verify | `bash ops/p3_proof_dry_run.sh` | All ✅ checks pass |
| Monitor | `journalctl -u quantum-apply-layer -f` | Plans/results in logs |
| Testnet | Edit env + restart | Executions with order IDs |

**Safe defaults**:
- Mode: dry_run (no execution)
- Allowlist: BTCUSDT only
- Kill switch: OFF
- K thresholds: 0.80 (critical), 0.60 (warning)

**Next steps**:
1. Deploy dry_run mode → verify → monitor 24h
2. Enable testnet mode → verify → monitor with small positions
3. Add ETHUSDT/SOLUSDT to allowlist (one at a time)
4. Implement P3.2 (governor limits, position tracking)
