# P3 VPS Verification Report

**Date**: [AUTO-GENERATED]  
**Operator**: [Name]  
**VPS**: 46.224.116.254  
**Mode**: P3.0 dry_run → P3.1 testnet (phased)

---

## Executive Summary

This report documents verification of P3 Apply Layer deployment on VPS with:
- ✅ Real Binance Futures testnet adapter (NOT placeholder)
- ✅ Path consistency (/home/qt/quantum_trader for systemd)
- ✅ Idempotency via Redis dedupe keys
- ✅ Safety gates (allowlist, kill_score, kill switch)
- ✅ Full audit trail (Redis streams)

**Status**: [PASS / FAIL / IN PROGRESS]

---

## PROOF 1: Service Status

### Objective
Verify quantum-apply-layer service running under systemd.

### Commands
```bash
systemctl status quantum-apply-layer
systemctl cat quantum-apply-layer | grep WorkingDirectory
```

### Results
```
[Paste systemctl status output]

WorkingDirectory=/home/qt/quantum_trader
```

### Assessment
- [ ] Service ACTIVE
- [ ] WorkingDirectory = /home/qt/quantum_trader (CORRECT)
- [ ] No restart loops

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 2: Path Consistency

### Objective
Confirm repo synced to /home/qt/quantum_trader (systemd WorkingDirectory).

### Commands
```bash
cd /root/quantum_trader && git rev-parse HEAD
cd /home/qt/quantum_trader && git rev-parse HEAD
diff -q /root/quantum_trader/microservices/apply_layer/main.py /home/qt/quantum_trader/microservices/apply_layer/main.py
```

### Results
```
/root commit: [hash]
/home/qt commit: [hash]

Match: YES / NO
```

### Assessment
- [ ] Commits match
- [ ] apply_layer/main.py synced
- [ ] systemd uses /home/qt path

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 3: Dry Run Mode Active

### Objective
Verify APPLY_MODE=dry_run and NO execution happening.

### Commands
```bash
grep APPLY_MODE /etc/quantum/apply-layer.env
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep executed
journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -i binance
```

### Results
```
APPLY_MODE=dry_run

All results: executed=false
Binance logs: 0 entries
```

### Assessment
- [ ] APPLY_MODE=dry_run
- [ ] All results have executed=false
- [ ] No Binance API calls in logs

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 4: Redis Streams Populated

### Objective
Verify apply.plan and apply.result streams receiving entries.

### Commands
```bash
redis-cli XLEN quantum:stream:apply.plan
redis-cli XLEN quantum:stream:apply.result
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3
```

### Results
```
apply.plan: [N] entries
apply.result: [N] entries

[Paste 3 recent plans showing plan_id, symbol, decision, steps]
```

### Assessment
- [ ] apply.plan stream populated
- [ ] apply.result stream populated
- [ ] Plans include required fields (plan_id, decision, steps)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 5: Idempotency Working

### Objective
Verify dedupe keys prevent duplicate execution.

### Commands
```bash
redis-cli KEYS "quantum:apply:dedupe:*" | wc -l
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep duplicate_plan
curl -s http://localhost:8043/metrics | grep dedupe_hits
```

### Results
```
Dedupe keys: [N]
Duplicate plans: [N]
Metric: quantum_apply_dedupe_hits_total [N]
```

### Assessment
- [ ] Dedupe keys created for EXECUTE plans
- [ ] TTL set (~6h)
- [ ] Duplicate detections logged (if proposals unchanged)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 6: Allowlist Enforcement

### Objective
Verify only BTCUSDT can execute (default allowlist).

### Commands
```bash
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep -E "(symbol|decision|reason_codes)"
```

### Results
```
BTCUSDT: decision=EXECUTE (or SKIP if blocked by other gates)
ETHUSDT: decision=SKIP, reason_codes=not_in_allowlist
SOLUSDT: decision=SKIP, reason_codes=not_in_allowlist
```

### Assessment
- [ ] BTCUSDT marked EXECUTE (if passes other gates)
- [ ] Non-allowlisted symbols SKIP with not_in_allowlist
- [ ] No executions for non-allowlisted symbols

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 7: Kill Score Safety Gates

### Objective
Verify kill_score thresholds block actions correctly.

### Commands
```bash
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 | grep -B3 -A3 "kill_score"
```

### Results
```
Example 1:
- symbol: BTCUSDT
- kill_score: 0.527
- decision: EXECUTE
- reason_codes: []

Example 2:
- symbol: ETHUSDT  
- kill_score: 0.823
- decision: BLOCKED
- reason_codes: kill_score_critical
```

### Assessment
- [ ] K >= 0.80 → BLOCKED (all actions)
- [ ] K >= 0.60 → BLOCKED (non-close) OR EXECUTE (close actions)
- [ ] K < 0.60 → EXECUTE (subject to other gates)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 8: Real Binance Code (NOT Placeholder)

### Objective
Confirm execute_testnet uses REAL Binance API (not simulated_success).

### Commands
```bash
grep -A20 "class BinanceTestnetClient" /home/qt/quantum_trader/microservices/apply_layer/main.py | head -25
grep "simulated_success" /home/qt/quantum_trader/microservices/apply_layer/main.py || echo "No placeholder found (GOOD)"
```

### Results
```
class BinanceTestnetClient:
    """Minimal Binance Futures Testnet client for reduceOnly orders"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com"
    ...

No placeholder found (GOOD)
```

### Assessment
- [ ] BinanceTestnetClient class exists
- [ ] Real API methods: ping(), get_position(), place_market_order()
- [ ] NO "simulated_success" placeholder
- [ ] reduceOnly flag implemented

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 9: Prometheus Metrics

### Objective
Verify metrics exposed and accurate.

### Commands
```bash
curl -s http://localhost:8043/metrics | grep quantum_apply
```

### Results
```
quantum_apply_plan_total{symbol="BTCUSDT",decision="EXECUTE"} [N]
quantum_apply_plan_total{symbol="ETHUSDT",decision="SKIP"} [N]
quantum_apply_dedupe_hits_total [N]
quantum_apply_last_success_epoch{symbol="BTCUSDT"} [timestamp]
```

### Assessment
- [ ] Metrics endpoint accessible
- [ ] Plan metrics present
- [ ] Dedupe hits tracked
- [ ] No execute_total in dry_run (correct)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 10: Integration with P0-P2

### Objective
Verify no breaking changes to existing pipeline.

### Commands
```bash
systemctl is-active quantum-harvest-proposal
systemctl is-active quantum-harvest-metrics-exporter
redis-cli HGETALL quantum:harvest:proposal:BTCUSDT | head -10
```

### Results
```
quantum-harvest-proposal: active
quantum-harvest-metrics-exporter: active

quantum:harvest:proposal:BTCUSDT:
  harvest_action: PARTIAL_75
  kill_score: 0.527
  last_update_epoch: 1769130500
```

### Assessment
- [ ] P2 services still active
- [ ] Harvest proposals still published
- [ ] Apply layer reads but does NOT modify proposals

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 11: Testnet Execution (P3.1 - After dry_run verified)

⚠️ **Only run after P3.0 dry_run verified for 24h**

### Objective
Verify REAL Binance testnet orders with reduceOnly.

### Commands
```bash
# After enabling testnet mode
grep APPLY_MODE /etc/quantum/apply-layer.env
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3
journalctl -u quantum-apply-layer --since "5 minutes ago" | grep "Order.*executed"
```

### Results
```
APPLY_MODE=testnet

Result 1:
  executed: true
  order_id: 12345678
  reduce_only: true
  side: SELL
  executed_qty: 0.001
```

### Assessment
- [ ] APPLY_MODE=testnet
- [ ] Results with executed=true
- [ ] Real order IDs (not sim_xxx)
- [ ] reduceOnly flag set
- [ ] Position checked before execution

**Status**: ✅ PASS / ❌ FAIL / ⏳ NOT YET (dry_run phase)

---

## Overall Assessment

**Total Proofs**: 11  
**Passed**: [ ] / 11  
**Failed**: [ ] / 11  
**Not Yet Run**: [ ] / 11

### Key Findings

**✅ Working**:
- [List what works]

**⚠️ Warnings**:
- [List warnings]

**❌ Issues**:
- [List issues]

### Recommendations

1. **Immediate**:
   - [Actions needed now]

2. **24h monitoring**:
   - Watch dedupe_hits rate
   - Monitor plan/result stream growth
   - Check for errors

3. **Before testnet**:
   - Add Binance credentials
   - Test connectivity manually
   - Verify small position size

### Sign-Off

**Operator**: [Name]  
**Date**: [YYYY-MM-DD HH:MM UTC]  
**Status**: ✅ APPROVED FOR DRY_RUN / ✅ APPROVED FOR TESTNET / ⚠️ WARNINGS / ❌ ISSUES

---

## Appendix: Quick Commands

```bash
# Deploy
sudo bash /root/quantum_trader/ops/p3_deploy.sh

# Verify
bash /root/quantum_trader/ops/p3_vps_verify_and_patch.sh

# Proof pack (dry_run)
bash /home/qt/quantum_trader/ops/p3_proof_dry_run.sh

# Monitor
journalctl -u quantum-apply-layer -f
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 5
curl http://localhost:8043/metrics | grep quantum_apply

# Enable testnet (after 24h dry_run)
sudo nano /etc/quantum/apply-layer.env
# Add BINANCE_TESTNET_API_KEY and SECRET
# Set APPLY_MODE=testnet
sudo systemctl restart quantum-apply-layer

# Proof pack (testnet)
bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh

# Emergency stop
sudo sed -i 's/APPLY_KILL_SWITCH=false/APPLY_KILL_SWITCH=true/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer
```

---

## Change Log

| Date | Change | Operator |
|------|--------|----------|
| 2026-01-23 | Initial P3.0 dry_run deployment | [Name] |
| [Date] | Verified dry_run 24h | [Name] |
| [Date] | Enabled P3.1 testnet | [Name] |
