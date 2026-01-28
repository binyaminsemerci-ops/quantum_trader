# P3 Apply Layer - Proof Pack Template

**Date**: [YYYY-MM-DD HH:MM:SS UTC]  
**Operator**: [Name/ID]  
**Mode**: [dry_run | testnet]  
**VPS**: [hostname/IP]

---

## PROOF 1: Service Health

**Objective**: Verify apply layer service is active and operational.

### Commands
```bash
systemctl status quantum-apply-layer
journalctl -u quantum-apply-layer --since "5 minutes ago" | tail -20
```

### Results
```
[Paste output here]
```

### Assessment
- [ ] Service active
- [ ] No errors in recent logs
- [ ] Service restarted successfully after reload

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 2: Configuration Verification

**Objective**: Verify correct mode and safety gates configured.

### Commands
```bash
cat /etc/quantum/apply-layer.env | grep -E "^(APPLY_MODE|APPLY_ALLOWLIST|K_BLOCK|APPLY_KILL_SWITCH)"
```

### Results
```
APPLY_MODE=[mode]
APPLY_ALLOWLIST=[symbols]
K_BLOCK_CRITICAL=[threshold]
K_BLOCK_WARNING=[threshold]
APPLY_KILL_SWITCH=[true/false]
```

### Assessment
- [ ] Mode matches expectation
- [ ] Allowlist conservative (BTCUSDT only for initial deployment)
- [ ] Kill score thresholds reasonable (0.80/0.60)
- [ ] Kill switch OFF (unless emergency)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 3: Apply Plans Created

**Objective**: Verify plans are created and published to Redis stream.

### Commands
```bash
redis-cli XLEN quantum:stream:apply.plan
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3
```

### Results
```
Stream length: [N] entries

[Paste plan entries here]
```

### Assessment
- [ ] Stream populated (>0 entries)
- [ ] Plans include all required fields (plan_id, symbol, decision, steps)
- [ ] Decision values appropriate (EXECUTE/SKIP/BLOCKED)
- [ ] Reason codes explain decisions

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 4: Apply Results Published

**Objective**: Verify execution results published to Redis stream.

### Commands
```bash
redis-cli XLEN quantum:stream:apply.result
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3
```

### Results
```
Stream length: [N] entries

[Paste result entries here]
```

### Assessment
- [ ] Stream populated (>0 entries if EXECUTE decisions exist)
- [ ] Results include executed/would_execute flags
- [ ] Steps results populated
- [ ] Order IDs present (testnet mode only)
- [ ] No unexpected errors

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 5: Idempotency Working

**Objective**: Verify dedupe keys prevent duplicate executions.

### Commands
```bash
redis-cli KEYS "quantum:apply:dedupe:*" | wc -l
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep -c "duplicate_plan"
curl -s http://localhost:8043/metrics | grep dedupe_hits
```

### Results
```
Dedupe keys: [N]
Duplicate plans: [N]
Dedupe hits metric: [N]
```

### Assessment
- [ ] Dedupe keys created for EXECUTE plans
- [ ] Duplicate plan detections logged (if proposals unchanged)
- [ ] TTL set on dedupe keys (6h default)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 6: Allowlist Enforcement

**Objective**: Verify only allowlisted symbols can execute.

### Commands
```bash
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 20 | grep -E "(symbol|decision|reason_codes)"
```

### Results
```
[Paste filtered plan entries showing symbol, decision, reason_codes]
```

### Assessment
- [ ] BTCUSDT marked EXECUTE (if passes other gates)
- [ ] Non-allowlisted symbols marked SKIP with "not_in_allowlist"
- [ ] No executions for non-allowlisted symbols

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 7: Kill Score Safety Gates

**Objective**: Verify kill score thresholds block actions correctly.

### Commands
```bash
# Find plans with high kill scores
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 | grep -B3 -A3 "kill_score"
```

### Results
```
[Paste examples showing kill_score and corresponding decisions]

Example high K:
- symbol: [SYMBOL]
- kill_score: [K]
- decision: [EXECUTE/BLOCKED]
- reason_codes: [codes]
```

### Assessment
- [ ] K >= 0.80 → BLOCKED (all actions)
- [ ] K >= 0.60 → BLOCKED (non-close actions) OR EXECUTE (close actions)
- [ ] K < 0.60 → EXECUTE (subject to other gates)

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 8: Mode-Specific Verification

### For dry_run mode:

**Objective**: Verify NO actual execution occurs.

### Commands
```bash
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | grep -E "(executed|would_execute)"
journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -i binance
```

### Results
```
[Check all results have executed=false and would_execute=true]
[Check NO Binance API calls in logs]
```

### Assessment
- [ ] All results: executed=false
- [ ] All results: would_execute=true
- [ ] No Binance logs

**Status**: ✅ PASS / ❌ FAIL

---

### For testnet mode:

**Objective**: Verify actual execution against Binance testnet.

### Commands
```bash
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5
journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -i testnet
```

### Results
```
[Paste results showing executed=true and order_ids]
[Paste testnet log entries]
```

### Assessment
- [ ] Results with executed=true
- [ ] Order IDs present in results
- [ ] TESTNET logs visible
- [ ] No Binance API errors
- [ ] Only allowlisted symbols executed

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 9: Prometheus Metrics

**Objective**: Verify metrics exposed and accurate.

### Commands
```bash
curl -s http://localhost:8043/metrics | grep quantum_apply
```

### Results
```
[Paste metrics output]

quantum_apply_plan_total{symbol="BTCUSDT",decision="EXECUTE"} [N]
quantum_apply_plan_total{symbol="ETHUSDT",decision="SKIP"} [N]
quantum_apply_dedupe_hits_total [N]
quantum_apply_execute_total{symbol="BTCUSDT",step="CLOSE_PARTIAL_75",status="success"} [N]
quantum_apply_last_success_epoch{symbol="BTCUSDT"} [timestamp]
```

### Assessment
- [ ] Metrics endpoint accessible
- [ ] Plan metrics present
- [ ] Execute metrics present (if testnet)
- [ ] Dedupe hits tracked
- [ ] Last success timestamp updating

**Status**: ✅ PASS / ❌ FAIL

---

## PROOF 10: Integration with P0-P2

**Objective**: Verify no breaking changes to existing pipeline.

### Commands
```bash
# Check existing services still running
systemctl is-active quantum-harvest-proposal
systemctl is-active quantum-harvest-metrics-exporter

# Check harvest proposals still being published
redis-cli HGETALL quantum:harvest:proposal:BTCUSDT | grep -E "(harvest_action|kill_score|last_update_epoch)"
```

### Results
```
[Confirm harvest proposal service still active]
[Confirm harvest proposals still published]
```

### Assessment
- [ ] P2 harvest proposal service active
- [ ] P2 harvest metrics exporter active
- [ ] Harvest proposals still published (fresh timestamps)
- [ ] Apply layer reads but does NOT modify harvest proposals

**Status**: ✅ PASS / ❌ FAIL

---

## OVERALL ASSESSMENT

**Total Proofs**: 10  
**Passed**: [ ] / 10  
**Failed**: [ ] / 10

### Summary
[Brief summary of proof pack results]

### Issues Found
[List any issues or warnings]

### Recommendations
[Next steps or improvements]

### Sign-Off

**Operator**: [Name]  
**Date**: [YYYY-MM-DD HH:MM:SS UTC]  
**Status**: ✅ APPROVED / ⚠️ APPROVED WITH WARNINGS / ❌ FAILED

---

## Appendix: Commands Reference

```bash
# Deploy
sudo bash /root/quantum_trader/ops/p3_deploy.sh

# Run proof pack (dry_run)
bash /home/qt/quantum_trader/ops/p3_proof_dry_run.sh

# Run proof pack (testnet)
bash /home/qt/quantum_trader/ops/p3_proof_testnet.sh

# Enable testnet mode
sudo sed -i 's/APPLY_MODE=dry_run/APPLY_MODE=testnet/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer

# Emergency stop
sudo sed -i 's/APPLY_KILL_SWITCH=false/APPLY_KILL_SWITCH=true/' /etc/quantum/apply-layer.env
sudo systemctl restart quantum-apply-layer

# Monitor
journalctl -u quantum-apply-layer -f
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 5
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5
curl http://localhost:8043/metrics | grep quantum_apply
```
