# 🔒 Production Hygiene Guide - P3 Apply Layer

**Last Updated:** 2026-01-25  
**Status:** Live on VPS (production-ready)

---

## 1️⃣ Hard Mode Switch

### What It Does
Toggles between **TESTNET (bypass permits)** and **PRODUCTION (require permits)** execution modes.

### Configuration
```bash
# TESTNET MODE - Governor bypass (development only)
export TESTNET=true
systemctl restart quantum-apply-layer

# PRODUCTION MODE - Require BOTH permits (default)
export TESTNET=false
systemctl restart quantum-apply-layer
```

### Why It Matters
- **TESTNET=true**: Skips all permit checks for testing/development
- **TESTNET=false**: Enforces both Governor (P3.2) + P3.3 permits before execution

### Current Mode
```bash
# Check current mode
journalctl -u quantum-apply-layer -n 1 --no-pager | grep -E "TESTNET|PRODUCTION"

# Expected output in PRODUCTION mode:
# ✅ PRODUCTION MODE - Both permits required (Governor + P3.3)
```

### Recommended Setup
```bash
# Apply in environment config
cat >> /etc/quantum/apply-layer.env << 'EOF'
TESTNET=false
EOF

systemctl restart quantum-apply-layer
journalctl -u quantum-apply-layer -n 1 | grep PRODUCTION
```

---

## 2️⃣ Safety Kill Switch

### What It Does
**Emergency stop** for all execution across the entire system. One Redis key to block everything.

### Activation (Emergency Only)
```bash
# ACTIVATE KILL SWITCH (STOP ALL EXECUTION)
redis-cli SET quantum:global:kill_switch true

# Verify it's active
redis-cli GET quantum:global:kill_switch
# Output: "true"
```

### Deactivation (Resume Normal)
```bash
# DEACTIVATE KILL SWITCH (RESUME NORMAL)
redis-cli SET quantum:global:kill_switch false

# Verify it's off
redis-cli GET quantum:global:kill_switch
# Output: "false"
```

### What Happens When Active
```
Apply Layer receives EXECUTE plan
  ↓
execute_testnet() invoked
  ↓
[KILL_SWITCH] Check: true
  ↓
[KILL_SWITCH] Execution halted - kill switch is ACTIVE
  ↓
Return error: "kill_switch_active"
  ↓
Order NOT executed
```

### Monitoring Kill Switch Status
```bash
# Check if kill switch is active
redis-cli GET quantum:global:kill_switch || echo "Not set (disabled)"

# Monitor for activations
redis-cli monitor | grep "global:kill_switch"
```

### When to Use
- **System malfunction detected** → Activate to stop all trades
- **Market anomaly** → Activate, investigate, then resume
- **Infrastructure issues** → Activate to prevent cascading failures
- **Maintenance required** → Activate before deploying changes

### Example: Emergency Scenario
```bash
# 1. Detect problem
journalctl -u quantum-apply-layer -f | grep ERROR

# 2. Activate kill switch
redis-cli SET quantum:global:kill_switch true
echo "Kill switch activated at $(date -u)"

# 3. Investigate
redis-cli HGETALL quantum:position:BTCUSDT | head -20

# 4. Fix issue
# ... apply fixes ...

# 5. Deactivate and resume
redis-cli SET quantum:global:kill_switch false
echo "Resume at $(date -u)"

# 6. Monitor recovery
journalctl -u quantum-apply-layer -f | head -50
```

---

## 3️⃣ Prometheus Metrics

### Metrics Available
```
Permit Metrics:
- p33_permit_deny_total{reason}     [counter] P3.3 denies (by reason)
- p33_permit_allow_total{}          [counter] P3.3 allows

Governor Metrics:
- governor_block_total{reason}      [counter] Governor blocks (by reason)

Execution Metrics:
- apply_executed_total{status}      [counter] Total executions (status: success|kill_switch|testnet_bypass)
- apply_plan_processed_total{decision} [counter] Plans processed (decision: EXECUTE|HOLD|REDUCE)

Wait-Loop Metrics:
- permit_wait_ms{}                  [gauge] Last permit wait time (ms)

Position Metrics:
- position_mismatch_seconds{}       [gauge] Seconds since last position match
```

### Prometheus Configuration
```yaml
# /etc/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'quantum-apply-layer'
    static_configs:
      - targets: ['localhost:8000']  # Apply Layer metrics port
        labels:
          service: apply_layer
```

### Querying Metrics
```bash
# Query permit denies by reason (last 5 min)
curl -s "http://localhost:9090/api/v1/query?query=rate(p33_permit_deny_total[5m])"

# Query execution success rate (last hour)
curl -s "http://localhost:9090/api/v1/query?query=rate(apply_executed_total{status='success'}[1h])"

# Query average permit wait time (last 15 min)
curl -s "http://localhost:9090/api/v1/query?query=avg(permit_wait_ms)"
```

### Alert Rules (Recommended)
```yaml
# /etc/prometheus/rules/apply-layer.yml
groups:
  - name: apply_layer
    interval: 30s
    rules:
      # Alert if many P3.3 denies
      - alert: P33DenyRateHigh
        expr: rate(p33_permit_deny_total[5m]) > 0.5
        for: 5m
        annotations:
          summary: "P3.3 deny rate high ({{ $value }}/sec)"
      
      # Alert if execution failures spike
      - alert: ExecutionFailureSpike
        expr: rate(apply_executed_total{status='success'}[5m]) < 0.1
        for: 10m
        annotations:
          summary: "Execution success rate dropped ({{ $value }}/sec)"
      
      # Alert if permit wait exceeds threshold
      - alert: PermitWaitTimeout
        expr: permit_wait_ms > 800
        for: 1m
        annotations:
          summary: "Permit wait time critical ({{ $value }}ms)"
```

### Dashboards
```
Recommended Grafana panels:
1. Execution Success Rate (%) - p33_permit_allow_total / apply_plan_processed_total
2. P3.3 Deny Reasons - pie chart of p33_permit_deny_total by reason
3. Permit Wait Time (ms) - gauge of permit_wait_ms
4. Governor Blocks - bar chart of governor_block_total by reason
5. Position Mismatch Duration - gauge of position_mismatch_seconds
```

---

## 4️⃣ Production Checklist

Before going to mainnet:

```
☑️ Hard Mode Switch
   ☐ TESTNET=false in /etc/quantum/apply-layer.env
   ☐ Service restarted with new config
   ☐ Verified in logs: "✅ PRODUCTION MODE"

☑️ Safety Kill Switch
   ☐ Tested activation: redis-cli SET quantum:global:kill_switch true
   ☐ Verified execution blocked: order NOT placed
   ☐ Tested deactivation: redis-cli SET quantum:global:kill_switch false
   ☐ Verified execution resumed: orders placed normally

☑️ Prometheus Metrics
   ☐ prometheus_client installed (pip list | grep prometheus)
   ☐ Metrics endpoint responding: curl http://localhost:8000/metrics
   ☐ Prometheus scrape config updated
   ☐ Dashboard created in Grafana
   ☐ Alert rules deployed

☑️ Monitoring & Alerting
   ☐ journalctl logs streaming to ELK/DataDog
   ☐ Prometheus alerts configured
   ☐ On-call rotation established
   ☐ Runbook for kill switch activation written

☑️ Documentation
   ☐ This guide reviewed and updated
   ☐ Team trained on kill switch
   ☐ Escalation procedures documented
```

---

## 5️⃣ Quick Reference

### Check Current Mode
```bash
journalctl -u quantum-apply-layer -n 1 --no-pager | grep -E "TESTNET|PRODUCTION"
```

### Activate Kill Switch
```bash
redis-cli SET quantum:global:kill_switch true
```

### Deactivate Kill Switch
```bash
redis-cli SET quantum:global:kill_switch false
```

### View Execution Metrics
```bash
journalctl -u quantum-apply-layer -n 20 --no-pager | grep "\[PERMIT_WAIT\]"
```

### Check Permit Denies
```bash
journalctl -u quantum-apply-layer --since "1 hour ago" --no-pager | grep "p33_denied" | wc -l
```

### View Position Reconciliation Status
```bash
redis-cli HGETALL quantum:position:BTCUSDT | grep -E "ledger_amount|exchange_amt"
```

---

## 6️⃣ Troubleshooting

### Problem: Execution blocked with "permit_timeout"
**Cause:** Governor or P3.3 not issuing permits within 1200ms  
**Solution:**
```bash
# Check if Governor is running
systemctl status quantum-governor

# Check if P3.3 is running
systemctl status quantum-p33-position-brain

# Increase wait timeout
echo "APPLY_PERMIT_WAIT_MS=2000" >> /etc/quantum/apply-layer.env
systemctl restart quantum-apply-layer
```

### Problem: P3.3 denying with "reconcile_required_qty_mismatch"
**Cause:** Exchange position ≠ Ledger position  
**Solution:**
```bash
# Check current position
redis-cli HGETALL quantum:position:BTCUSDT

# If correct, update ledger
redis-cli HSET quantum:position:BTCUSDT ledger_amount 0.062

# Verify
redis-cli HGETALL quantum:position:BTCUSDT
```

### Problem: Kill switch stuck on "true"
**Solution:**
```bash
# Force deactivate
redis-cli SET quantum:global:kill_switch false

# Verify
redis-cli GET quantum:global:kill_switch
# Expected: "false"

# Resume service
systemctl restart quantum-apply-layer
```

---

## 7️⃣ Deployment Instructions

### VPS Deployment (Production)
```bash
#!/bin/bash
set -e

# 1. Update code
cd /root/quantum_trader
git pull origin main

# 2. Verify code is correct
grep -q "TESTNET_MODE" microservices/apply_layer/main.py && echo "✅ Code updated"

# 3. Restart service
systemctl restart quantum-apply-layer

# 4. Verify PRODUCTION mode is set
sleep 2
journalctl -u quantum-apply-layer -n 1 --no-pager | grep "PRODUCTION MODE" && echo "✅ Production mode active"

# 5. Verify metrics available
sleep 1
curl -s http://localhost:8000/metrics | head -5 && echo "✅ Metrics endpoint ready"

# 6. Test kill switch
redis-cli SET quantum:global:kill_switch true
sleep 1
journalctl -u quantum-apply-layer -n 5 --no-pager | grep KILL_SWITCH && echo "✅ Kill switch working"
redis-cli SET quantum:global:kill_switch false

# 7. Verify normal execution resumed
sleep 2
echo "✅ Deployment complete - system ready"
```

### Environment Config Template
```bash
# /etc/quantum/apply-layer.env

# Hard Mode Switch
TESTNET=false

# Permit Configuration
APPLY_PERMIT_WAIT_MS=1200
APPLY_PERMIT_STEP_MS=100

# Logging
APPLY_LOG_LEVEL=INFO

# Redis
QUANTUM_REDIS_HOST=localhost
QUANTUM_REDIS_PORT=6379
QUANTUM_REDIS_DB=0

# Metrics
PROMETHEUS_PORT=8000

# Binance Testnet
BINANCE_TESTNET_API_KEY=<key>
BINANCE_TESTNET_API_SECRET=<secret>
```

---

## 8️⃣ Runbook: Emergency Stop

**Situation:** System malfunction, need immediate stop  
**Time to activate:** < 10 seconds

```bash
# STEP 1: Activate kill switch (< 2 seconds)
redis-cli SET quantum:global:kill_switch true
echo "Kill switch activated at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# STEP 2: Verify no new orders (< 5 seconds)
journalctl -u quantum-apply-layer -f --no-pager -n 5

# STEP 3: Check current positions
redis-cli HGETALL quantum:position:BTCUSDT

# STEP 4: Notify team (manual)
echo "System HALTED - investigating..."

# STEP 5: Once fixed, deactivate
redis-cli SET quantum:global:kill_switch false
echo "System resumed at $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
```

---

## 9️⃣ Metrics Deep Dive

### Example: Analyze Permit Denial Pattern
```bash
# Count denies by reason (last 24h)
journalctl -u quantum-apply-layer --since "24 hours ago" --no-pager \
  | grep "p33_denied" \
  | grep -oE "reason=[^ ]+" \
  | sort | uniq -c | sort -rn

# Example output:
#  245 reason=reconcile_required_qty_mismatch
#   12 reason=position_size_exceeds_limit
#    3 reason=kill_score_too_high
```

### Example: Monitor Success Rate
```bash
# Real-time success rate (every 30 seconds)
watch -n 30 'journalctl -u quantum-apply-layer --since "5 minutes ago" --no-pager | tee /tmp/last_5min.log && echo "---" && grep "PERMIT_WAIT.*OK" /tmp/last_5min.log | wc -l && echo "successful permits"'
```

### Example: Alert on High Deny Rate
```bash
# Check if denies exceed 50 in last 10 minutes
DENIES=$(journalctl -u quantum-apply-layer --since "10 minutes ago" --no-pager | grep "p33_denied" | wc -l)
if [ $DENIES -gt 50 ]; then
  echo "ALERT: High P3.3 deny rate ($DENIES in 10 min)"
  # Trigger alert
fi
```

---

## 🔟 FAQ

**Q: Is TESTNET=true safe to leave on?**  
A: No. Always use TESTNET=false for production. TESTNET=true bypasses all safety checks.

**Q: What if kill switch is activated and I can't deactivate it?**  
A: Restart Redis or access Redis directly: `redis-cli DEL quantum:global:kill_switch`

**Q: How long does kill switch activation take?**  
A: < 500ms. The check happens before order placement.

**Q: Can I manually force an execution?**  
A: No. All executions must go through proper permit chain (Governor + P3.3).

**Q: What's the difference between kill switch and TESTNET=true?**  
A: Kill switch: Emergency stop (blocks all)  
TESTNET=true: Development bypass (still executes, just without permits)

---

**Status:** 🟢 **LIVE and ACTIVE**  
**Last Tested:** 2026-01-25  
**Next Review:** 2026-02-01
