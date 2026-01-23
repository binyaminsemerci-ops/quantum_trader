# P0.4C Operations Guide

**Status:** ✅ PRODUCTION (Monitoring 45+ positions)  
**Commit:** 2ed4585d (P0.4C: Add evidence report script and proof)

---

## 🎯 Quick Status Check

```bash
# Check services
systemctl status quantum-exit-monitor quantum-execution

# View monitored positions
curl -s http://localhost:8007/positions | python3 -m json.tool | head -50

# Count active positions
curl -s http://localhost:8007/positions | python3 -c "import sys,json; print(f'Positions: {len(json.load(sys.stdin)[\"positions\"])}')"
```

---

## 📊 Evidence Collection

### Generate Complete Proof Report
```bash
# Run evidence collector (READ-ONLY)
bash /home/qt/quantum_trader/ops/p04c_proof_report.sh

# View report
cat /home/qt/quantum_trader/P0.4C_PROOF_REPORT.md
```

### Manual Spot Checks (1-liners)

**Recent CLOSE_EXECUTED events:**
```bash
tail -n 200 /var/log/quantum/execution.log | grep "CLOSE_EXECUTED"
```

**Recent margin bypasses:**
```bash
tail -n 200 /var/log/quantum/execution.log | grep "MARGIN CHECK SKIPPED"
```

**Recent terminal states:**
```bash
tail -n 200 /var/log/quantum/execution.log | grep "TERMINAL STATE: FILLED"
```

**Complete proof chain (last symbol):**
```bash
SYMBOL=$(grep "EXIT_PUBLISH" /var/log/quantum/exit-monitor.log | tail -1 | grep -oP '\b[A-Z]+USDT\b' | head -1)
echo "=== Proof Chain for $SYMBOL ===" && \
grep -E "EXIT_PUBLISH.*$SYMBOL|MARGIN CHECK SKIPPED.*$SYMBOL|CLOSE_EXECUTED.*$SYMBOL|TERMINAL STATE: FILLED.*$SYMBOL" \
/var/log/quantum/{exit-monitor.log,execution.log} | tail -10
```

**All proof stages in one line:**
```bash
tail -n 200 /var/log/quantum/execution.log | grep -E "CLOSE_EXECUTED|TERMINAL STATE: FILLED|MARGIN CHECK SKIPPED"
```

---

## 🧪 Manual Testing (Use with Caution)

**Manual close endpoint (requires token):**
```bash
# Test close (force=true required)
curl -X POST "http://localhost:8007/manual-close/ATOMUSDT?reason=TEST&force=true" \
  -H "X-Exit-Token: $EXIT_MONITOR_MANUAL_CLOSE_TOKEN"

# Check if manual close is enabled
grep "EXIT_MONITOR_MANUAL_CLOSE_ENABLED" /etc/quantum/testnet.env
```

**Disable manual close in production:**
```bash
# Set to false after testing period
sed -i 's/EXIT_MONITOR_MANUAL_CLOSE_ENABLED=true/EXIT_MONITOR_MANUAL_CLOSE_ENABLED=false/' /etc/quantum/testnet.env
systemctl restart quantum-exit-monitor
```

---

## 🔍 Code Verification (1-liners)

**Verify allowed_fields includes P0.4C fields:**
```bash
grep -A 10 "allowed_fields = {" /home/qt/quantum_trader/services/execution_service.py | grep -E "reduce_only|reason"
```

**Verify margin bypass exists:**
```bash
grep -B 2 -A 2 "MARGIN CHECK SKIPPED" /home/qt/quantum_trader/services/execution_service.py | head -5
```

**Verify CLOSE_EXECUTED logging:**
```bash
grep -A 7 "CLOSE_EXECUTED" /home/qt/quantum_trader/services/execution_service.py | head -8
```

---

## 📈 Monitoring

**Watch for new exits (live):**
```bash
tail -f /var/log/quantum/exit-monitor.log | grep --line-buffered "EXIT_PUBLISH"
```

**Watch for close executions (live):**
```bash
tail -f /var/log/quantum/execution.log | grep --line-buffered -E "CLOSE_EXECUTED|MARGIN CHECK SKIPPED"
```

**Exit monitor health:**
```bash
curl -s http://localhost:8007/health | python3 -m json.tool
```

---

## 🚨 Troubleshooting

**No MARGIN CHECK SKIPPED logs:**
- Check if `reduce_only` field is in allowed_fields (line 963)
- Verify exit_monitor publishes with `reduce_only=True`
- Check Redis stream: `redis-cli XREAD COUNT 1 STREAMS trade.intent $`

**No CLOSE_EXECUTED logs:**
- Check if `reduce_only` is being deserialized correctly
- Verify allowed_fields includes 'reduce_only' and 'reason'
- Check execution service logs for parsing errors

**Margin errors on closes:**
- Ensure margin bypass (line 562) is before margin calculation
- Verify `getattr(intent, 'reduce_only', False)` is True
- Check if allowed_fields is filtering out reduce_only

---

## 📚 Key Files

- **Evidence Script:** `/home/qt/quantum_trader/ops/p04c_proof_report.sh`
- **Proof Report:** `/home/qt/quantum_trader/P0.4C_PROOF_REPORT.md`
- **Exit Monitor:** `/home/qt/quantum_trader/services/exit_monitor_service.py`
- **Execution Service:** `/home/qt/quantum_trader/services/execution_service.py`
- **EventBus Bridge:** `/home/qt/quantum_trader/ai_engine/services/eventbus_bridge.py`

---

## 🎓 Lessons Learned

**Root Cause (Fixed):**
The critical bug was `allowed_fields` in execution_service.py (line 957-963) not including `'reduce_only'` and `'reason'`. Redis streams had `"reduce_only": true` but deserialization filtered it out, causing `intent.reduce_only` to default to `False`, which triggered margin checks that rejected close orders.

**Defense in Depth:**
1. Schema validation (TradeIntent with reduce_only + reason)
2. allowed_fields whitelist (includes P0.4C fields)
3. Schema guard (warns if reduce_only=True but missing source/reason)
4. Margin bypass (closes are risk-reducing, never blocked)
5. Audit logging (CLOSE_EXECUTED with full trace)

---

**Last Updated:** 2026-01-22T10:15Z  
**Git Commit:** 2ed4585d  
**Status:** ✅ PRODUCTION VERIFIED (6/6 checks PASS)
