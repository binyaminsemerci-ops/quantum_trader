# HARVEST FIX - VPS PROOF COMMANDS

## PRE-FIX STATE (Baseline)

### Check current CLOSE plans (should see stale ETHUSDT/BTCUSDT)
```bash
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -B5 -A10 "FULL_CLOSE_PROPOSED" | head -80
```

### Check current executions (should be all executed=False)
```bash
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 | head -150
```

### Check current positions
```bash
redis-cli KEYS "quantum:position:*"
redis-cli HGETALL quantum:position:ANKRUSDT
redis-cli HGETALL quantum:position:GPSUSDT
redis-cli HGETALL quantum:position:ARCUSDT
redis-cli HGETALL quantum:position:RIVERUSDT
redis-cli HGETALL quantum:position:FHEUSDT
redis-cli HGETALL quantum:position:HYPEUSDT
```

### Check apply-layer logs (should be only ENTRY logs, no CLOSE)
```bash
journalctl -u quantum-apply-layer --since "10 minutes ago" | grep -E "\[CLOSE\]|\[ENTRY\]" | tail -50
```

### Check harvest-brain logs (stuck on old events)
```bash
tail -100 /mnt/HC_Volume_104287969/quantum-logs/harvest_brain.log
```

---

## DEPLOYMENT SEQUENCE

### 1. Pull latest code
```bash
cd /home/qt/quantum_trader
git pull
```

### 2. Stop harvest-brain
```bash
systemctl stop quantum-harvest-brain
systemctl status quantum-harvest-brain
```

### 3. Fix harvest-brain offset
```bash
cd /home/qt/quantum_trader
bash scripts/fix_harvest_brain_offset.sh
```

### 4. Restart apply-layer (with CLOSE handler)
```bash
systemctl restart quantum-apply-layer
sleep 3
systemctl status quantum-apply-layer
```

### 5. Restart harvest-brain
```bash
systemctl start quantum-harvest-brain
sleep 2
systemctl status quantum-harvest-brain
```

---

## POST-FIX VERIFICATION (Proof)

### A) Verify CLOSE handler is active
```bash
# Check apply-layer logs for CLOSE processing
journalctl -u quantum-apply-layer --since "2 minutes ago" | grep "\[CLOSE\]"

# Expected: Lines like:
# [CLOSE] ETHUSDT: Processing FULL_CLOSE_PROPOSED plan_id=...
# [CLOSE] ETHUSDT: SKIP_CLOSE_QTY_ZERO (stale plan, close_qty=0.0)
```

### B) Wait for fresh CLOSE proposal from exitbrain
```bash
# Monitor exitbrain for new kill_score calculations
journalctl -u quantum-exitbrain-v35 -f | grep -E "kill_score|CLOSE"

# OR check apply.plan for fresh proposals (not ETHUSDT/BTCUSDT)
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 50 | grep -B5 "FULL_CLOSE_PROPOSED"
```

### C) Verify CLOSE execution (once exitbrain generates fresh proposal)
```bash
# Check apply.result for executed=True + reduceOnly=True
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 | grep -B10 "reduceOnly"

# Expected output:
# action: FULL_CLOSE_PROPOSED
# executed: True
# reduceOnly: True
# close_qty: 33892.56
# filled_qty: 33892.56
# order_id: 12345678
# status: FILLED
```

### D) Verify position updated/deleted
```bash
# Check position after close
redis-cli HGETALL quantum:position:ANKRUSDT
# Expected: Either deleted (full close) or reduced quantity (partial close)

# List all remaining positions
redis-cli KEYS "quantum:position:*"
```

### E) Check Binance testnet position
```bash
# Via apply-layer client (has credentials)
# Or manually check Binance testnet UI
curl -X GET "https://testnet.binancefuture.com/fapi/v2/positionRisk" \
  -H "X-MBX-APIKEY: $BINANCE_TESTNET_API_KEY" \
  --data "timestamp=$(date +%s)000&signature=..."
```

---

## GREP-FRIENDLY LOG PATTERNS

### Apply-layer CLOSE logs
```bash
# Processing CLOSE
journalctl -u quantum-apply-layer | grep "Processing FULL_CLOSE_PROPOSED"

# CLOSE execution
journalctl -u quantum-apply-layer | grep "CLOSE_EXECUTE"

# CLOSE completion
journalctl -u quantum-apply-layer | grep "CLOSE_DONE"

# CLOSE skips (no position, zero qty, duplicate)
journalctl -u quantum-apply-layer | grep "SKIP_NO_POSITION\|SKIP_CLOSE_QTY_ZERO\|SKIP_DUPLICATE"
```

### Apply.result stream
```bash
# Show all executed=True entries
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 100 | grep -B15 "^True$"

# Show reduceOnly executions
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 100 | awk '/reduceOnly/{getline; if(/True/) {for(i=0;i<20;i++){print prev[i%20]}; print "reduceOnly"; print $0; for(i=0;i<10;i++){getline; print}}} {prev[NR%20]=$0}'
```

---

## AUTOMATED VERIFICATION SCRIPT

Run comprehensive verification:
```bash
cd /home/qt/quantum_trader
bash scripts/verify_harvest_system.sh
```

---

## SUCCESS CRITERIA

✅ **Immediate (within 1 minute):**
- Apply-layer logs show `[CLOSE]` processing
- Stale ETHUSDT/BTCUSDT plans get `SKIP_CLOSE_QTY_ZERO`
- Harvest-brain no longer stuck (no repeated EGLDUSDT qty=0.0)

✅ **Within 5-15 minutes (exitbrain cycle):**
- Fresh CLOSE proposal for testnet symbols (ANKRUSDT, GPSUSDT, etc.)
- Apply.result shows `executed=True` + `reduceOnly=True`
- Position quantity reduced or deleted in Redis
- Binance testnet position matches Redis state

✅ **Within 30 minutes:**
- Multiple CLOSE executions observed
- Positions closing when profit/loss targets hit
- Realized PnL starts accumulating
- Duplicate blocks continue (anti-dup gate still working)

---

## TROUBLESHOOTING

### If no CLOSE proposals after 15 minutes:
```bash
# Check if exitbrain is running
systemctl status quantum-exitbrain-v35

# Check exitbrain logs
journalctl -u quantum-exitbrain-v35 --since "10 minutes ago" | tail -100

# Check current positions have kill_score data
redis-cli KEYS "quantum:position:*"
```

### If CLOSE proposals exist but not executing:
```bash
# Check apply-layer logs for errors
journalctl -u quantum-apply-layer --since "5 minutes ago" | grep -E "ERROR|Failed"

# Check Binance API credentials
systemctl cat quantum-apply-layer | grep BINANCE_TESTNET

# Test Binance API connection
curl -X GET "https://testnet.binancefuture.com/fapi/v1/ping"
```

### If positions not updating after execution:
```bash
# Check Redis position keys
redis-cli KEYS "quantum:position:*" | xargs -I {} redis-cli HGETALL {}

# Check apply.result for actual filled_qty
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 20 | grep -B5 "filled_qty"
```

---

## FINAL VALIDATION

Run all checks after 30 minutes:
```bash
echo "=== HARVEST SYSTEM FINAL VALIDATION ==="
echo
echo "1. CLOSE plans produced (last hour):"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 500 | grep -c "FULL_CLOSE_PROPOSED"
echo
echo "2. CLOSE executions (last hour):"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 500 | grep -B1 "reduceOnly" | grep -c "^True$"
echo
echo "3. Current open positions:"
redis-cli KEYS "quantum:position:*" | wc -l
echo
echo "4. Duplicate blocks (last 30 min):"
journalctl -u quantum-apply-layer --since "30 minutes ago" | grep -c "SKIP_OPEN_DUPLICATE"
echo
echo "5. Apply-layer CLOSE activity (last 30 min):"
journalctl -u quantum-apply-layer --since "30 minutes ago" | grep -c "\[CLOSE\]"
echo
echo "Expected:"
echo "  - CLOSE plans: >10 (exitbrain active)"
echo "  - CLOSE executions: >0 (apply-layer working)"
echo "  - Open positions: Decreasing over time"
echo "  - Duplicate blocks: >50 (anti-dup active)"
echo "  - CLOSE activity: >10 (processing active)"
```
