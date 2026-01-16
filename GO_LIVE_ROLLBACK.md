# GO-LIVE ROLLBACK PLAN

**Purpose**: Emergency procedure to revert from MAINNET to TESTNET/PAPER mode  
**Execution Time**: <5 minutes  
**Risk**: 🟢 LOW (safe operation)

---

## ⚠️ ROLLBACK TRIGGERS

Execute rollback immediately if:
- Critical errors (e.g., -4045 loop, precision errors)
- Unexpected loss >5% account
- Position side conflicts
- TP/SL placement failures
- Rate limit hits
- Operator decision (any concern)

**Rule**: When in doubt, ROLLBACK.

---

## 🔴 ROLLBACK PROCEDURE

### Step 1: STOP NEW ENTRIES (Immediate)
```bash
# Option A: Run abort script (RECOMMENDED)
bash scripts/go_live_abort.sh

# Option B: Manual kill-switch
curl -X POST http://localhost:8000/api/kill-switch/activate \
  -H "Content-Type: application/json" \
  -d '{"reason": "Emergency rollback", "operator": "YOUR_NAME"}'
```

**Verification**:
```bash
# Check kill-switch status
curl http://localhost:8000/api/kill-switch/status
# Expected: {"active": true, "reason": "Emergency rollback"}
```

---

### Step 2: CANCEL OPEN ORDERS
```bash
# Via VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 << 'EOF'
docker exec quantum_auto_executor python3 -c "
from binance.um_futures import UMFutures
import os
client = UMFutures(
    key=os.environ['BINANCE_API_KEY'],
    secret=os.environ['BINANCE_API_SECRET']
)
# Cancel all open orders
client.cancel_open_orders(symbol='BTCUSDT')  # Repeat for each symbol
print('All orders cancelled')
"
EOF
```

**Verification**:
```bash
# Check open orders (should be empty)
docker exec quantum_auto_executor python3 -c "
from binance.um_futures import UMFutures
import os
client = UMFutures(key=os.environ['BINANCE_API_KEY'], secret=os.environ['BINANCE_API_SECRET'])
print(client.get_orders())
"
```

---

### Step 3: CLOSE POSITIONS (Optional - Manual Decision)

**WARNING**: This step closes open positions. Only execute if:
- Losing position with no stop-loss
- Runaway position size
- Operator decision required

```bash
# Check current positions
docker exec quantum_auto_executor python3 -c "
from binance.um_futures import UMFutures
import os
client = UMFutures(key=os.environ['BINANCE_API_KEY'], secret=os.environ['BINANCE_API_SECRET'])
positions = client.get_position_risk()
for pos in positions:
    if float(pos['positionAmt']) != 0:
        print(f\"{pos['symbol']}: {pos['positionAmt']} (PnL: {pos['unRealizedProfit']})\")
"

# Close specific position (MANUAL - replace BTCUSDT and LONG)
docker exec quantum_auto_executor python3 -c "
from binance.um_futures import UMFutures
import os
client = UMFutures(key=os.environ['BINANCE_API_KEY'], secret=os.environ['BINANCE_API_SECRET'])
# LONG position: side=SELL, SHORT position: side=BUY
response = client.new_order(
    symbol='BTCUSDT',
    side='SELL',  # or 'BUY' for SHORT
    type='MARKET',
    quantity=0.001,  # Replace with actual position size
    reduceOnly=True
)
print(response)
"
```

---

### Step 4: REVERT TO TESTNET/PAPER MODE

```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Navigate to quantum_trader directory
cd /home/qt/quantum_trader

# Update .env file
cat > .env.rollback << 'EOF'
# ROLLBACK TO SAFE MODE
BINANCE_USE_TESTNET=true
PAPER_TRADING=true
LIVE_TRADING_ENABLED=false

# Keep observability active
ENABLE_PROMETHEUS=true
ENABLE_GRAFANA=true
EOF

# Backup current .env
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)

# Apply rollback env
cp .env.rollback .env

# Restart critical services
docker compose -f systemctl.vps.yml restart ai-engine
docker compose -f systemctl.vps.yml restart auto-executor
docker compose -f systemctl.vps.yml restart risk-brain
docker compose -f systemctl.vps.yml restart exit-brain

echo "✅ Services restarted in TESTNET/PAPER mode"
```

**Verification**:
```bash
# Check service logs
journalctl -u quantum_auto_executor.service --tail 20 | grep -i "testnet\|paper"
# Expected: Lines showing "TESTNET mode" or "PAPER TRADING enabled"

# Check no real orders being submitted
journalctl -u quantum_auto_executor.service --tail 50 | grep "ORDER_SUBMIT"
# Expected: No new ORDER_SUBMIT logs after restart
```

---

### Step 5: VERIFY ROLLBACK COMPLETE

```bash
# Checklist
echo "=== ROLLBACK VERIFICATION ==="

# 1. Kill-switch active
curl -s http://localhost:8000/api/kill-switch/status | grep '"active":true' && echo "✅ Kill-switch active" || echo "❌ Kill-switch not active"

# 2. No open orders
ORDERS=$(docker exec quantum_auto_executor python3 -c "from binance.um_futures import UMFutures; import os; client = UMFutures(key=os.environ['BINANCE_API_KEY'], secret=os.environ['BINANCE_API_SECRET']); print(len(client.get_orders()))")
[ "$ORDERS" -eq 0 ] && echo "✅ No open orders" || echo "⚠️ $ORDERS open orders remain"

# 3. Testnet mode enabled
grep "BINANCE_USE_TESTNET=true" .env && echo "✅ Testnet mode enabled" || echo "❌ Still in mainnet mode"

# 4. Paper trading enabled
grep "PAPER_TRADING=true" .env && echo "✅ Paper trading enabled" || echo "❌ Paper trading disabled"

# 5. Services running
systemctl list-units --filter name=quantum_auto_executor --filter status=running && echo "✅ Executor running" || echo "❌ Executor not running"

echo "=== END VERIFICATION ==="
```

**Expected Output**:
```
✅ Kill-switch active
✅ No open orders
✅ Testnet mode enabled
✅ Paper trading enabled
✅ Executor running
```

---

### Step 6: DOCUMENT ROLLBACK

Create `GO_LIVE_ROLLBACK_PROOF.md`:

```bash
cat > GO_LIVE_ROLLBACK_PROOF.md << EOF
# GO-LIVE ROLLBACK PROOF

**Date**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
**Operator**: [YOUR_NAME]
**Reason**: [REASON FOR ROLLBACK]

## Actions Taken
- [x] Activated kill-switch
- [x] Cancelled open orders
- [ ] Closed positions (if applicable)
- [x] Reverted to testnet/paper mode
- [x] Restarted services
- [x] Verified rollback complete

## Final State
- Mode: TESTNET + PAPER TRADING
- Open Orders: 0
- Open Positions: [LIST OR "0"]
- Services: Running
- Kill-Switch: Active

## Logs Snapshot
\`\`\`
$(journalctl -u quantum_auto_executor.service --tail 20)
\`\`\`

## Next Steps
- [ ] Investigate root cause
- [ ] Fix issues
- [ ] Re-run preflight
- [ ] Restart Go-Live from Phase A
EOF

echo "✅ Rollback proof document created"
```

---

## 🔄 ROLLBACK DECISION TREE

```
 [Critical Error Detected]
         |
         v
   [Is it fixable in <5 min?]
      /        \
    NO         YES
     |          |
     v          v
  ROLLBACK   Fix + Monitor
     |          |
     v          v
 [Follow Steps 1-6]  [If fix fails → ROLLBACK]
```

---

## 📋 POST-ROLLBACK CHECKLIST

After rollback is complete:

- [ ] Document incident in `GO_LIVE_INCIDENTS.md`
- [ ] Review logs to identify root cause
- [ ] Fix identified issues
- [ ] Update Go-Live checklist with new safeguards
- [ ] Re-run preflight (`scripts/go_live_preflight.sh`)
- [ ] Consider re-starting from Phase A (Shadow Mode)

---

## 🎯 ROLLBACK SUCCESS CRITERIA

✅ Kill-switch active  
✅ No open orders  
✅ Services in TESTNET/PAPER mode  
✅ No new real orders being submitted  
✅ Rollback documented

**If all criteria met**: Rollback successful. System is safe.

---

## ⚡ EMERGENCY CONTACT

**If rollback fails or unclear**:
- Stop all Docker containers: `docker compose -f systemctl.vps.yml stop`
- Manually close positions via Binance UI
- Contact: [OPERATOR_CONTACT]

**Remember**: Capital preservation > uptime. Always err on the side of caution.

