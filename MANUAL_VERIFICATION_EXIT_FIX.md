# MANUAL VERIFICATION GUIDE - Exit Gateway Fix

Run these commands in your terminal to verify the fix is working:

## 1. Verify file was uploaded and fix is present

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker exec quantum_backend grep -c 'REDUCE-ONLY FIX' /app/backend/services/execution/exit_order_gateway.py"
```

**Expected:** `1` (meaning the comment is found)

---

## 2. Check for reduceOnly logic

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker exec quantum_backend grep -n 'reduceOnly' /app/backend/services/execution/exit_order_gateway.py | head -10"
```

**Expected output:**
```
276:        # [REDUCE-ONLY FIX] Always set reduceOnly=true for exit orders (futures)
279:            if 'closePosition' not in order_params or not order_params.get('closePosition'):
280:                order_params['reduceOnly'] = True
282:                    f"[EXIT_GATEWAY] {symbol}: Added reduceOnly=true for {order_kind} order"
...
```

---

## 3. Check backend is running

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "systemctl list-units | grep quantum_backend"
```

**Expected:** Container should be running (Up X seconds/minutes)

---

## 4. Check for -4164 errors in recent logs

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 200 quantum_backend 2>&1 | grep '4164'"
```

**Expected:** No output (no errors)  
**If errors:** Shows old errors from before fix

---

## 5. Check for notional-related errors

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 200 quantum_backend 2>&1 | grep -i 'notional must be'"
```

**Expected:** No output (no errors)

---

## 6. Check for exit orders with reduceOnly (wait for trade)

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 500 quantum_backend 2>&1 | grep -E 'EXIT_GATEWAY.*reduceOnly=True'"
```

**Expected after trade:**
```
[EXIT_GATEWAY] 📤 Submitting sl order: module=execution_tpsl_shield, symbol=BTCUSDT, type=STOP_MARKET, positionSide=LONG, reduceOnly=True, params_keys=[...]
```

**Before any trade:** No output (normal)

---

## 7. Check for minNotional guard activity

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 500 quantum_backend 2>&1 | grep -E 'Notional.*min|MIN_NOTIONAL'"
```

**Expected:** May have output like:
```
[EXIT_GATEWAY] BTCUSDT: Notional $3.50 < $5.00 but reduceOnly/closePosition is set, allowing order.
```

**Or:** No output if no small orders yet (normal)

---

## 8. Verify backend health

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "curl -s http://localhost:8000/health | jq"
```

**Expected:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-24T...",
  ...
}
```

---

## 9. Count exit orders in logs (all time)

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "journalctl -u quantum_backend.service 2>&1 | grep -c 'EXIT_GATEWAY.*Submitting'"
```

**Expected:** Number > 0 (shows historical exit orders)  
**If 0:** No exit orders placed yet

---

## 10. Full verification script (comprehensive)

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "bash /tmp/verify_fix.sh"
```

**Expected:** Full report showing:
- ✅ reduceOnly fix deployed
- ✅ minNotional guard deployed
- ✅ No -4164 errors
- ✅ Backend healthy

---

## SUCCESS INDICATORS

✅ **Fix is deployed if:**
1. `REDUCE-ONLY FIX` comment found in file (command #1)
2. `reduceOnly` appears multiple times in file (command #2)
3. Backend container is running (command #3)
4. No -4164 errors in recent logs (command #4)

✅ **Fix is working if:**
5. Exit orders show `reduceOnly=True` in logs (command #6) - after trade
6. minNotional guard logs appear (command #7) - if small orders
7. No "notional must be" errors (command #5)

---

## WHAT TO LOOK FOR

### Good (Fix Working) 🎉
```
[EXIT_GATEWAY] BTCUSDT: Added reduceOnly=true for sl order
[EXIT_GATEWAY] 📤 Submitting sl order: reduceOnly=True
[EXIT_GATEWAY] ✅ Order placed successfully: order_id=123456
```

### Bad (Old Error) ❌
```
APIError(code=-4164): Order's notional must be no smaller than 5
Order submission failed
```

### Expected (Guard Active) ⚠️
```
[EXIT_GATEWAY] ETHUSDT: Notional $3.25 < $5.00 but reduceOnly is set, allowing order.
```

---

## TROUBLESHOOTING

### If file not found:
```bash
# Check file exists
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker exec quantum_backend ls -la /app/backend/services/execution/exit_order_gateway.py"

# Re-upload
scp -i ~/.ssh/hetzner_fresh C:\quantum_trader\backend\services\execution\exit_order_gateway.py root@46.224.116.254:/home/qt/quantum_trader/backend/services/execution/
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker restart quantum_backend"
```

### If backend not running:
```bash
# Check logs
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 50 quantum_backend"

# Restart
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker restart quantum_backend"
```

### If still seeing -4164 errors:
```bash
# Check if old logs or new
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 50 quantum_backend 2>&1 | grep -A 2 -B 2 '4164'"

# Look for timestamp - if old (before restart), ignore
```

---

## QUICK ONE-LINER CHECK

```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "echo '=== Fix Deployed ===' && docker exec quantum_backend grep -c 'REDUCE-ONLY FIX' /app/backend/services/execution/exit_order_gateway.py && echo '=== Recent Errors ===' && docker logs --tail 200 quantum_backend 2>&1 | grep -c '4164' && echo '=== Backend Status ===' && systemctl list-units | grep quantum_backend"
```

**Expected:**
```
=== Fix Deployed ===
1
=== Recent Errors ===
0
=== Backend Status ===
quantum_backend   Up 2 minutes
```

✅ **If you see this output, fix is deployed and working!**

