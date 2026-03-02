#!/bin/bash
# ============================================================
# PROOF CHECK — Verify all drastic fixes are live
# ============================================================
echo "============================================================"
echo " PROOF CHECK — $(date)"
echo "============================================================"

echo ""
echo "=== [1] apply-layer.env — SYMBOLS (should be 12, not 400+) ==="
grep "^SYMBOLS=" /etc/quantum/apply-layer.env | tr ',' '\n' | wc -l
grep "^SYMBOLS=" /etc/quantum/apply-layer.env

echo ""
echo "=== [2] apply-layer.env — RISK LIMITS ==="
grep -E "RISK_DAILY|RISK_MAX|K_OPEN|MIN_POSITION|MAX_POSITION" /etc/quantum/apply-layer.env

echo ""
echo "=== [3] Kelly sizing — CHECK ALL 12 SYMBOLS ==="
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    size=$(redis-cli HGET quantum:layer4:sizing:$sym size_usdt 2>/dev/null)
    rec=$(redis-cli HGET quantum:layer4:sizing:$sym recommendation 2>/dev/null)
    ts_val=$(redis-cli HGET quantum:layer4:sizing:$sym ts 2>/dev/null)
    age=$(($(date +%s) - ts_val))
    echo "  $sym: size_usdt=$size rec=$rec age=${age}s"
done

echo ""
echo "=== [4] Layer4 optimizer — MUST BE INACTIVE ==="
systemctl is-active quantum-layer4-portfolio-optimizer.service 2>/dev/null || echo "inactive"

echo ""
echo "=== [5] Cron job for Kelly refresh ==="
crontab -l 2>/dev/null | grep refresh_kelly

echo ""
echo "=== [6] harvest_v2 config — R THRESHOLDS ==="
redis-cli HMGET quantum:config:harvest_v2 r_target_base partial_25_r partial_50_r partial_75_r trailing_step stream_live

echo ""
echo "=== [7] ai-engine.env MIN_ORDER_USD ==="
grep "MIN_ORDER_USD" /etc/quantum/ai-engine.env

echo ""
echo "=== [8] SERVICES ALL ACTIVE ==="
for svc in quantum-apply-layer quantum-harvest-v2 quantum-intent-executor quantum-ensemble-predictor; do
    status=$(systemctl is-active $svc 2>/dev/null || echo "unknown")
    echo "  $svc: $status"
done

echo ""
echo "=== [9] BINANCE TESTNET — CURRENT OPEN POSITIONS ==="
source /etc/quantum/testnet.env 2>/dev/null
API_KEY="${BINANCE_TESTNET_API_KEY}"
SECRET="${BINANCE_TESTNET_SECRET_KEY:-$BINANCE_TESTNET_API_SECRET}"
python3 - << PYEOF
import requests, hmac, hashlib, time, os
api_key = "${API_KEY}" or os.environ.get("BINANCE_TESTNET_API_KEY","")
secret = "${SECRET}" or os.environ.get("BINANCE_TESTNET_SECRET_KEY","")
ts = int(time.time()*1000)
msg = f"timestamp={ts}"
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
r = requests.get(
    "https://testnet.binancefuture.com/fapi/v2/positionRisk",
    headers={"X-MBX-APIKEY": api_key},
    params={"timestamp": ts, "signature": sig}, timeout=10
)
positions = [p for p in r.json() if abs(float(p.get("positionAmt", 0))) > 0]
if positions:
    for p in positions:
        amt = float(p["positionAmt"])
        upnl = float(p["unRealizedProfit"])
        entry = float(p["entryPrice"])
        notional = abs(amt) * entry
        side = "LONG" if amt > 0 else "SHORT"
        print(f"  {p['symbol']}: {side} qty={abs(amt)} entry={entry} upnl={upnl:.4f} notional={notional:.1f}USDT lev={p['leverage']}x")
else:
    print("  NO OPEN POSITIONS")
PYEOF

echo ""
echo "=== [10] apply.result stream — LATEST 5 RESULTS ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 | grep -E "plan_id|symbol|decision|executed|error|timestamp" | head -40

echo ""
echo "=== [11] intent-executor LIVE METRICS (harvest executed count) ==="
journalctl -u quantum-intent-executor.service -n 5 --no-pager 2>/dev/null | grep -E "Metrics:|harvest_exec|executed_true"

echo ""
echo "=== [12] harvest_v2 shadow stream — LAST 3 (proof it is scanning) ==="
redis-cli XREVRANGE quantum:stream:harvest.v2.shadow + - COUNT 3 | grep -E "symbol|R_net|decision|emit_reason|timestamp" | head -30

echo ""
echo "=== [13] BACKUP EXISTS ==="
ls /etc/quantum/backups.*/apply-layer.env.bak 2>/dev/null | head -3

echo ""
echo "=== [14] STREAM LENGTHS (proof system is active) ==="
echo "apply.plan length:   $(redis-cli XLEN quantum:stream:apply.plan)"
echo "apply.result length: $(redis-cli XLEN quantum:stream:apply.result)"
echo "harvest.v2.shadow:   $(redis-cli XLEN quantum:stream:harvest.v2.shadow)"

echo ""
echo "============================================================"
echo " PROOF CHECK COMPLETE"
echo "============================================================"
