#!/bin/bash
# FULL SYSTEM AUDIT — READ ONLY — NO CHANGES
echo "========================================================"
echo " QUANTUM TRADER — FULL SYSTEM AUDIT — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"

echo ""
echo "════════════════════════════════════════"
echo " LAYER 0 — SERVICES"
echo "════════════════════════════════════════"
for svc in \
    quantum-apply-layer \
    quantum-intent-executor \
    quantum-meta-regime \
    quantum-feature-publisher \
    quantum-governor \
    quantum-exchange-stream-bridge \
    quantum-harvest-v2 \
    quantum-universe-service \
    quantum-layer4-portfolio-optimizer \
    quantum-risk-manager \
    quantum-ensemble-predictor \
    quantum-data-collector; do
    STATUS=$(systemctl is-active $svc 2>/dev/null || echo "not-found")
    UPTIME=$(systemctl show $svc --property=ActiveEnterTimestamp 2>/dev/null | cut -d= -f2 | xargs)
    echo "  $svc: $STATUS | started: $UPTIME"
done

echo ""
echo "════════════════════════════════════════"
echo " LAYER 1 — REDIS KEY COUNTS"
echo "════════════════════════════════════════"
for pattern in \
    "quantum:position:[A-Z]*" \
    "quantum:cooldown:*" \
    "quantum:harvest:proposal:*" \
    "quantum:position:ledger:*" \
    "quantum:intent_executor:done:*" \
    "quantum:layer4:sizing:*" \
    "quantum:dedup:plan:*" \
    "quantum:apply:dedupe:*" \
    "quantum:churn:blacklist:*" \
    "quantum:risk:*" \
    "quantum:permit:p33:*" \
    "quantum:stream:signals" \
    "quantum:stream:features" \
    "quantum:stream:exchange.raw"; do
    COUNT=$(redis-cli KEYS "$pattern" 2>/dev/null | grep -v "^$" | wc -l)
    echo "  $pattern: $COUNT"
done

echo ""
echo "════════════════════════════════════════"
echo " LAYER 2 — OPEN POSITIONS (Redis)"
echo "════════════════════════════════════════"
POS_KEYS=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | grep -v "ledger\|snapshot\|history" | sort)
if [ -z "$POS_KEYS" ]; then
    echo "  [NONE] No open positions in Redis"
else
    for key in $POS_KEYS; do
        echo "  --- $key ---"
        redis-cli HGETALL "$key" 2>/dev/null | paste - - | head -15
    done
fi

echo ""
echo "════════════════════════════════════════"
echo " LAYER 3 — HARVEST PROPOSALS"
echo "════════════════════════════════════════"
PROP_KEYS=$(redis-cli KEYS 'quantum:harvest:proposal:*' 2>/dev/null | sort)
if [ -z "$PROP_KEYS" ]; then
    echo "  [NONE] No harvest proposals"
else
    for key in $PROP_KEYS; do
        SYM=$(echo $key | sed 's/quantum:harvest:proposal://')
        ACTION=$(redis-cli HGET "$key" harvest_action 2>/dev/null)
        DEC=$(redis-cli HGET "$key" decision 2>/dev/null)
        RNET=$(redis-cli HGET "$key" R_net 2>/dev/null)
        echo "  $SYM: action=$ACTION decision=$DEC R_net=$RNET"
    done
fi

echo ""
echo "════════════════════════════════════════"
echo " LAYER 4 — KELLY SIZING (Layer4)"
echo "════════════════════════════════════════"
redis-cli KEYS 'quantum:layer4:sizing:*' 2>/dev/null | sort | while read key; do
    SYM=$(echo $key | sed 's/quantum:layer4:sizing://')
    REC=$(redis-cli HGET "$key" recommendation 2>/dev/null)
    SIZE=$(redis-cli HGET "$key" size_usdt 2>/dev/null)
    KELLY=$(redis-cli HGET "$key" kelly_adj 2>/dev/null)
    TS=$(redis-cli HGET "$key" ts 2>/dev/null)
    AGE=$(($(date +%s) - ${TS:-0}))
    echo "  $SYM: rec=$REC size=$SIZE kelly=$KELLY age=${AGE}s"
done

echo ""
echo "════════════════════════════════════════"
echo " LAYER 5 — RISK STATE"
echo "════════════════════════════════════════"
echo "  daily_pnl:          $(redis-cli GET quantum:risk:daily_pnl 2>/dev/null)"
echo "  consecutive_losses: $(redis-cli GET quantum:risk:consecutive_losses 2>/dev/null)"
echo "  kill_switch:        $(redis-cli GET quantum:kill_switch 2>/dev/null)"
echo "  churn_global_freeze:$(redis-cli EXISTS quantum:churn:global_freeze 2>/dev/null)"
echo "  meta_regime:        $(redis-cli GET quantum:meta:regime 2>/dev/null)"
echo "  meta_regime_ts:     $(redis-cli GET quantum:meta:regime:ts 2>/dev/null)"
echo "  permit_p33:         $(redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | head -5)"

echo ""
echo "════════════════════════════════════════"
echo " LAYER 6 — SIGNAL STREAM (last 10)"
echo "════════════════════════════════════════"
redis-cli XREVRANGE quantum:stream:signals + - COUNT 10 2>/dev/null | grep -E "symbol|risk_context|expected_edge|action|decision" | paste - - | head -30

echo ""
echo "════════════════════════════════════════"
echo " LAYER 7 — EXCHANGE.RAW SYMBOLS (last 200)"
echo "════════════════════════════════════════"
redis-cli XREVRANGE quantum:stream:exchange.raw + - COUNT 200 2>/dev/null | grep -E "^symbol$" -A1 | grep -v "^symbol$" | grep -v "^--" | sort | uniq -c | sort -rn | head -25

echo ""
echo "════════════════════════════════════════"
echo " LAYER 8 — META-REGIME LOG (last 5)"
echo "════════════════════════════════════════"
journalctl -u quantum-meta-regime.service -n 5 --no-pager 2>/dev/null | tail -5

echo ""
echo "════════════════════════════════════════"
echo " LAYER 9 — APPLY-LAYER LOG (last 10)"
echo "════════════════════════════════════════"
journalctl -u quantum-apply-layer.service -n 10 --no-pager 2>/dev/null | grep -v "^--" | tail -10

echo ""
echo "════════════════════════════════════════"
echo " LAYER 10 — INTENT-EXECUTOR LOG (last 10)"
echo "════════════════════════════════════════"
journalctl -u quantum-intent-executor.service -n 10 --no-pager 2>/dev/null | grep -v "^--" | tail -10

echo ""
echo "════════════════════════════════════════"
echo " LAYER 11 — GOVERNOR LOG (last 5)"
echo "════════════════════════════════════════"
journalctl -u quantum-governor.service -n 5 --no-pager 2>/dev/null | tail -5

echo ""
echo "════════════════════════════════════════"
echo " LAYER 12 — FEATURE STREAM HEALTH"
echo "════════════════════════════════════════"
FEAT_LEN=$(redis-cli XLEN quantum:stream:features 2>/dev/null)
FEAT_LAST=$(redis-cli XREVRANGE quantum:stream:features + - COUNT 1 2>/dev/null | grep -oE "[0-9]{10}")
echo "  stream length: $FEAT_LEN"
echo "  last entry ts: $FEAT_LAST"
NOW=$(date +%s)
if [ -n "$FEAT_LAST" ]; then
    AGE=$((NOW - FEAT_LAST))
    echo "  age: ${AGE}s ago"
fi

echo ""
echo "════════════════════════════════════════"
echo " LAYER 13 — COOLDOWNS"
echo "════════════════════════════════════════"
CD_KEYS=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null)
CD_COUNT=$(echo "$CD_KEYS" | grep -c "[A-Z]" 2>/dev/null || echo 0)
echo "  total cooldowns: $CD_COUNT"
echo "$CD_KEYS" | sort | head -20 | while read key; do
    TTL=$(redis-cli TTL "$key" 2>/dev/null)
    echo "    $key TTL=${TTL}s"
done

echo ""
echo "════════════════════════════════════════"
echo " LAYER 14 — APPLY.PLAN STREAM (last 5)"
echo "════════════════════════════════════════"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 5 2>/dev/null | grep -E "symbol|decision|reason_codes|action" | head -30

echo ""
echo "════════════════════════════════════════"
echo " LAYER 15 — BINANCE TESTNET POSITIONS"
echo "════════════════════════════════════════"
source /etc/quantum/testnet.env 2>/dev/null
/opt/quantum/venvs/ai-client-base/bin/python3 - << 'PYEOF'
import os, hmac, hashlib, time, urllib.request, urllib.parse, json
api_key = os.environ.get("BINANCE_API_KEY","")
api_secret = os.environ.get("BINANCE_API_SECRET","")
if not api_key:
    print("  [ERROR] No API key in env")
    exit(0)
ts = int(time.time()*1000)
params = f"timestamp={ts}"
sig = hmac.new(api_secret.encode(), params.encode(), hashlib.sha256).hexdigest()
url = f"https://testnet.binancefutures.com/fapi/v2/positionRisk?{params}&signature={sig}"
req = urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    open_pos = [p for p in data if float(p.get("positionAmt",0)) != 0]
    if not open_pos:
        print("  [FLAT] No open positions on Binance testnet")
    else:
        for p in open_pos:
            print(f"  {p['symbol']}: qty={p['positionAmt']} side={p['positionSide']} pnl={p['unrealizedProfit']}")
except Exception as e:
    print(f"  [ERROR] {e}")
PYEOF

echo ""
echo "════════════════════════════════════════"
echo " LAYER 16 — EXECUTED TRADES TODAY"
echo "════════════════════════════════════════"
DONE_COUNT=$(redis-cli KEYS 'quantum:intent_executor:done:*' 2>/dev/null | wc -l)
echo "  done markers in Redis: $DONE_COUNT"
echo "  Recent executed (apply-layer logs):"
journalctl -u quantum-apply-layer.service --since "1 hour ago" --no-pager 2>/dev/null | grep -i "executed=True\|FILLED\|executed_true" | tail -10

echo ""
echo "════════════════════════════════════════"
echo " LAYER 17 — EXCHANGE BRIDGE CONFIG"
echo "════════════════════════════════════════"
cat /etc/quantum/exchange-stream-bridge.env 2>/dev/null

echo ""
echo "════════════════════════════════════════"
echo " LAYER 18 — APPLY_LAYER SOURCE INTEGRITY"
echo "════════════════════════════════════════"
echo "  File: $(ls -la /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  PARTIAL_25 occurrences: $(grep -c PARTIAL_25 /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  PARTIAL_50 occurrences: $(grep -c PARTIAL_50 /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  PARTIAL_75 occurrences: $(grep -c PARTIAL_75 /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  FULL_CLOSE occurrences: $(grep -c FULL_CLOSE_PROPOSED /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  kelly_200usdt occurrences: $(grep -c kelly_200usdt /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  CLOSE_PARTIAL_25 step: $(grep -c CLOSE_PARTIAL_25 /opt/quantum/microservices/apply_layer/main.py 2>/dev/null)"
echo "  Backups:"
ls -la /opt/quantum/microservices/apply_layer/main.py.bak* 2>/dev/null | tail -5

echo ""
echo "========================================================"
echo " AUDIT COMPLETE — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
