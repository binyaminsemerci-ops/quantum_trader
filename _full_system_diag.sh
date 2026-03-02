#!/bin/bash
# =============================================================
# TOTAL SYSTEM DIAGNOSIS — ALL LAYERS
# =============================================================
NOW=$(date +%s)
echo "============================================================"
echo " TOTAL SYSTEM DIAGNOSIS — $(date)"
echo "============================================================"

# ─── LAYER 0: SYSTEM RESOURCES ───────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 0: SYSTEM RESOURCES"
echo "══════════════════════════════════════════"
echo "--- CPU (top 10 consumers) ---"
ps aux --sort=-%cpu | awk 'NR==1 || NR<=11 {printf "%-10s %5s %5s %s\n", $1, $2, $3, $11}' | head -12

echo ""
echo "--- RAM ---"
free -h

echo ""
echo "--- DISK ---"
df -h / | tail -1

echo ""
echo "--- LOAD ---"
uptime

# ─── LAYER 1: REDIS HEALTH ───────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 1: REDIS HEALTH"
echo "══════════════════════════════════════════"
echo "--- Redis info (memory/clients/ops) ---"
redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human"
redis-cli INFO clients | grep connected_clients
redis-cli INFO stats | grep -E "instantaneous_ops_per_sec|total_commands_processed"
redis-cli INFO persistence | grep -E "rdb_last_save_time|aof_enabled"

echo ""
echo "--- Stream lengths (key streams) ---"
for stream in \
  "quantum:stream:apply.plan" \
  "quantum:stream:apply.result" \
  "quantum:stream:signal.score" \
  "quantum:stream:harvest.v2.shadow" \
  "quantum:stream:features.BTCUSDT" \
  "quantum:stream:features.ETHUSDT" \
  "quantum:stream:trade.intent" \
  "harvest.v2.shadow" \
  "quantum:stream:harvest.intent"; do
    len=$(redis-cli XLEN "$stream" 2>/dev/null || echo "N/A")
    echo "  $stream → $len"
done

echo ""
echo "--- Consumer group lag (apply.plan) ---"
redis-cli XINFO GROUPS quantum:stream:apply.plan 2>/dev/null | grep -E "name|pending|last-delivered|consumers" | head -20

echo ""
echo "--- Consumer group lag (signal.score) ---"
redis-cli XINFO GROUPS quantum:stream:signal.score 2>/dev/null | grep -E "name|pending|last-delivered" | head -20

echo ""
echo "--- Redis key counts by prefix ---"
redis-cli KEYS "quantum:position:*" | grep -v "snapshot\|ledger" | wc -l | xargs echo "  active positions:"
redis-cli KEYS "quantum:position:snapshot:*" | wc -l | xargs echo "  position snapshots:"
redis-cli KEYS "quantum:layer4:sizing:*" | wc -l | xargs echo "  kelly sizing keys:"
redis-cli KEYS "quantum:harvest:proposal:*" | wc -l | xargs echo "  harvest proposals:"
redis-cli KEYS "quantum:churn:blacklist:*" | wc -l | xargs echo "  churn blacklisted:"
redis-cli EXISTS quantum:churn:global_freeze | xargs echo "  global_freeze (1=yes):"
redis-cli KEYS "quantum:cooldown:*" | wc -l | xargs echo "  active cooldowns:"

# ─── LAYER 2: SERVICES STATUS ────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 2: ALL QUANTUM SERVICES"
echo "══════════════════════════════════════════"
systemctl list-units 'quantum-*' --type=service --all 2>/dev/null \
  | awk '{printf "  %-55s %s\n", $1, $3}' \
  | grep -v "^$" | head -60

echo ""
echo "--- Failed/inactive services ---"
systemctl list-units 'quantum-*' --type=service --state=failed,inactive --all 2>/dev/null \
  | awk '{print "  FAIL/INACTIVE:", $1}' | head -20

# ─── LAYER 3: SIGNAL PIPELINE ────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 3: SIGNAL PIPELINE"
echo "══════════════════════════════════════════"

echo "--- Latest signal.score entry ---"
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 1 \
  | grep -E "symbol|suggested_action|confidence|timestamp" | head -10

echo ""
echo "--- Signal age (seconds since last signal) ---"
LAST_TS=$(redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 1 | grep "timestamp" | tail -1 | awk '{print $2}' | cut -d. -f1)
if [ -n "$LAST_TS" ] && [ "$LAST_TS" -gt 0 ] 2>/dev/null; then
    AGE=$((NOW - LAST_TS))
    echo "  Last signal: ${AGE}s ago (>60s = stale)"
else
    echo "  Cannot determine signal age"
fi

echo ""
echo "--- Latest features stream (BTCUSDT) ---"
redis-cli XREVRANGE quantum:stream:features.BTCUSDT + - COUNT 1 \
  | grep -E "timestamp|symbol|close|volume" | head -8

echo ""
echo "--- Ensemble predictor process env (SYNTHETIC_MODE) ---"
EP_PID=$(ps aux | grep ensemble_predictor | grep -v grep | awk '{print $2}' | head -1)
echo "  PID: $EP_PID"
if [ -n "$EP_PID" ]; then
    cat /proc/$EP_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "SYNTH|MODE|CONFIDENCE|ENSEMBLE" | head -10
else
    echo "  NOT RUNNING"
fi

# ─── LAYER 4: APPLY LAYER ────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 4: APPLY LAYER"
echo "══════════════════════════════════════════"
echo "--- apply-layer env (live from process) ---"
AL_PID=$(ps aux | grep apply_layer | grep -v grep | awk '{print $2}' | head -1)
echo "  PID: $AL_PID"
if [ -n "$AL_PID" ]; then
    cat /proc/$AL_PID/environ 2>/dev/null | tr '\0' '\n' \
      | grep -E "SYMBOLS|RISK_DAILY|K_OPEN|MAX_POS|MIN_POS|CONSECUTIVE" | head -15
fi

echo ""
echo "--- Latest apply.plan entries (last 3) ---"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 \
  | grep -E "symbol|action|decision|source|R_net|timestamp" | head -30

echo ""
echo "--- apply.plan rate (last 60s) ---"
PLAN_LEN=$(redis-cli XLEN quantum:stream:apply.plan)
echo "  Total apply.plan entries: $PLAN_LEN"

echo ""
echo "--- Governor state ---"
redis-cli GET quantum:kill 2>/dev/null | xargs echo "  quantum:kill (1=KILL):"
redis-cli GET quantum:mode 2>/dev/null | xargs echo "  quantum:mode:"
redis-cli GET quantum:churn:global_freeze 2>/dev/null | xargs echo "  global_freeze:"

# ─── LAYER 5: INTENT EXECUTOR ────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 5: INTENT EXECUTOR"
echo "══════════════════════════════════════════"
echo "--- intent-executor live metrics (last log line) ---"
journalctl -u quantum-intent-executor.service -n 3 --no-pager 2>/dev/null \
  | grep -E "Metrics:|executed_true|harvest_exec|blocked_source|p35_guard"

echo ""
echo "--- Latest apply.result (last 3) ---"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 \
  | grep -E "symbol|decision|executed|error|timestamp" | head -30

echo ""
echo "--- intent-executor SOURCE_ALLOWLIST (from process env) ---"
IE_PID=$(ps aux | grep intent_executor | grep -v grep | awk '{print $2}' | head -1)
echo "  PID: $IE_PID"
if [ -n "$IE_PID" ]; then
    cat /proc/$IE_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "ALLOWLIST|SOURCE" | head -5
fi

# ─── LAYER 6: HARVEST ────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 6: HARVEST ENGINE"
echo "══════════════════════════════════════════"
echo "--- harvest_v2 config ---"
redis-cli HMGET quantum:config:harvest_v2 stream_live r_target_base partial_25_r trailing_step

echo ""
echo "--- harvest_v2 last 3 shadow decisions ---"
redis-cli XREVRANGE quantum:stream:harvest.v2.shadow + - COUNT 3 \
  | grep -E "symbol|R_net|decision|emit_reason|timestamp" | head -24

echo ""
echo "--- harvest proposals (live in Redis) ---"
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT; do
    ACTION=$(redis-cli HGET quantum:harvest:proposal:$sym harvest_action 2>/dev/null)
    RNET=$(redis-cli HGET quantum:harvest:proposal:$sym R_net 2>/dev/null)
    UPNL=$(redis-cli HGET quantum:harvest:proposal:$sym position_unrealized_pnl 2>/dev/null)
    if [ -n "$ACTION" ]; then
        echo "  $sym: action=$ACTION R_net=$RNET upnl=$UPNL"
    fi
done

# ─── LAYER 7: POSITIONS & PNL ────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 7: OPEN POSITIONS"
echo "══════════════════════════════════════════"
echo "--- Binance testnet positions ---"
source /etc/quantum/testnet.env 2>/dev/null
python3 - << 'PYEOF'
import requests, hmac, hashlib, time, os
api_key = os.environ.get("BINANCE_TESTNET_API_KEY","")
secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY", os.environ.get("BINANCE_TESTNET_API_SECRET",""))
ts = int(time.time()*1000)
msg = f"timestamp={ts}"
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
r = requests.get(
    "https://testnet.binancefuture.com/fapi/v2/positionRisk",
    headers={"X-MBX-APIKEY": api_key},
    params={"timestamp": ts, "signature": sig}, timeout=10
)
data = r.json()
if isinstance(data, list):
    pos = [p for p in data if abs(float(p.get("positionAmt",0))) > 0]
    if pos:
        total_upnl = 0
        for p in pos:
            amt = float(p["positionAmt"]); upnl = float(p["unRealizedProfit"])
            entry = float(p["entryPrice"]); notional = abs(amt)*entry
            side = "LONG" if amt > 0 else "SHORT"
            total_upnl += upnl
            print(f"  {p['symbol']}: {side} qty={abs(amt):.4f} entry={entry:.4f} upnl={upnl:+.4f} notional={notional:.1f}USDT lev={p['leverage']}x")
        print(f"  TOTAL UNREALIZED PNL: {total_upnl:+.4f} USDT")
    else:
        print("  NO OPEN POSITIONS")
else:
    print(f"  API ERROR: {data}")
PYEOF

echo ""
echo "--- Redis position keys ---"
for k in $(redis-cli KEYS "quantum:position:*" | grep -v "snapshot\|ledger" | sort); do
    SYM=$(redis-cli HGET $k symbol 2>/dev/null)
    SIDE=$(redis-cli HGET $k side 2>/dev/null)
    QTY=$(redis-cli HGET $k quantity 2>/dev/null)
    UPNL=$(redis-cli HGET $k unrealized_pnl 2>/dev/null)
    echo "  $k: $SYM $SIDE qty=$QTY upnl=$UPNL"
done

# ─── LAYER 8: KELLY SIZING ───────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 8: KELLY SIZING (fresh check)"
echo "══════════════════════════════════════════"
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT; do
    size=$(redis-cli HGET quantum:layer4:sizing:$sym size_usdt 2>/dev/null)
    rec=$(redis-cli HGET quantum:layer4:sizing:$sym recommendation 2>/dev/null)
    ts_val=$(redis-cli HGET quantum:layer4:sizing:$sym ts 2>/dev/null)
    if [ -n "$ts_val" ] && [ "$ts_val" -gt 0 ] 2>/dev/null; then
        age=$((NOW - ts_val))
        fresh="FRESH"
        [ $age -gt 300 ] && fresh="STALE!"
        echo "  $sym: size=$size rec=$rec age=${age}s [$fresh]"
    else
        echo "  $sym: NO DATA"
    fi
done
echo "  Layer4 optimizer status: $(systemctl is-active quantum-layer4-portfolio-optimizer.service 2>/dev/null || echo inactive)"
echo "  Cron refresh: $(crontab -l 2>/dev/null | grep refresh_kelly || echo 'NOT FOUND')"

# ─── LAYER 9: RECENT EXECUTION QUALITY ───────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 9: EXECUTION QUALITY (last 10 results)"
echo "══════════════════════════════════════════"
EXECUTED=0; SKIPPED=0; BLOCKED=0; HARVEST=0
while IFS= read -r line; do
    echo "$line" | grep -q "executed.*True\|executed.*true" && EXECUTED=$((EXECUTED+1))
    echo "$line" | grep -q '"SKIP"' && SKIPPED=$((SKIPPED+1))
    echo "$line" | grep -q '"BLOCK"' && BLOCKED=$((BLOCKED+1))
    echo "$line" | grep -q "harvest_v2" && HARVEST=$((HARVEST+1))
done < <(redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 2>/dev/null)
echo "  Last 50 results: executed=$EXECUTED skipped=$SKIPPED blocked=$BLOCKED harvest_v2=$HARVEST"

echo ""
echo "--- Unique error reasons in last 50 apply.results ---"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 50 2>/dev/null \
  | grep "^error$" -A1 | grep -v "^error$" | grep -v "^--$" \
  | sort | uniq -c | sort -rn | head -10

# ─── LAYER 10: RISK GATES ────────────────────────────────────
echo ""
echo "══════════════════════════════════════════"
echo " LAYER 10: RISK GATES"
echo "══════════════════════════════════════════"
echo "--- Daily loss tracking ---"
redis-cli GET quantum:risk:daily_pnl 2>/dev/null | xargs echo "  daily_pnl:"
redis-cli GET quantum:risk:consecutive_losses 2>/dev/null | xargs echo "  consecutive_losses:"
redis-cli GET quantum:risk:daily_loss_limit_hit 2>/dev/null | xargs echo "  daily_limit_hit:"

echo ""
echo "--- Governor permits ---"
redis-cli TTL quantum:governor:permit 2>/dev/null | xargs echo "  governor permit TTL (s):"
redis-cli TTL quantum:p33:permit 2>/dev/null | xargs echo "  P3.3 permit TTL (s):"
redis-cli TTL quantum:p26:permit 2>/dev/null | xargs echo "  P2.6 permit TTL (s):"

echo ""
echo "--- Heat gate ---"
redis-cli HGET quantum:state:heat total_heat 2>/dev/null | xargs echo "  total_heat:"

echo ""
echo "============================================================"
echo " DIAGNOSIS COMPLETE — $(date)"
echo "============================================================"
