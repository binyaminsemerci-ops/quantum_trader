#!/bin/bash
# ============================================================
# TOTAL SYSTEM DIAGNOSTIC — quantum_trader
# Covers: Services, Redis streams, event bus, positions,
#         signal flow, Kelly sizing, harvest pipeline, PnL
# ============================================================
TS=$(date +"%Y-%m-%d %H:%M:%S UTC")
echo "============================================================"
echo " TOTAL SYSTEM DIAGNOSTIC"
echo " $TS"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# SECTION 0 — SYSTEM RESOURCES
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [0] SYSTEM RESOURCES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- CPU load ---"
uptime
echo "--- Memory ---"
free -h | head -3
echo "--- Disk ---"
df -h / | tail -1
echo "--- Process count ---"
ps aux | grep -c qt

# ─────────────────────────────────────────────────────────────
# SECTION 1 — ALL QUANTUM SERVICES
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [1] ALL QUANTUM SYSTEMD SERVICES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
systemctl list-units --type=service --state=active 2>/dev/null | grep quantum | awk '{printf "  %-55s %s\n", $1, $3}'

echo ""
echo "--- FAILED / INACTIVE quantum services ---"
systemctl list-units --type=service 2>/dev/null | grep quantum | grep -v running | awk '{printf "  %-55s %s %s\n", $1, $3, $4}'

# ─────────────────────────────────────────────────────────────
# SECTION 2 — REDIS HEALTH
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [2] REDIS HEALTH"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
redis-cli PING
redis-cli INFO memory | grep -E "used_memory_human|maxmemory_human|mem_fragmentation_ratio"
redis-cli INFO stats | grep -E "total_commands_processed|rejected_connections|keyspace_hits|keyspace_misses" | head -6
echo "--- Total keys in DB0 ---"
redis-cli DBSIZE

# ─────────────────────────────────────────────────────────────
# SECTION 3 — REDIS STREAMS (EVENT BUS)
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [3] REDIS STREAMS (EVENT BUS)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STREAMS=$(redis-cli KEYS "quantum:stream:*" 2>/dev/null | sort)
echo "--- Stream lengths ---"
for stream in $STREAMS; do
    len=$(redis-cli XLEN "$stream" 2>/dev/null)
    # Get age of last message
    last_id=$(redis-cli XREVRANGE "$stream" + - COUNT 1 2>/dev/null | head -1)
    if [ -n "$last_id" ] && [ "$last_id" != "(empty array)" ]; then
        # Extract timestamp from stream ID (milliseconds)
        ts_ms=$(echo "$last_id" | cut -d'-' -f1)
        now_ms=$(($(date +%s) * 1000))
        age_sec=$(( (now_ms - ts_ms) / 1000 ))
        echo "  $stream: len=$len  last_msg=${age_sec}s ago"
    else
        echo "  $stream: len=$len  (empty)"
    fi
done

# ─────────────────────────────────────────────────────────────
# SECTION 4 — SIGNAL PIPELINE (Layer by Layer)
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [4] SIGNAL PIPELINE — Layer by Layer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "--- [4a] LAYER 1: Market features (features.*) ---"
FEAT_STREAMS=$(redis-cli KEYS "quantum:stream:features.*" 2>/dev/null | sort | head -8)
for fs in $FEAT_STREAMS; do
    len=$(redis-cli XLEN "$fs" 2>/dev/null)
    last=$(redis-cli XREVRANGE "$fs" + - COUNT 1 2>/dev/null | head -1)
    ts_ms=$(echo "$last" | cut -d'-' -f1)
    age=$(( ($(date +%s)*1000 - ts_ms) / 1000 ))
    echo "  $fs: len=$len  last=${age}s ago"
done

echo ""
echo "--- [4b] LAYER 2: signal.score (ensemble predictor output) ---"
len=$(redis-cli XLEN quantum:stream:signal.score 2>/dev/null)
last_msg=$(redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 1 2>/dev/null)
echo "  quantum:stream:signal.score: len=$len"
echo "$last_msg" | grep -E "symbol|suggested_action|confidence|timestamp" | head -8

echo ""
echo "--- [4c] LAYER 3: apply.plan (apply_layer decisions) ---"
len=$(redis-cli XLEN quantum:stream:apply.plan 2>/dev/null)
echo "  quantum:stream:apply.plan: len=$len"
echo "  Last 3 plans:"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 | grep -E "symbol|action|decision|kill_score|source|timestamp" | head -20

echo ""
echo "--- [4d] LAYER 3: apply.result (executor feedback) ---"
len=$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null)
echo "  quantum:stream:apply.result: len=$len"
echo "  Last 3 results:"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 3 | grep -E "symbol|decision|executed|error|timestamp" | head -24

echo ""
echo "--- [4e] LAYER 4: harvest.v2.shadow (harvest_v2 scanning) ---"
len=$(redis-cli XLEN quantum:stream:harvest.v2.shadow 2>/dev/null)
echo "  quantum:stream:harvest.v2.shadow: len=$len"
echo "  Last 3 entries:"
redis-cli XREVRANGE quantum:stream:harvest.v2.shadow + - COUNT 3 | grep -E "symbol|R_net|decision|emit_reason|timestamp" | head -24

echo ""
echo "--- [4f] trade.intent stream (legacy path) ---"
len=$(redis-cli XLEN quantum:stream:trade.intent 2>/dev/null)
last=$(redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 1 2>/dev/null | head -1)
ts_ms=$(echo "$last" | cut -d'-' -f1)
age=$(( ($(date +%s)*1000 - ts_ms) / 1000 ))
echo "  quantum:stream:trade.intent: len=$len  last=${age}s ago"

# ─────────────────────────────────────────────────────────────
# SECTION 5 — OPEN POSITIONS
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [5] POSITIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "--- [5a] Redis position keys (harvest_v2 source) ---"
redis-cli KEYS "quantum:position:*" | grep -v "snapshot\|ledger" | sort
echo ""
POS_KEYS=$(redis-cli KEYS "quantum:position:*" | grep -v "snapshot\|ledger")
for pk in $POS_KEYS; do
    sym=$(redis-cli HGET "$pk" symbol 2>/dev/null)
    side=$(redis-cli HGET "$pk" side 2>/dev/null)
    qty=$(redis-cli HGET "$pk" quantity 2>/dev/null)
    entry=$(redis-cli HGET "$pk" entry_price 2>/dev/null)
    upnl=$(redis-cli HGET "$pk" unrealized_pnl 2>/dev/null)
    risk=$(redis-cli HGET "$pk" entry_risk_usdt 2>/dev/null)
    src=$(redis-cli HGET "$pk" source 2>/dev/null)
    echo "  $pk → sym=$sym side=$side qty=$qty entry=$entry upnl=$upnl risk=$risk src=$src"
done

echo ""
echo "--- [5b] Binance testnet LIVE positions ---"
source /etc/quantum/testnet.env 2>/dev/null
python3 - << 'PYEOF'
import requests, hmac, hashlib, time, os
api_key = os.environ.get("BINANCE_TESTNET_API_KEY","")
secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY", os.environ.get("BINANCE_TESTNET_API_SECRET",""))
ts = int(time.time()*1000)
msg = f"timestamp={ts}"
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
r = requests.get("https://testnet.binancefuture.com/fapi/v2/positionRisk",
    headers={"X-MBX-APIKEY": api_key},
    params={"timestamp": ts, "signature": sig}, timeout=10)
positions = [p for p in r.json() if abs(float(p.get("positionAmt",0))) > 0]
if positions:
    total_upnl = sum(float(p["unRealizedProfit"]) for p in positions)
    for p in positions:
        amt = float(p["positionAmt"])
        upnl = float(p["unRealizedProfit"])
        entry = float(p["entryPrice"])
        notional = abs(amt) * entry
        side = "LONG" if amt > 0 else "SHORT"
        print(f"  {p['symbol']}: {side} qty={abs(amt)} entry={entry:.4f} upnl={upnl:+.4f} USDT notional={notional:.1f} USDT lev={p['leverage']}x")
    print(f"  ── TOTAL unrealized PnL: {total_upnl:+.4f} USDT")
else:
    print("  NO OPEN POSITIONS ON BINANCE TESTNET")
PYEOF

# ─────────────────────────────────────────────────────────────
# SECTION 6 — HARVEST ENGINE STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [6] HARVEST ENGINE STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "--- [6a] harvest_v2 config ---"
redis-cli HGETALL quantum:config:harvest_v2

echo ""
echo "--- [6b] harvest.heat keys (active) ---"
redis-cli KEYS "quantum:harvest:heat:*" | sort | while read k; do
    val=$(redis-cli GET "$k" 2>/dev/null)
    echo "  $k = $val"
done

echo ""
echo "--- [6c] harvest.proposals (active) ---"
redis-cli KEYS "quantum:harvest:proposal:*" | sort | while read k; do
    sym=$(echo "$k" | awk -F: '{print $NF}')
    action=$(redis-cli HGET "$k" harvest_action 2>/dev/null)
    R=$(redis-cli HGET "$k" R_net 2>/dev/null)
    ks=$(redis-cli HGET "$k" kill_score 2>/dev/null)
    echo "  $k → action=$action R_net=$R kill_score=$ks"
done

echo ""
echo "--- [6d] Kelly sizing floor (all 12) ---"
NOW=$(date +%s)
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    size=$(redis-cli HGET quantum:layer4:sizing:$sym size_usdt 2>/dev/null)
    rec=$(redis-cli HGET quantum:layer4:sizing:$sym recommendation 2>/dev/null)
    ts_val=$(redis-cli HGET quantum:layer4:sizing:$sym ts 2>/dev/null)
    age=$((NOW - ts_val))
    fresh="FRESH"
    [ $age -gt 300 ] && fresh="STALE!"
    echo "  $sym: size=$size rec=$rec age=${age}s ($fresh)"
done

# ─────────────────────────────────────────────────────────────
# SECTION 7 — GOVERNOR & PERMITS
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [7] GOVERNOR & PERMITS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Kill switch ---"
redis-cli GET quantum:kill 2>/dev/null || echo "  (not set = GO)"
echo "--- Mode ---"
redis-cli GET quantum:mode 2>/dev/null || echo "  (not set)"
echo "--- Governor execution permits (sample) ---"
for sym in BTCUSDT BNBUSDT XRPUSDT SOLUSDT ETHUSDT; do
    gov=$(redis-cli TTL "quantum:governor:permit:$sym" 2>/dev/null)
    p33=$(redis-cli TTL "quantum:p33:permit:$sym" 2>/dev/null)
    p26=$(redis-cli TTL "quantum:p26:permit:$sym" 2>/dev/null)
    echo "  $sym: governor_ttl=$gov p33_ttl=$p33 p26_ttl=$p26"
done
echo "--- Churn state ---"
redis-cli EXISTS quantum:churn:global_freeze 2>/dev/null && echo "  GLOBAL FREEZE ACTIVE" || echo "  No global freeze"
redis-cli KEYS "quantum:churn:blacklist:*" 2>/dev/null | head -5

# ─────────────────────────────────────────────────────────────
# SECTION 8 — INTENT EXECUTOR METRICS
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [8] INTENT EXECUTOR METRICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
journalctl -u quantum-intent-executor.service -n 3 --no-pager 2>/dev/null | grep "Metrics:"
echo ""
echo "--- intent-executor counters in Redis ---"
redis-cli KEYS "quantum:metrics:intent_executor:*" 2>/dev/null | head -10 | while read k; do
    val=$(redis-cli GET "$k" 2>/dev/null)
    echo "  $k = $val"
done

# ─────────────────────────────────────────────────────────────
# SECTION 9 — DAILY RISK STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [9] DAILY RISK STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Daily loss tracker ---"
redis-cli GET quantum:risk:daily_loss 2>/dev/null || echo "  (not set = 0)"
redis-cli GET quantum:risk:daily_pnl 2>/dev/null || echo "  daily_pnl not set"
redis-cli GET quantum:risk:consecutive_losses 2>/dev/null || echo "  consecutive_losses not set"
redis-cli GET quantum:risk:daily_brake 2>/dev/null || echo "  daily_brake not set"
echo "--- apply-layer risk limits (from env) ---"
grep -E "RISK_DAILY|RISK_MAX|K_OPEN" /etc/quantum/apply-layer.env

# ─────────────────────────────────────────────────────────────
# SECTION 10 — RECENT PnL (Last 24h from Binance)
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [10] RECENT PnL — Last 24h Binance testnet"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
source /etc/quantum/testnet.env 2>/dev/null
python3 - << 'PYEOF'
import requests, hmac, hashlib, time, os
from collections import defaultdict

api_key = os.environ.get("BINANCE_TESTNET_API_KEY","")
secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY", os.environ.get("BINANCE_TESTNET_API_SECRET",""))

def signed_get(endpoint, params={}):
    ts = int(time.time()*1000)
    p = dict(params); p["timestamp"] = ts
    qs = "&".join(f"{k}={v}" for k,v in p.items())
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    r = requests.get(f"https://testnet.binancefuture.com{endpoint}",
        headers={"X-MBX-APIKEY": api_key},
        params={**p, "signature": sig}, timeout=10)
    return r.json()

# Last 24h
since = int((time.time() - 86400) * 1000)
income = signed_get("/fapi/v1/income", {"limit": 1000, "startTime": since})

if isinstance(income, list):
    realized = sum(float(x["income"]) for x in income if x["incomeType"] == "REALIZED_PNL")
    commission = sum(float(x["income"]) for x in income if x["incomeType"] == "COMMISSION")
    funding = sum(float(x["income"]) for x in income if x["incomeType"] == "FUNDING_FEE")
    total = realized + commission + funding
    count = len([x for x in income if x["incomeType"] == "REALIZED_PNL"])
    
    # Per-symbol breakdown
    by_sym = defaultdict(float)
    for x in income:
        if x["incomeType"] == "REALIZED_PNL":
            by_sym[x["symbol"]] += float(x["income"])
    
    print(f"  Last 24h (since {time.strftime('%H:%M UTC', time.gmtime(since/1000))}):")
    print(f"  Realized PnL:  {realized:+.4f} USDT  ({count} fills)")
    print(f"  Commissions:   {commission:+.4f} USDT")
    print(f"  Funding:       {funding:+.4f} USDT")
    print(f"  ── TOTAL:      {total:+.4f} USDT")
    if by_sym:
        print(f"  Per-symbol:")
        for sym, pnl in sorted(by_sym.items(), key=lambda x: x[1]):
            print(f"    {sym}: {pnl:+.4f}")
else:
    print(f"  ERROR: {income}")
PYEOF

# ─────────────────────────────────────────────────────────────
# SECTION 11 — APPLY LAYER LIVE STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [11] APPLY LAYER LIVE STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Recent apply-layer logs ---"
journalctl -u quantum-apply-layer.service -n 15 --no-pager 2>/dev/null | grep -E "OPEN|CLOSE|BLOCKED|SKIP|kill_score|layer4|kelly|EXECUTE|ERROR" | head -20

echo ""
echo "--- Last 3 apply.plan messages with full detail ---"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3

# ─────────────────────────────────────────────────────────────
# SECTION 12 — ENSEMBLE PREDICTOR STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [12] ENSEMBLE PREDICTOR STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Recent signal.score samples ---"
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 5 | grep -E "symbol|suggested_action|confidence|expected_edge|timestamp" | head -30
echo ""
echo "--- Ensemble predictor env (SYNTHETIC_MODE?) ---"
cat /proc/$(ps aux | grep ensemble_predictor | grep -v grep | awk '{print $2}' | head -1)/environ 2>/dev/null | tr '\0' '\n' | grep -E "SYNTH|MODE|CONFIDENCE|ENSEMBLE" | head -10

# ─────────────────────────────────────────────────────────────
# SECTION 13 — LEDGER STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [13] LEDGER STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Ledger keys ---"
redis-cli KEYS "quantum:position:ledger:*" | sort
echo ""
for lk in $(redis-cli KEYS "quantum:position:ledger:*" | sort); do
    sym=$(echo "$lk" | awk -F: '{print $NF}')
    side=$(redis-cli HGET "$lk" side 2>/dev/null)
    qty=$(redis-cli HGET "$lk" quantity 2>/dev/null)
    entry=$(redis-cli HGET "$lk" entry_price 2>/dev/null)
    pnl=$(redis-cli HGET "$lk" realized_pnl 2>/dev/null)
    src=$(redis-cli HGET "$lk" source 2>/dev/null)
    echo "  $lk → side=$side qty=$qty entry=$entry realized_pnl=$pnl src=$src"
done

# ─────────────────────────────────────────────────────────────
# SECTION 14 — COOLDOWN / DEDUP KEYS
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [14] COOLDOWN / IDEMPOTENCY KEYS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "--- Active cooldowns (TTL > 0) ---"
for sym in BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT DOTUSDT OPUSDT LTCUSDT; do
    ttl=$(redis-cli TTL "quantum:cooldown:last_exec_ts:$sym" 2>/dev/null)
    [ "$ttl" -gt 0 ] 2>/dev/null && echo "  $sym: cooldown TTL=${ttl}s"
done
echo "--- APPLY_DEDUPE keys (recent plans) ---"
redis-cli KEYS "quantum:apply:dedupe:*" 2>/dev/null | wc -l | xargs echo "  dedupe keys active:"

# ─────────────────────────────────────────────────────────────
# SECTION 15 — ANTI-CHURN STATE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [15] ANTI-CHURN STATE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
redis-cli EXISTS quantum:churn:global_freeze && echo "  ⛔ GLOBAL FREEZE ACTIVE" || echo "  ✅ No global freeze"
BLACKLISTED=$(redis-cli KEYS "quantum:churn:blacklist:*" 2>/dev/null)
if [ -n "$BLACKLISTED" ]; then
    echo "  Blacklisted symbols:"
    echo "$BLACKLISTED" | while read k; do
        ttl=$(redis-cli TTL "$k" 2>/dev/null)
        echo "    $k  TTL=${ttl}s"
    done
else
    echo "  No symbols blacklisted"
fi

# ─────────────────────────────────────────────────────────────
# SECTION 16 — BALANCE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [16] ACCOUNT BALANCE — Binance testnet"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
source /etc/quantum/testnet.env 2>/dev/null
python3 - << 'PYEOF'
import requests, hmac, hashlib, time, os
api_key = os.environ.get("BINANCE_TESTNET_API_KEY","")
secret = os.environ.get("BINANCE_TESTNET_SECRET_KEY", os.environ.get("BINANCE_TESTNET_API_SECRET",""))
ts = int(time.time()*1000)
msg = f"timestamp={ts}"
sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
r = requests.get("https://testnet.binancefuture.com/fapi/v2/account",
    headers={"X-MBX-APIKEY": api_key},
    params={"timestamp": ts, "signature": sig}, timeout=10)
data = r.json()
if "totalWalletBalance" in data:
    print(f"  Wallet balance:     {float(data['totalWalletBalance']):.4f} USDT")
    print(f"  Available balance:  {float(data['availableBalance']):.4f} USDT")
    print(f"  Unrealized PnL:     {float(data['totalUnrealizedProfit']):.4f} USDT")
    print(f"  Margin balance:     {float(data['totalMarginBalance']):.4f} USDT")
    # Show non-zero asset balances
    for asset in data.get("assets", []):
        wb = float(asset.get("walletBalance",0))
        if wb > 0:
            print(f"  Asset {asset['asset']}: walletBalance={wb:.4f}")
else:
    print(f"  ERROR: {data}")
PYEOF

# ─────────────────────────────────────────────────────────────
# SECTION 17 — SUMMARY / HEALTH SCORE
# ─────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " [17] HEALTH SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check each critical component
check() {
    local name=$1; local result=$2; local expected=$3
    if echo "$result" | grep -q "$expected" 2>/dev/null; then
        echo "  ✅ $name: OK ($expected)"
    else
        echo "  ❌ $name: FAIL (got: $result, expected: $expected)"
    fi
}

check "apply-layer service"     "$(systemctl is-active quantum-apply-layer 2>/dev/null)"           "active"
check "harvest-v2 service"      "$(systemctl is-active quantum-harvest-v2 2>/dev/null)"            "active"
check "intent-executor service" "$(systemctl is-active quantum-intent-executor 2>/dev/null)"       "active"
check "ensemble-predictor"      "$(systemctl is-active quantum-ensemble-predictor 2>/dev/null)"    "active"
check "layer4 optimizer STOPPED" "$(systemctl is-active quantum-layer4-portfolio-optimizer 2>/dev/null)" "inactive"
check "Redis PING"              "$(redis-cli PING 2>/dev/null)"                                    "PONG"
check "harvest_v2 LIVE mode"    "$(redis-cli HGET quantum:config:harvest_v2 stream_live 2>/dev/null)" "apply.plan"
check "Kelly BNBUSDT 200"       "$(redis-cli HGET quantum:layer4:sizing:BNBUSDT size_usdt 2>/dev/null)" "200.0"
check "SYMBOLS=12"              "$(grep '^SYMBOLS=' /etc/quantum/apply-layer.env | tr ',' '\n' | wc -l | tr -d ' ')" "12"
check "RISK_DAILY_LOSS=-150"    "$(grep 'RISK_DAILY_LOSS' /etc/quantum/apply-layer.env)"            "150"
check "harvest.v2.shadow active" "$(redis-cli XLEN quantum:stream:harvest.v2.shadow 2>/dev/null)"  "[0-9]"
check "apply.result active"     "$(redis-cli XLEN quantum:stream:apply.result 2>/dev/null)"        "[0-9]"

echo ""
echo "============================================================"
echo " TOTAL SYSTEM DIAGNOSTIC COMPLETE"
echo " $TS"
echo "============================================================"
