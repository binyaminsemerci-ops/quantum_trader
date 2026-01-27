#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%d_%H%M%S)"
DIR="/tmp/exit_proof_${TS}"
mkdir -p "$DIR"/{logs,proof}

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "========================================="
log "PHASE 2: EXIT HARD PROOF (3 ROUNDS)"
log "========================================="

# Get allowed symbols from executor env
ALLOWED_SYMBOLS=$(grep "^INTENT_EXECUTOR_ALLOWLIST=" /etc/quantum/intent-executor.env | cut -d= -f2)
log "Allowed symbols: $ALLOWED_SYMBOLS"

# Stop intent bridge to reduce noise
log "Stopping intent-bridge to reduce OPEN flood..."
systemctl stop quantum-intent-bridge 2>/dev/null || true
sleep 2

# Fetch positions and filter to allowed symbols
log "Fetching positions from Binance testnet..."
python3 > "$DIR/proof/positions.json" << PYEOF
import os, time, hmac, hashlib, json, urllib.request, urllib.parse

# Load env
env_file="/etc/quantum/apply-layer.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v=line.split("=",1)
            os.environ[k]=v.strip()

api_key=os.getenv("BINANCE_TESTNET_API_KEY","")
api_secret=os.getenv("BINANCE_TESTNET_API_SECRET","")
if not api_key or not api_secret:
    print(json.dumps({"error":"Missing BINANCE_TESTNET_API_KEY/SECRET"}))
    exit(1)

def sign(qs):
    return hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

ts=int(time.time()*1000)
qs=f"timestamp={ts}"
sig=sign(qs)
url=f"https://testnet.binancefuture.com/fapi/v2/positionRisk?{qs}&signature={sig}"
req=urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
data=json.loads(urllib.request.urlopen(req, timeout=10).read())

# Filter to allowed symbols and non-zero positions
allowed=set("$ALLOWED_SYMBOLS".split(","))
positions=[]
for p in data:
    if p["symbol"] not in allowed:
        continue
    amt=float(p.get("positionAmt","0") or 0)
    if amt!=0:
        positions.append({
            "symbol": p["symbol"],
            "positionAmt": amt,
            "side": "LONG" if amt>0 else "SHORT"
        })

positions.sort(key=lambda x: abs(x["positionAmt"]), reverse=True)
print(json.dumps({"positions": positions[:3]}, indent=2))
PYEOF

if grep -q "error" "$DIR/proof/positions.json"; then
    log "❌ Failed to fetch positions"
    cat "$DIR/proof/positions.json"
    systemctl start quantum-intent-bridge 2>/dev/null || true
    exit 1
fi

POS_COUNT=$(jq -r ".positions | length" "$DIR/proof/positions.json" 2>/dev/null || echo "0")
log "Found $POS_COUNT positions in allowed symbols"

if [ "$POS_COUNT" -eq 0 ]; then
    log "⚠️  No open positions in allowed symbols - cannot run EXIT proof"
    log "Creating artificial positions for proof..."
    
    # Create small artificial positions for proof (very small amounts)
    for symbol in ETHUSDT BTCUSDT TRXUSDT; do
        log "Opening small position: $symbol"
        python3 << PYEOF
import os, time, hmac, hashlib, json, urllib.request, urllib.parse

env_file="/etc/quantum/apply-layer.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line=line.strip()
            if not line or "=" not in line:
                continue
            k,v=line.split("=",1)
            os.environ[k]=v.strip()

api_key=os.getenv("BINANCE_TESTNET_API_KEY","")
api_secret=os.getenv("BINANCE_TESTNET_API_SECRET","")
base_url="https://testnet.binancefuture.com"

def sign(qs):
    return hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

# Very small qty for proof
symbol="$symbol"
qty=0.001 if symbol!="TRXUSDT" else 10

params=f"symbol={symbol}&side=BUY&type=MARKET&quantity={qty}&timestamp={int(time.time()*1000)}"
sig=sign(params)
url=f"{base_url}/fapi/v1/order?{params}&signature={sig}"
req=urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": api_key})
try:
    resp=urllib.request.urlopen(req, timeout=10)
    data=json.loads(resp.read())
    print(f"Opened {symbol}: {data.get('orderId')}")
except Exception as e:
    print(f"Failed to open {symbol}: {e}")
PYEOF
        sleep 1
    done
    
    sleep 5
    
    # Re-fetch positions
    log "Re-fetching positions..."
    python3 > "$DIR/proof/positions.json" << PYEOF2
import os, time, hmac, hashlib, json, urllib.request, urllib.parse

env_file="/etc/quantum/apply-layer.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k,v=line.split("=",1)
            os.environ[k]=v.strip()

api_key=os.getenv("BINANCE_TESTNET_API_KEY","")
api_secret=os.getenv("BINANCE_TESTNET_API_SECRET","")

def sign(qs):
    return hmac.new(api_secret.encode(), qs.encode(), hashlib.sha256).hexdigest()

ts=int(time.time()*1000)
qs=f"timestamp={ts}"
sig=sign(qs)
url=f"https://testnet.binancefuture.com/fapi/v2/positionRisk?{qs}&signature={sig}"
req=urllib.request.Request(url, headers={"X-MBX-APIKEY": api_key})
data=json.loads(urllib.request.urlopen(req, timeout=10).read())

allowed=set("$ALLOWED_SYMBOLS".split(","))
positions=[]
for p in data:
    if p["symbol"] not in allowed:
        continue
    amt=float(p.get("positionAmt","0") or 0)
    if amt!=0:
        positions.append({
            "symbol": p["symbol"],
            "positionAmt": amt,
            "side": "LONG" if amt>0 else "SHORT"
        })

positions.sort(key=lambda x: abs(x["positionAmt"]), reverse=True)
print(json.dumps({"positions": positions[:3]}, indent=2))
PYEOF2
    
    POS_COUNT=$(jq -r ".positions | length" "$DIR/proof/positions.json" 2>/dev/null || echo "0")
    if [ "$POS_COUNT" -eq 0 ]; then
        log "❌ Still no positions - aborting"
        systemctl start quantum-intent-bridge 2>/dev/null || true
        exit 0
    fi
fi

log "Positions to close:"
jq -r '.positions[] | "\(.symbol) \(.side) amt=\(.positionAmt)"' "$DIR/proof/positions.json"

# Inject CLOSE plans for each position
round=1
jq -r '.positions[] | "\(.symbol) \(.positionAmt) \(.side)"' "$DIR/proof/positions.json" | while read -r symbol pos_amt side; do
    if [ "$side" = "LONG" ]; then
        close_side="SELL"
    else
        close_side="BUY"
    fi
    
    # 50% of position for proof, min 0.001
    qty=$(python3 -c "amt=abs(float('$pos_amt')); q=max(amt*0.5, 0.001); print(f'{q:.6f}'.rstrip('0').rstrip('.'))")
    
    plan_id="exit_hardproof_${symbol}_R${round}_${TS}"
    
    log "Round $round: Injecting CLOSE $symbol $close_side qty=$qty"
    
    redis-cli XADD quantum:stream:apply.plan "*" \
        plan_id "$plan_id" \
        action "FULL_CLOSE_PROPOSED" \
        decision "EXECUTE" \
        symbol "$symbol" \
        side "$close_side" \
        type "MARKET" \
        qty "$qty" \
        reduceOnly "true" \
        source "intent_bridge" \
        timestamp "$(date +%s)" > /dev/null
    
    echo "$plan_id $symbol $close_side $qty" >> "$DIR/proof/injected_plans.txt"
    
    # Wait for result
    log "Waiting for result (max 60s)..."
    found=0
    for i in {1..60}; do
        if redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 400 | grep -q "$plan_id"; then
            redis-cli --raw XREVRANGE quantum:stream:apply.result + - COUNT 400 | grep -B3 -A25 "$plan_id" | head -30 > "$DIR/proof/result_R${round}_${symbol}.txt"
            log "✅ Result captured for round $round"
            found=1
            break
        fi
        sleep 1
    done
    
    if [ "$found" -eq 0 ]; then
        log "⚠️  No result found for round $round within timeout"
    fi
    
    round=$((round+1))
    sleep 3
done

# Capture logs
journalctl -u quantum-governor --since "5 minutes ago" --no-pager > "$DIR/logs/governor_5min.txt"
journalctl -u quantum-intent-executor --since "5 minutes ago" --no-pager > "$DIR/logs/executor_5min.txt"

# Restart intent bridge
log "Restarting intent-bridge..."
systemctl start quantum-intent-bridge 2>/dev/null || true

log "✅ EXIT proof complete"
echo "$DIR" > /tmp/exit_proof_last_dir.txt

# Print summary
log "========================================="
log "SUMMARY"
log "========================================="
if [ -f "$DIR/proof/injected_plans.txt" ]; then
    while read -r plan_id symbol side qty; do
        round=$(echo "$plan_id" | grep -o "R[0-9]" || echo "R?")
        result_file=$(ls "$DIR/proof/result_${round}_"* 2>/dev/null | grep -i "$symbol" | head -1)
        if [ -n "$result_file" ] && [ -f "$result_file" ]; then
            executed=$(grep "executed" "$result_file" | head -1 | cut -d: -f2 | tr -d ' ",' || echo "?")
            order_id=$(grep "order_id" "$result_file" | head -1 | cut -d: -f2 | tr -d ' ",' || echo "?")
            status=$(grep "order_status" "$result_file" | head -1 | cut -d: -f2 | tr -d ' ",' || echo "?")
            log "$round ✅ $symbol $side qty=$qty | executed=$executed order_id=$order_id status=$status"
        else
            log "$round ❌ $symbol $side qty=$qty | NO RESULT"
        fi
    done < "$DIR/proof/injected_plans.txt"
else
    log "No plans were injected"
fi

log "Artifacts saved in: $DIR"
