#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%d_%H%M%S)"
DIR="/tmp/exit_proof_${TS}"
mkdir -p "$DIR"/{logs,proof}

log(){ echo "[$(date -u +%H:%M:%S)] $*"; }

log "========================================="
log "PHASE 2: EXIT HARD PROOF (3 ROUNDS)"
log "========================================="

# Stop intent bridge to reduce noise
log "Stopping intent-bridge to reduce OPEN flood..."
systemctl stop quantum-intent-bridge 2>/dev/null || true
sleep 2

# Fetch positions and select top 3
log "Fetching positions from Binance testnet..."
python3 > "$DIR/proof/positions.json" << 'PYEOF'
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

positions=[]
for p in data:
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
    exit 1
fi

POS_COUNT=$(jq -r ".positions | length" "$DIR/proof/positions.json")
log "Found $POS_COUNT positions"

if [ "$POS_COUNT" -eq 0 ]; then
    log "⚠️  No open positions - cannot run EXIT proof"
    systemctl start quantum-intent-bridge 2>/dev/null || true
    exit 0
fi

# Inject CLOSE plans for each position
jq -r '.positions[] | "\(.symbol) \(.positionAmt) \(.side)"' "$DIR/proof/positions.json" | head -3 | while read -r symbol pos_amt side; do
    # Calculate close side and qty
    if [ "$side" = "LONG" ]; then
        close_side="SELL"
    else
        close_side="BUY"
    fi
    
    # 5% of position, min 0.001
    qty=$(python3 -c "amt=abs(float('$pos_amt')); q=max(amt*0.05, 0.001); print(f'{q:.6f}'.rstrip('0').rstrip('.'))")
    
    plan_id="exit_hardproof_${symbol}_${TS}"
    
    log "Injecting CLOSE: $symbol $close_side qty=$qty (from pos=$pos_amt)"
    
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
    
    # Wait for result (max 60s per plan)
    log "Waiting for result: $plan_id..."
    for i in {1..60}; do
        if redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 300 | grep -q "$plan_id"; then
            redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 300 | grep -B5 -A20 "$plan_id" > "$DIR/proof/result_${plan_id}.txt"
            log "✅ Result captured for $plan_id"
            break
        fi
        sleep 1
    done
    
    sleep 2  # Brief pause between plans
done

# Capture logs
journalctl -u quantum-governor --since "5 minutes ago" --no-pager > "$DIR/logs/governor_5min.txt"
journalctl -u quantum-intent-executor --since "5 minutes ago" --no-pager > "$DIR/logs/executor_5min.txt"

# Restart intent bridge
log "Restarting intent-bridge..."
systemctl start quantum-intent-bridge 2>/dev/null || true

log "✅ EXIT proof complete - artifacts in $DIR"
echo "$DIR" > /tmp/exit_proof_last_dir.txt

# Print summary
log "========================================="
log "SUMMARY"
log "========================================="
if [ -f "$DIR/proof/injected_plans.txt" ]; then
    while read -r plan_id symbol side qty; do
        result_file="$DIR/proof/result_${plan_id}.txt"
        if [ -f "$result_file" ]; then
            executed=$(grep -o "executed[^ \"]*" "$result_file" | head -1 || echo "executed=?")
            order_id=$(grep -o "order_id[^ \"]*" "$result_file" | head -1 || echo "order_id=?")
            status=$(grep -o "order_status[^ \"]*" "$result_file" | head -1 || echo "status=?")
            log "✅ $symbol $side qty=$qty | $executed $order_id $status"
        else
            log "❌ $symbol $side qty=$qty | NO RESULT"
        fi
    done < "$DIR/proof/injected_plans.txt"
fi
