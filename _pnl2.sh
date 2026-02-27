#!/bin/bash
echo "=== INTENT-EXECUTOR: HARVEST timeline siste time ==="
echo ""
echo "--- FØR FIX (401 errors) ---"
journalctl -u quantum-intent-executor --since "1 hour ago" --no-pager \
  | grep -E "HARVEST SUCCESS|Unauthorized|HARVEST SKIP|HARVEST CLOSE|Published trade" \
  | head -60

echo ""
echo "--- HARVEST_EXECUTED metric økning ---"
journalctl -u quantum-intent-executor --since "1 hour ago" --no-pager \
  | grep "harvest_executed=" | awk '{for(i=1;i<=NF;i++) if($i~/harvest_executed=/) print $1,$2,$i}' \
  | tail -5

echo ""
echo "=== TRADE.CLOSED STREAM — alle handler ==="
python3 - << 'EOF'
import redis, json
from datetime import datetime, timezone, timedelta

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
min_id = str(int(one_hour_ago.timestamp() * 1000)) + "-0"

entries = r.xrange("quantum:stream:trade.closed", min=min_id, max="+")
print(f"Antall trade.closed events: {len(entries)}")

total_usd = 0.0
wins = 0
losses = 0

for eid, fields in entries:
    ts = datetime.fromtimestamp(int(eid.split('-')[0])/1000, tz=timezone.utc)
    payload = fields.get("payload", "{}")
    try:
        d = json.loads(payload)
    except:
        d = fields

    sym    = d.get("symbol", "?")
    side   = d.get("side", "?")
    pnl    = d.get("pnl_usd", d.get("realized_pnl_usd", "?"))
    pnl_p  = d.get("pnl_pct", "?")
    r_val  = d.get("r_multiple", d.get("R", "?"))
    src    = d.get("source", fields.get("source", "?"))
    
    print(f"  {ts.strftime('%H:%M:%S')}  {sym:12}  {side:5}  PnL=${str(pnl):8}  %={str(pnl_p):8}  R={str(r_val):6}  src={src}")
    
    try:
        total_usd += float(str(pnl).replace('$',''))
        if float(str(pnl_p).replace('%','')) >= 0:
            wins += 1
        else:
            losses += 1
    except:
        pass

print(f"\n  Total realized USD: ${total_usd:.2f}")
print(f"  Wins: {wins}  Losses: {losses}")
EOF

echo ""
echo "=== STREAM INFO ==="
redis-cli xinfo stream quantum:stream:trade.closed 2>/dev/null | grep -E "length|last-generated"

echo ""
echo "=== ALLE TRADE.CLOSED SISTE 24T ==="
python3 - << 'EOF'
import redis, json
from datetime import datetime, timezone, timedelta

r = redis.Redis(host='localhost', port=6379, decode_responses=True)
since = datetime.now(timezone.utc) - timedelta(hours=24)
min_id = str(int(since.timestamp() * 1000)) + "-0"

entries = r.xrange("quantum:stream:trade.closed", min=min_id, max="+")
print(f"Totalt siste 24t: {len(entries)} handler\n")

total_usd = 0.0

for eid, fields in entries:
    ts = datetime.fromtimestamp(int(eid.split('-')[0])/1000, tz=timezone.utc)
    payload = fields.get("payload", "{}")
    try:
        d = json.loads(payload)
    except:
        d = fields

    sym    = d.get("symbol", "?")
    side   = d.get("side", "?")
    pnl    = d.get("pnl_usd", d.get("realized_pnl_usd", "?"))
    pnl_p  = d.get("pnl_pct", "?")
    r_val  = d.get("r_multiple", d.get("R", "?"))
    
    print(f"  {ts.strftime('%m-%d %H:%M')}  {sym:12}  {side:5}  PnL=${str(pnl):8}  %={str(pnl_p):8}  R={str(r_val)}")
    
    try:
        total_usd += float(str(pnl).replace('$',''))
    except:
        pass

print(f"\n  Total realized USD siste 24t: ${total_usd:.2f}")
EOF
