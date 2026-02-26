#!/bin/bash
echo "============================================================"
echo "SYSTEM AUDIT — $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "============================================================"

echo ""
echo "--- 1. QUANTUM SERVICES ---"
systemctl list-units --type=service --all --no-pager --plain --no-legend \
  | grep quantum \
  | awk '{s=($4=="running")?"✅":"❌"; printf "  %s %-52s %s\n", s, $1, $4}'

echo ""
echo "--- 2. REDIS KEY COUNTS ---"
for pattern in \
  "quantum:position:*" \
  "quantum:harvest:proposal:*" \
  "quantum:harvest:heat:*" \
  "quantum:harvest_v2:state:*" \
  "quantum:position:snapshot:*" \
  "quantum:state:positions:*"
do
  count=$(redis-cli --scan --pattern "$pattern" 2>/dev/null | wc -l)
  printf "  %-42s %3d keys\n" "$pattern" "$count"
done

echo ""
echo "--- 3. OPEN POSITIONS ---"
redis-cli --scan --pattern "quantum:position:*" | grep -v "snapshot\|backup\|phantom" | while read key; do
  sym=$(echo "$key" | awk -F: '{print $NF}')
  side=$(redis-cli hget "$key" side 2>/dev/null)
  qty=$(redis-cli hget "$key" quantity 2>/dev/null || redis-cli hget "$key" qty 2>/dev/null)
  entry=$(redis-cli hget "$key" entry_price 2>/dev/null)
  risk=$(redis-cli hget "$key" entry_risk_usdt 2>/dev/null)
  echo "  $sym  side=$side  qty=$qty  entry=$entry  risk_usdt=$risk"
done
POSCOUNT=$(redis-cli --scan --pattern "quantum:position:*" | grep -v "snapshot\|backup\|phantom" | wc -l)
[ "$POSCOUNT" -eq 0 ] && echo "  (none)"

echo ""
echo "--- 4. MANUAL LANE ---"
val=$(redis-cli get "quantum:manual_lane:enabled" 2>/dev/null)
ttl=$(redis-cli ttl "quantum:manual_lane:enabled" 2>/dev/null)
if [ -n "$val" ] && [ "$val" != "" ]; then
  hrs=$((ttl/3600)); mins=$(((ttl%3600)/60))
  echo "  ACTIVE ✅  value=$val  TTL=${ttl}s (~${hrs}h${mins}m)"
else
  echo "  INACTIVE ❌"
fi

echo ""
echo "--- 5. HARVEST PROPOSAL SYMBOLS ---"
grep SYMBOLS /etc/quantum/harvest-proposal.env

echo ""
echo "--- 6. EXECUTION PIPELINE (last 5 min) ---"
python3 - <<'PYEOF'
import redis, re, time
from collections import Counter
r = redis.Redis()
cutoff = (int(time.time()) - 300) * 1000
entries = r.xrange("quantum:stream:apply.result", min=str(cutoff))
decisions = Counter()
binance_errors = Counter()
for _, d in entries:
    dec = d.get(b"decision", b"None").decode()
    decisions[dec] += 1
    err = d.get(b"error", b"").decode()
    m = re.search(r'"code"\s*:\s*(-?[0-9]+)', err)
    if m:
        sym = d.get(b"symbol", b"?").decode()
        binance_errors[f"{sym} {m.group(1)}"] += 1
print(f"  Entries: {len(entries)}  Decisions: {dict(decisions.most_common(5))}")
if binance_errors:
    print(f"  ❌ Binance errors: {dict(binance_errors)}")
else:
    print(f"  ✅ Binance errors: ZERO")
PYEOF

echo ""
echo "--- 7. EXECUTION LOG (last 3 lines) ---"
tail -3 /var/log/quantum/execution.log 2>/dev/null | while IFS= read -r line; do
  echo "  $line"
done

echo ""
echo "--- 8. INTENT-EXECUTOR guard stats ---"
python3 - <<'PYEOF'
import redis
r = redis.Redis()
key = "quantum:intent_executor:stats"
if r.exists(key):
    stats = {k.decode(): v.decode() for k,v in r.hgetall(key).items()}
    for k in sorted(stats):
        print(f"  {k:<35} {stats[k]}")
else:
    # Try from log
    import subprocess
    res = subprocess.run(["grep", "-a", "guard_blocked\|p35\|executed_true", 
                         "/var/log/quantum/intent_executor.log"],
                        capture_output=True, text=True)
    for line in res.stdout.strip().splitlines()[-3:]:
        print(f"  {line.strip()[:100]}")
    if not res.stdout.strip():
        print("  (no stats key found)")
PYEOF

echo ""
echo "============================================================"
echo "AUDIT COMPLETE"
echo "============================================================"
