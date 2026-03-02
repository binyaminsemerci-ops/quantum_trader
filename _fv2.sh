#!/bin/bash
echo "=== FINAL VERIFICATION ==="

echo ""
echo "--- P3 Ghost Slots ---"
redis-cli lrange quantum:stream:harvest.tick 0 0 | python3 -c "
import sys, json
raw = sys.stdin.read().strip()
if raw:
    d = json.loads(raw)
    print('scanned=' + str(d.get('scanned','?')) + ' evaluated=' + str(d.get('evaluated','?')) + ' skipped_invalid=' + str(d.get('skipped_invalid','?')))
else:
    print('no_data')
"

echo ""
echo "--- P4 Cooldown TTLs ---"
for k in $(redis-cli keys "quantum:cooldown:*" | head -5); do
  ttl=$(redis-cli ttl "$k")
  echo "  $k TTL=$ttl"
done

echo ""
echo "--- P5 unknown_variant in apply-layer (new PID only) ---"
NEW_AL=$(systemctl show quantum-apply-layer --property=MainPID | cut -d= -f2)
echo "apply-layer PID=$NEW_AL"
journalctl -u quantum-apply-layer -n 300 --no-pager 2>&1 | grep unknown_variant | tail -3

echo ""
echo "--- P5 BNBUSDT last decision ---"
journalctl -u quantum-apply-layer -n 300 --no-pager 2>&1 | grep BNBUSDT | grep decision | tail -3

echo ""
echo "--- P6 WRONGTYPE count ---"
count=$(journalctl -u quantum-intent-executor -n 300 --no-pager 2>&1 | grep WRONGTYPE | wc -l)
echo "WRONGTYPE_COUNT=$count"

echo ""
echo "--- Services ---"
for s in quantum-harvest-v2 quantum-intent-executor quantum-apply-layer quantum-position-state-brain; do
  echo "$s=$(systemctl is-active $s)"
done

echo ""
echo "=== DONE ==="
