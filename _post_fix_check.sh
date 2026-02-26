#!/bin/bash
echo "=== PORTFOLIO STATE post-close ==="
redis-cli hgetall quantum:state:portfolio | grep -E "positions_count|balance|equity|unrealized|pnl"

echo ""
echo "=== SLOT FIX CODE ==="
grep -r "authoritative_count\|SLOT_FIX\|authoritative" /home/qt/quantum_trader/microservices/autonomous_trader/ 2>/dev/null | grep -v ".pyc" | grep -v "__pycache__" | head -20

echo ""
echo "=== AUTONOMOUS TRADER LAST 8 LOGS ==="
journalctl -u quantum-autonomous-trader -n 8 --no-pager

echo ""
echo "=== POSITION HASHES STATUS SAMPLE ==="
for key in $(redis-cli keys "quantum:position:*" | head -5); do
  echo "--- $key ---"
  redis-cli hget "$key" status
done

echo ""
echo "=== INTENT EXECUTOR LAST 3 ==="
journalctl -u quantum-intent-executor -n 3 --no-pager
