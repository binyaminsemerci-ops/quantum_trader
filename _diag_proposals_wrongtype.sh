#!/bin/bash
echo "=== Proposal full content ==="
for k in $(redis-cli KEYS "quantum:harvest:proposal:*" 2>/dev/null); do
    echo "--- $k ---"
    redis-cli HGETALL "$k"
    echo ""
done

echo ""
echo "=== WRONGTYPE errors with timestamps (last 10 min) ==="
journalctl -u quantum-intent-executor.service --since "10 minutes ago" --no-pager 2>/dev/null \
  | grep -E "WRONGTYPE|processing plan|Error.*permit" | tail -30

echo ""
echo "=== intent-executor: what key causes WRONGTYPE? ==="
journalctl -u quantum-intent-executor.service --since "5 minutes ago" --no-pager 2>/dev/null \
  | grep -B5 "WRONGTYPE" | grep -E "plan|symbol|key|permit" | tail -20

echo ""
echo "=== Check intent-executor main.py line around permit (home/qt) ==="
grep -n "hgetall\|permit\|WRONGTYPE" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -20

echo ""
echo "=== ADAUSDT: Still being executed? ==="
journalctl -u quantum-intent-executor.service --since "2 minutes ago" --no-pager 2>/dev/null \
  | grep -E "ADA|ADAUSDT" | tail -10

echo ""
echo "=== AVAXUSDT cooldown (should have TTL) ==="
redis-cli TTL "quantum:cooldown:last_exec_ts:AVAXUSDT"
redis-cli TTL "quantum:cooldown:last_exec_ts:ADAUSDT"
