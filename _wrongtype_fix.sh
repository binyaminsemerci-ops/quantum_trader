#!/bin/bash
echo "=== Find WRONGTYPE keys in permit namespace ==="
# Check all p33 permit keys — they should be HASH type
wrong_count=0
for key in $(redis-cli KEYS 'quantum:permit:p33:*' 2>/dev/null | head -100); do
    ktype=$(redis-cli TYPE "$key" 2>/dev/null)
    if [ "$ktype" != "hash" ] && [ "$ktype" != "none" ]; then
        echo "  WRONG TYPE: $key => $ktype (expected hash)"
        wrong_count=$((wrong_count + 1))
    fi
done
echo "Wrong-type permit keys found: $wrong_count"

echo ""
echo "=== Intent executor source: what Redis ops near WRONGTYPE? ==="
grep -n "WRONGTYPE\|hset\|sadd\|hmset\|hget\|lpush\|rpush\|smembers" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -30

echo ""
echo "=== Intent executor: find allow/permit check lines ==="
grep -n "permit\|allow\|p33\|p35\|guard" \
    /opt/quantum/microservices/intent_executor/main.py 2>/dev/null | head -25

echo ""
echo "=== Check key types for known permit keys ==="
for key in quantum:permit:governor quantum:permit:p33 quantum:permit:p26 quantum:permit:p35; do
    ktype=$(redis-cli TYPE "$key" 2>/dev/null)
    echo "  $key => $ktype"
done

echo ""
echo "=== Check trade.intent stream type and recent entries ==="
redis-cli TYPE quantum:stream:trade.intent
redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 2 2>/dev/null | head -20

echo ""
echo "=== Check DONE key types (should be STRING) ==="
for key in $(redis-cli KEYS 'quantum:intent_executor:done:*' 2>/dev/null | head -20); do
    ktype=$(redis-cli TYPE "$key" 2>/dev/null)
    if [ "$ktype" != "string" ]; then
        echo "  WRONG: $key => $ktype"
    fi
done
echo "(done key scan complete)"

echo ""
echo "=== Check quantum:stream:signal.score consumer groups ==="
redis-cli XINFO GROUPS quantum:stream:signal.score 2>/dev/null | head -20

echo ""
echo "=== WRONGTYPE fix: restart intent executor ==="
systemctl restart quantum-intent-executor.service 2>/dev/null
sleep 3
echo "Intent executor status: $(systemctl is-active quantum-intent-executor.service 2>/dev/null)"
journalctl -u quantum-intent-executor.service -n 8 --no-pager 2>/dev/null | tail -10
