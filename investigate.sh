#!/bin/bash
echo "=== Find 4th permit key writer ==="

# Sample a current TTL=-1 key and check its content for clues
echo "--- Sample TTL=-1 permit key contents ---"
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -20); do
    T=$(redis-cli TTL "$KEY")
    if [ "$T" = "-1" ]; then
        echo "Key: $KEY"
        redis-cli HGETALL "$KEY" 2>/dev/null | head -10
        echo "---"
        break
    fi
done

# Check all services that might exist beyond the 3 already patched
echo ""
echo "--- All hset permit_key across ALL files (not just microservices) ---"
grep -rn "hset.*permit_key\|hset(permit" /opt/quantum/ /home/qt/quantum_trader/ 2>/dev/null \
  | grep -v ".bak\|.pyc\|#" \
  | grep -v "H1 fix\|H1 TTL" \
  | grep -v "__pycache__"

echo ""
echo "--- All setex with permit_key (in case a new writer uses setex but short TTL) ---"
grep -rn "setex.*permit_key" /opt/quantum/ /home/qt/quantum_trader/ 2>/dev/null \
  | grep -v ".bak\|.pyc" | grep -v "__pycache__"

echo ""
echo "--- governor/main.py actual TTL ---"
grep -n "permit_key\|setex\|expire" /opt/quantum/microservices/governor/main.py | grep -i "permit" | head -10
grep -n "permit_key\|setex\|expire" /home/qt/quantum_trader/microservices/governor/main.py | grep -i "permit" | head -10

echo ""
echo "=== Redis key bloat — top key patterns ==="
# Sample 5000 random keys and count by prefix
redis-cli RANDOMKEY 2>/dev/null >/dev/null  # warm up
redis-cli KEYS "*" 2>/dev/null | sed 's/:[^:]*$//' | sed 's/:[^:]*$//' | sort | uniq -c | sort -rn | head -30
