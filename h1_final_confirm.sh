#!/bin/bash
echo "=== Current permit keys and their TTLs ==="
KEYS=$(redis-cli KEYS "quantum:permit:*" 2>/dev/null)
COUNT=$(echo "$KEYS" | grep -c "quantum" 2>/dev/null || echo 0)
echo "Total: $COUNT"
echo "$KEYS" | head -10 | while read KEY; do
    [ -z "$KEY" ] && continue
    TTL=$(redis-cli TTL "$KEY")
    REASON=$(redis-cli HGET "$KEY" reason 2>/dev/null || redis-cli GET "$KEY" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('source','?'))" 2>/dev/null || echo "?")
    echo "  $KEY: TTL=$TTL reason=$REASON"
done

echo ""
echo "=== Permanent keys count (TTL=-1) ==="
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
echo "Permanent (no TTL): $PERM"

echo ""
echo "=== Final git commit (home repo) ==="
cd /home/qt/quantum_trader
git add microservices/intent_bridge/main.py
git commit -m "fix(H1): add 24h TTL to intent_bridge permit key (home dir - actual running file)" 2>&1

echo "=== ALL DONE ==="
