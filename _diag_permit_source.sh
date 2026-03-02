#!/bin/bash
echo "=== Lines 970-1020 in home/qt intent_executor (Get P3.3 permit area) ==="
sed -n '970,1030p' /home/qt/quantum_trader/microservices/intent_executor/main.py

echo ""
echo "=== Search for anything that SET (not HSET) permit keys ==="
grep -rn "permit:p33\|quantum:permit" /opt/quantum/microservices/ /home/qt/quantum_trader/microservices/ 2>/dev/null \
  | grep -v ".pyc" | grep -v "__pycache__" \
  | grep -vE "hset|hgetall|TYPE|comment|#" \
  | head -30

echo ""
echo "=== auto_permit_p33.py content ==="
cat /opt/quantum/scripts/auto_permit_p33.py

echo ""
echo "=== Search for 'redis.set.*permit' or 'r.set.*permit' ==="
grep -rn "\.set(" /opt/quantum/microservices/ /home/qt/quantum_trader/microservices/ 2>/dev/null \
  | grep -i "permit" | grep -v ".pyc" | grep -v "__pycache__" | head -20

echo ""
echo "=== Current STRING-type permits (new ones created after our fix) ==="
for k in $(redis-cli KEYS "quantum:permit:p33:*" 2>/dev/null); do
    T=$(redis-cli TYPE "$k")
    if [ "$T" = "string" ]; then
        echo "  STRING: $k VALUE=$(redis-cli GET $k)"
    fi
done
echo "  (None = no STRING permits = P6 fixed)"
