#!/bin/bash
python3 /tmp/fix_home_ib.py

echo ""
echo "=== Verifying patch ==="
sed -n '868,882p' /home/qt/quantum_trader/microservices/intent_bridge/main.py

echo ""
echo "=== Restarting intent-bridge ==="
systemctl restart quantum-intent-bridge
sleep 3
systemctl is-active quantum-intent-bridge

echo ""
echo "=== Flushing old TTL=-1 permit keys ==="
redis-cli KEYS "quantum:permit:*" 2>/dev/null | xargs -r redis-cli DEL >/dev/null 2>&1
echo "Flushed permit keys"

sleep 5
echo "Permit key count after flush + wait:"
redis-cli KEYS "quantum:permit:*" 2>/dev/null | wc -l

echo ""
echo "=== Sampling new key TTLs ==="
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -5); do
    TTL=$(redis-cli TTL "$KEY")
    echo "  $KEY -> TTL=$TTL"
done

echo "DONE"
