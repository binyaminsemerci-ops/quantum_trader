#!/bin/bash
echo "=== intent_bridge service definition ==="
systemctl cat quantum-intent-bridge 2>/dev/null | grep -E "ExecStart|WorkingDir|User"

echo ""
echo "=== harvest_brain service definition ==="
systemctl cat quantum-harvest-brain 2>/dev/null | grep -E "ExecStart|WorkingDir|User"

echo ""
echo "=== Checking patched files are actually loaded (verify H1 fix in running files) ==="
echo "intent_bridge line 838:"
sed -n '836,840p' /opt/quantum/microservices/intent_bridge/main.py
echo "harvest_brain line 940:"
sed -n '938,942p' /opt/quantum/microservices/harvest_brain/harvest_brain.py

echo ""
echo "=== Key age: sample a recent permit key and check its hash fields ==="
KEY=$(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -1)
echo "Key: $KEY"
echo "TTL: $(redis-cli TTL "$KEY")"
redis-cli HGETALL "$KEY" 2>/dev/null
