#!/bin/bash
echo "=== apply_layer/main.py permit key creation (lines 210-240) ==="
sed -n '210,240p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== lines 520-545 ==="
sed -n '520,545p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== lines 1530-1545 ==="
sed -n '1530,1545p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== AI engine status after restart ==="
systemctl is-active quantum-ai-engine

echo ""
echo "=== Permit key count ==="
COUNT=0
for key in $(redis-cli --scan --pattern "quantum:permit:*" --count 500 2>/dev/null | head -100); do
  COUNT=$((COUNT+1))
done
echo "Sample of permit keys: $COUNT (stopped at 100)"

echo ""
echo "=== also check home path apply_layer ==="
sed -n '210,240p' /home/qt/quantum_trader/microservices/apply_layer/main.py 2>/dev/null | head -5
