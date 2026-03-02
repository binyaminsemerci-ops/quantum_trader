#!/bin/bash
# Step 1: Run H1 TTL fix
echo "=== H1: Patching permit key writers ==="
python3 /tmp/fix_h1_ttl_v2.py

# Step 2: Verify patches applied
echo ""
echo "=== VERIFY: Check expire added to intent_bridge ==="
grep -n "expire(permit_key" /opt/quantum/microservices/intent_bridge/main.py || echo "MISSING in intent_bridge"
grep -n "expire(permit_key" /opt/quantum/microservices/harvest_brain/harvest_brain.py || echo "MISSING in harvest_brain"
grep -n "expire(permit_key" /opt/quantum/microservices/risk_brake_v1_patch.py || echo "MISSING in risk_brake"

echo ""
echo "=== RESTART: Restarting affected services ==="
systemctl restart quantum-intent-bridge 2>/dev/null && echo "intent-bridge restarted" || echo "quantum-intent-bridge not found, checking..."
systemctl restart quantum-harvest-brain 2>/dev/null && echo "harvest-brain restarted" || echo "quantum-harvest-brain not found, checking..."
systemctl restart quantum-risk-brake 2>/dev/null && echo "risk-brake restarted" || echo "quantum-risk-brake not found"

# Check for alternative service names
for svc in intent-bridge intent_bridge harvest-brain harvest_brain risk-brake risk_brake; do
    state=$(systemctl is-active quantum-$svc 2>/dev/null)
    [ "$state" = "active" ] && echo "quantum-$svc: active"
done

echo ""
echo "=== H1: Permit key count after restart ==="
sleep 3
redis-cli --scan --pattern "quantum:permit:*" 2>/dev/null | wc -l

echo ""
echo "=== FINAL: Service status ==="
systemctl is-active quantum-backend && echo "backend: OK"
systemctl is-active quantum-ai-engine && echo "ai-engine: OK"
systemctl is-active quantum-portfolio-state-publisher 2>/dev/null && echo "portfolio-publisher: OK"

echo ""
echo "=== DONE ==="
