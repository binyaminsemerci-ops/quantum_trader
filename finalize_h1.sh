#!/bin/bash
set -e

echo "=== Restarting intent-bridge and harvest-brain ==="

# Try both naming conventions
for svc in quantum-intent-bridge quantum-intentbridge quantum-intent_bridge; do
    if systemctl is-enabled "$svc" &>/dev/null; then
        systemctl restart "$svc" && echo "$svc restarted" && break
    fi
done

for svc in quantum-harvest-brain quantum-harvestbrain quantum-harvest_brain; do
    if systemctl is-enabled "$svc" &>/dev/null; then
        systemctl restart "$svc" && echo "$svc restarted" && break
    fi
done

sleep 4

echo ""
echo "=== Service states ==="
for svc in quantum-backend quantum-ai-engine quantum-portfolio-state-publisher; do
    state=$(systemctl is-active "$svc" 2>/dev/null || echo unknown)
    echo "$svc: $state"
done

# Try guessing intent/harvest service names
for svc in $(systemctl list-units --type=service --state=active | grep -i 'intent\|harvest\|governor\|risk' | awk '{print $1}'); do
    echo "$svc: active"
done

echo ""
echo "=== H1: Permit keys in Redis ==="
redis-cli --scan --pattern "quantum:permit:*" 2>/dev/null | wc -l

echo ""
echo "=== Final confirm: expire lines present ==="
grep -n "H1 fix" /opt/quantum/microservices/intent_bridge/main.py
grep -n "H1 fix" /opt/quantum/microservices/harvest_brain/harvest_brain.py
grep -n "expire(permit_key" /opt/quantum/microservices/risk_brake_v1_patch.py

echo ""
echo "=== Git commit ==="
cd /home/qt/quantum_trader
git add -A
git status --short | head -30
git commit -m "fix(audit): H1 add TTL to permit key writers (intent_bridge, harvest_brain)" --allow-empty
echo "Git commit done"
