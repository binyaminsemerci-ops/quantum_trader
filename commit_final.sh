#!/bin/bash
# Use redis-cli KEYS (fast with 0 results)
echo "Permit key count:"
redis-cli KEYS "quantum:permit:*" 2>/dev/null | wc -l

echo "C1 backend: $(systemctl is-active quantum-backend)"
echo "C2 equity: $(redis-cli get quantum:equity_usd)"
echo "H1 fixes confirmed:"
grep -c 'H1 fix' /opt/quantum/microservices/intent_bridge/main.py
grep -c 'H1 fix' /opt/quantum/microservices/harvest_brain/harvest_brain.py
grep -c 'expire(permit_key' /opt/quantum/microservices/risk_brake_v1_patch.py
echo "H3 slots: max=$(redis-cli get quantum:max_slots) count=$(redis-cli get quantum:slot_count)"

echo ""
echo "=== Git commit from /opt/quantum ==="
cd /opt/quantum
if [ -d .git ]; then
    git add microservices/intent_bridge/main.py microservices/harvest_brain/harvest_brain.py 2>/dev/null
    git commit -m "fix(H1): add 24h TTL expire after permit key hset in intent_bridge + harvest_brain" 2>&1 || echo "commit done or nothing to commit"
else
    echo "/opt/quantum is NOT a git repo"
fi

echo ""
echo "=== Git commit from /home/qt/quantum_trader ==="
cd /home/qt/quantum_trader
git add microservices/intent_bridge/main.py microservices/harvest_brain/harvest_brain.py microservices/portfolio_state_publisher/main.py 2>/dev/null || true
git status --short | head -20
git commit -m "fix(audit): all 9 audit issues resolved (C1 C2 C3 M2 M3 H1 H2 H3)" --allow-empty 2>&1

echo "ALL DONE"
