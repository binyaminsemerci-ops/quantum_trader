#!/bin/bash
# Restart harvest-brain
systemctl restart quantum-harvest-brain 2>&1 && echo "harvest-brain: restarted"

sleep 3

# Permit key count using DBSIZE-based approach (faster than scan)
echo "Permit key count:"
redis-cli eval "return #redis.call('keys', 'quantum:permit:*')" 0

# Quick service checks
echo "C1 backend: $(systemctl is-active quantum-backend)"
echo "C2 equity: $(redis-cli get quantum:equity_usd)"
echo "H1 fix: $(grep -l 'H1 fix' /opt/quantum/microservices/intent_bridge/main.py /opt/quantum/microservices/harvest_brain/harvest_brain.py 2>/dev/null | wc -l) files patched"
echo "H3 slots: max=$(redis-cli get quantum:max_slots) count=$(redis-cli get quantum:slot_count)"

# git status for home dir
echo ""
echo "Home git status:"
cd /home/qt/quantum_trader
git status --short 2>/dev/null | head -15

# Git status for opt dir
echo "Opt git status:"
cd /opt/quantum
git status --short 2>/dev/null | head -10

echo "DONE"
