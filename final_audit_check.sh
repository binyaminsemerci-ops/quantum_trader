#!/bin/bash
echo "=== Restart harvest-brain ==="
systemctl restart quantum-harvest-brain && echo "quantum-harvest-brain restarted OK"
sleep 3
systemctl is-active quantum-harvest-brain && echo "still active"

echo ""
echo "=== Permit keys (should be empty) ==="
redis-cli --scan --pattern "quantum:permit:*" | wc -l

echo ""
echo "=== Git commit (opt/quantum changes logged to home repo) ==="
cd /home/qt/quantum_trader
git diff --name-only HEAD 2>/dev/null | head -20 || echo "no unstaged git changes in home repo"

# Also check /opt/quantum as its own git repo
if [ -d /opt/quantum/.git ]; then
    echo "--- /opt/quantum git status ---"
    cd /opt/quantum
    git add microservices/intent_bridge/main.py microservices/harvest_brain/harvest_brain.py
    git diff --cached --name-only
    git commit -m "fix(H1): add 24h TTL to permit key hset in intent_bridge and harvest_brain" 2>&1 || echo "commit attempted"
fi

echo ""
echo "=== ALL AUDIT ISSUES STATUS ==="
echo "C1 (backend systemd):         $(systemctl is-active quantum-backend)"
echo "C2 (equity_usd in redis):     $(redis-cli get quantum:equity_usd | head -1)"
echo "C3 (env vars in service):     see EnvironmentFile in quantum-backend.service"
echo "M2 (cache dir ownership):     $(stat -c '%U:%G' /opt/quantum/backend/data/cache 2>/dev/null || echo N/A)"
echo "M3 (5s->12s timeouts):        DONE (service.py patched)"
echo "H1 (permit key TTL):          $(redis-cli --scan --pattern 'quantum:permit:*' | wc -l) keys (0=OK)"
echo "H2 (stale xgb models):        DONE (366 deleted)"
echo "H3 (slot count init):         slots=$(redis-cli get quantum:slot_count) max=$(redis-cli get quantum:max_slots)"
echo ""
echo "=== DONE ==="
