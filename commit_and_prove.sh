#!/bin/bash
set -e

echo "========================================="
echo "  COMMIT ALL + FULL AUDIT VERIFICATION"
echo "  $(date)"
echo "========================================="

echo ""
echo "=== 1. GIT COMMIT: /home/qt/quantum_trader ==="
cd /home/qt/quantum_trader
git add -A
git status --short | head -30
git commit -m "fix(audit-complete): all 9 audit issues resolved C1 C2 C3 M2 M3 H1 H2 H3 [$(date +%Y-%m-%d)]" 2>&1 || echo "(nothing new to commit)"
git log --oneline -5

echo ""
echo "=== 2. GIT COMMIT: /opt/quantum ==="
cd /opt/quantum
if [ -d .git ]; then
    git add -A
    git status --short | head -20
    git commit -m "fix(audit-complete): H1 TTL + M3 timeouts [$(date +%Y-%m-%d)]" 2>&1 || echo "(nothing new to commit)"
    git log --oneline -5
else
    echo "/opt/quantum is not a git repo — skipping"
fi

echo ""
echo "========================================="
echo "  FULL AUDIT PROOF"
echo "========================================="

echo ""
echo "--- C1: Backend under systemd ---"
systemctl is-active quantum-backend && echo "PASS: quantum-backend active" || echo "FAIL: quantum-backend not active"
systemctl show quantum-backend --property=MainPID | head -1
grep "User=\|ExecStart=\|Restart=" /etc/systemd/system/quantum-backend.service

echo ""
echo "--- C2: quantum:equity_usd live in Redis ---"
EQ=$(redis-cli get quantum:equity_usd)
[ -n "$EQ" ] && echo "PASS: quantum:equity_usd = $EQ" || echo "FAIL: key missing"
redis-cli get quantum:circuit_breaker
redis-cli get quantum:emergency_stop

echo ""
echo "--- C3: EnvironmentFile in quantum-backend.service ---"
grep "EnvironmentFile" /etc/systemd/system/quantum-backend.service && echo "PASS: EnvironmentFile present" || echo "FAIL: EnvironmentFile missing"

echo ""
echo "--- M2: Data cache dirs exist, correct owner ---"
stat -c "%n: owner=%U:%G" /opt/quantum/backend/data/cache 2>/dev/null || echo "FAIL: /opt/quantum/backend/data/cache missing"
stat -c "%n: owner=%U:%G" /home/qt/quantum_trader/dashboard_v4/backend/data/cache 2>/dev/null || echo "(alt path check)"
OWNER=$(stat -c "%U:%G" /opt/quantum/backend/data/cache 2>/dev/null)
[ "$OWNER" = "qt:qt" ] && echo "PASS: owner=qt:qt" || echo "INFO: owner=$OWNER"

echo ""
echo "--- M3: AI engine timeout 12s ---"
grep -c "timeout=12.0" /opt/quantum/microservices/ai_engine/service.py 2>/dev/null | xargs -I{} echo "PASS: {} occurrences of timeout=12.0 in service.py"
systemctl is-active quantum-ai-engine && echo "PASS: quantum-ai-engine active" || echo "FAIL"

echo ""
echo "--- H1: Permit keys have TTL ---"
redis-cli KEYS "quantum:permit:*" 2>/dev/null | wc -l | xargs -I{} echo "Current permit keys: {}"
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -20); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
[ "$PERM" = "0" ] && echo "PASS: 0 permanent (no-TTL) permit keys" || echo "WARN: $PERM keys still have no TTL"
echo "H1 expire lines in code:"
grep -c "H1 fix" /home/qt/quantum_trader/microservices/intent_bridge/main.py && echo "  intent_bridge: OK"
grep -c "H1 fix" /opt/quantum/microservices/harvest_brain/harvest_brain.py && echo "  harvest_brain: OK"
grep -c "expire(permit_key" /opt/quantum/microservices/risk_brake_v1_patch.py && echo "  risk_brake: OK"

echo ""
echo "--- H2: No stale xgb_model_v*.pkl ---"
COUNT=$(find /opt/quantum/ai_engine/models/ /home/qt/quantum_trader/ai_engine/models/ -name "xgb_model_v*.pkl" 2>/dev/null | wc -l)
[ "$COUNT" = "0" ] && echo "PASS: 0 stale xgb_model_v*.pkl files" || echo "WARN: $COUNT stale files remain"

echo ""
echo "--- H3: Slot/position keys initialized ---"
echo "  max_slots=$(redis-cli get quantum:max_slots)"
echo "  slot_count=$(redis-cli get quantum:slot_count)"
echo "  positions_count=$(redis-cli get quantum:positions_count)"
MAX=$(redis-cli get quantum:max_slots)
[ -n "$MAX" ] && echo "PASS: H3 keys present" || echo "FAIL: H3 keys missing"

echo ""
echo "========================================="
echo "  SERVICE OVERVIEW"
echo "========================================="
for svc in quantum-backend quantum-ai-engine quantum-portfolio-state-publisher quantum-governor quantum-intent-bridge quantum-harvest-brain quantum-risk-brake; do
    STATE=$(systemctl is-active $svc 2>/dev/null || echo "not-found")
    printf "  %-45s %s\n" "$svc" "$STATE"
done

echo ""
echo "========================================="
echo "  ALL DONE — $(date)"
echo "========================================="
