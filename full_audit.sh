#!/bin/bash
echo "========================================="
echo "  FULL SYSTEM AUDIT — $(date)"
echo "========================================="

# ---- C1: BACKEND SYSTEMD ----
echo ""
echo "=== C1: Backend under systemd ==="
systemctl is-active quantum-backend && echo "PASS: quantum-backend active" || echo "FAIL: quantum-backend not running"
systemctl show quantum-backend --property=MainPID,User,ExecStart,Restart | grep -v "^$"
echo "Uptime: $(ps -o etimes= -p $(systemctl show quantum-backend --property=MainPID --value) 2>/dev/null | awk '{printf "%d min\n", $1/60}' || echo unknown)"

# ---- C2: EQUITY_USD ----
echo ""
echo "=== C2: quantum:equity_usd live ==="
EQ=$(redis-cli get quantum:equity_usd 2>/dev/null)
[ -n "$EQ" ] && echo "PASS: quantum:equity_usd = $EQ" || echo "FAIL: key missing"
AGE=$(redis-cli object idletime quantum:equity_usd 2>/dev/null)
echo "  Key idle time: ${AGE}s (should be < 30s)"
echo "  circuit_breaker: $(redis-cli get quantum:circuit_breaker)"
echo "  emergency_stop:  $(redis-cli get quantum:emergency_stop)"

# ---- C3: ENV VARS ----
echo ""
echo "=== C3: EnvironmentFile in backend service ==="
grep "EnvironmentFile" /etc/systemd/system/quantum-backend.service && echo "PASS" || echo "FAIL: no EnvironmentFile"

# ---- M2: CACHE DIR ----
echo ""
echo "=== M2: Cache dirs exist with correct owner ==="
for DIR in /opt/quantum/backend/data/cache /home/qt/quantum_trader/dashboard_v4/backend/data/cache; do
    if [ -d "$DIR" ]; then
        OWNER=$(stat -c "%U:%G" "$DIR")
        [ "$OWNER" = "qt:qt" ] && echo "PASS: $DIR ($OWNER)" || echo "WARN: $DIR ($OWNER)"
    else
        echo "FAIL: missing $DIR"
    fi
done

# ---- M3: TIMEOUTS ----
echo ""
echo "=== M3: AI timeout=12.0 (was 5s) ==="
N=$(grep -c "timeout=12.0" /opt/quantum/microservices/ai_engine/service.py 2>/dev/null)
echo "  timeout=12.0 occurrences: $N (expect >=7)"
[ "$N" -ge 7 ] && echo "PASS" || echo "FAIL: too few"
systemctl is-active quantum-ai-engine && echo "  quantum-ai-engine: active" || echo "  FAIL: ai-engine down"

# ---- H1: PERMIT KEY TTL ----
echo ""
echo "=== H1: Permit key TTL ==="
TOTAL=$(redis-cli KEYS "quantum:permit:*" 2>/dev/null | wc -l)
echo "  Total permit keys: $TOTAL"
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -30); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
[ "$PERM" = "0" ] && echo "PASS: 0 permanent keys" || echo "WARN: $PERM keys without TTL"
echo "  Code fixes:"
grep -c "H1 fix\|H1 TTL" /home/qt/quantum_trader/microservices/intent_bridge/main.py 2>/dev/null | xargs echo "    intent_bridge (home):"
grep -c "H1 fix\|H1 TTL" /opt/quantum/microservices/harvest_brain/harvest_brain.py 2>/dev/null | xargs echo "    harvest_brain (opt):"
grep -c "expire(permit_key" /opt/quantum/microservices/risk_brake_v1_patch.py 2>/dev/null | xargs echo "    risk_brake (opt):"

# ---- H2: STALE XGB MODELS ----
echo ""
echo "=== H2: Stale xgb_model_v*.pkl ==="
COUNT=$(find /opt/quantum/ai_engine/models/ /home/qt/quantum_trader/ai_engine/models/ -name "xgb_model_v*.pkl" 2>/dev/null | wc -l)
[ "$COUNT" = "0" ] && echo "PASS: 0 stale model files" || echo "WARN: $COUNT stale files remain"

# ---- H3: SLOT KEYS ----
echo ""
echo "=== H3: Slot/position keys ==="
MAX=$(redis-cli get quantum:max_slots 2>/dev/null)
SLOTS=$(redis-cli get quantum:slot_count 2>/dev/null)
POS=$(redis-cli get quantum:positions_count 2>/dev/null)
[ -n "$MAX" ] && echo "PASS: max_slots=$MAX slot_count=$SLOTS positions_count=$POS" || echo "FAIL: keys missing"

# ---- EXTRA: NEW ISSUES CHECK ----
echo ""
echo "=== EXTRA: New issue detection ==="

# Disk usage
DISK=$(df -h / | awk 'NR==2{print $5}')
echo "  Disk usage /: $DISK"

# Memory
MEM=$(free -m | awk '/Mem:/{printf "used=%dMB total=%dMB (%.0f%%)\n", $3,$2,$3/$2*100}')
echo "  RAM: $MEM"

# Redis memory
RMEM=$(redis-cli info memory 2>/dev/null | grep "used_memory_human" | head -1)
echo "  Redis: $RMEM"

# Redis total keys
RKEYS=$(redis-cli dbsize 2>/dev/null)
echo "  Redis total keys: $RKEYS"

# Check for exploding streams
STREAM_LEN=$(redis-cli xlen quantum:apply.plan 2>/dev/null)
echo "  quantum:apply.plan stream length: $STREAM_LEN"

# Any services in failed state
echo ""
echo "  Failed systemd units (quantum-*):"
systemctl list-units --state=failed "quantum-*" 2>/dev/null | grep -v "^$\|LOAD\|ACTIVE\|--" || echo "  none"

# Last 5 journal errors for key services
echo ""
echo "  Recent ERROR in quantum-backend (last 2h):"
journalctl -u quantum-backend --since "2 hours ago" -p err --no-pager 2>/dev/null | tail -5 || echo "  none / not accessible"

echo ""
echo "  Recent ERROR in quantum-ai-engine (last 2h):"
journalctl -u quantum-ai-engine --since "2 hours ago" -p err --no-pager 2>/dev/null | tail -5 || echo "  none"

# ---- SERVICE OVERVIEW ----
echo ""
echo "=== SERVICE OVERVIEW ==="
SERVICES=(
    quantum-backend
    quantum-ai-engine
    quantum-portfolio-state-publisher
    quantum-governor
    quantum-intent-bridge
    quantum-intent-executor
    quantum-harvest-brain
    quantum-harvest-brain-2
    quantum-harvest-v2
    quantum-risk-brake
    quantum-risk-brain
    quantum-risk-safety
    quantum-risk-proposal
    quantum-apply-layer
)
for svc in "${SERVICES[@]}"; do
    STATE=$(systemctl is-active $svc 2>/dev/null || echo not-found)
    printf "  %-45s %s\n" "$svc" "$STATE"
done

# ---- GIT STATUS ----
echo ""
echo "=== GIT STATUS ==="
echo "  Home repo:"
cd /home/qt/quantum_trader
git log --oneline -3
git status --short | wc -l | xargs echo "  Uncommitted files:"

echo ""
echo "========================================="
echo "  AUDIT COMPLETE — $(date)"
echo "========================================="
