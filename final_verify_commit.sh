#!/bin/bash
echo "=== Final State Check ==="
echo "Redis total keys: $(redis-cli dbsize)"
echo "Redis memory:     $(redis-cli info memory | grep used_memory_human | head -1)"
echo ""
echo "Metrics keys remaining:"
redis-cli KEYS "quantum:metrics:exit:*" | wc -l
echo ""

echo "=== Permit key check ==="
redis-cli KEYS "quantum:permit:*" | wc -l
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" | head -30); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
echo "Permit TTL=-1: $PERM"

echo ""
echo "=== All active services ==="
for svc in quantum-backend quantum-ai-engine quantum-portfolio-state-publisher quantum-governor quantum-intent-bridge quantum-harvest-brain quantum-risk-brake; do
    printf "  %-45s %s\n" "$svc" "$(systemctl is-active $svc 2>/dev/null)"
done

echo ""
echo "=== Git commit + push ==="
cd /home/qt/quantum_trader
git add microservices/harvest_brain/harvest_brain.py scripts/auto_permit_p33.py
STAGED=$(git diff --cached --name-only 2>/dev/null)
if [ -n "$STAGED" ]; then
    echo "Staged: $STAGED"
    git commit -m "fix(H1+Redis): patch harvest_brain home + auto_permit_p33 TTL; delete 1.2M stale metrics keys" 2>&1
fi
git log --oneline -4
git push origin main 2>&1 || echo "Push failed: $(git push origin main 2>&1)"

echo ""
echo "DONE"
