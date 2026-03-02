#!/bin/bash
# Kill any running fix_all.sh instances
pkill -f "fix_all.sh" 2>/dev/null || true
pkill -f "xargs redis-cli DEL" 2>/dev/null || true
sleep 1

echo "=== Fast delete quantum:metrics:exit keys ==="
python3 /tmp/fast_delete_metrics.py

echo ""
echo "=== Permit keys check ==="
PERM=0
for KEY in $(redis-cli KEYS "quantum:permit:*" 2>/dev/null | head -30); do
    T=$(redis-cli TTL "$KEY")
    [ "$T" = "-1" ] && PERM=$((PERM+1))
done
echo "Permit TTL=-1: $PERM"
echo "Total permit: $(redis-cli KEYS 'quantum:permit:*' 2>/dev/null | wc -l)"

echo ""
echo "=== Git commit + push ==="
cd /home/qt/quantum_trader
git add microservices/harvest_brain/harvest_brain.py scripts/auto_permit_p33.py
git diff --cached --name-only || true
git commit -m "fix(H1+Redis): patch harvest_brain home + auto_permit_p33 TTL, delete 1.2M stale metrics keys" 2>&1 || echo "nothing to commit"
git push origin main 2>&1 || git pull --rebase origin main && git push origin main

echo ""
echo "=== Final state ==="
echo "Redis keys: $(redis-cli dbsize)"
echo "Redis mem:  $(redis-cli info memory | grep used_memory_human | head -1)"
echo "DONE"
