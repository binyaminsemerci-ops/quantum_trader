#!/bin/bash
cd /home/qt/quantum_trader

echo "=== Resolving add/add conflicts by keeping ours ==="
# Get all conflicted files
CONFLICTED=$(git diff --name-only --diff-filter=U 2>/dev/null)
echo "Conflicted files:"
echo "$CONFLICTED"

# For each conflicted file, keep ours (VPS local version)
for f in $CONFLICTED; do
    git checkout --ours "$f" 2>/dev/null && echo "  Kept ours: $f"
    git add "$f" 2>/dev/null
done

echo ""
echo "=== Commit merge ==="
git commit --no-edit -m "merge: resolve add/add conflicts keeping VPS audit-fix versions" 2>&1

echo ""
echo "=== Push ==="
git push origin main 2>&1

echo ""
echo "=== Last 5 commits ==="
git log --oneline -5

echo ""
echo "=== Verify critical H1 file still has TTL fix ==="
grep -c "H1 fix" microservices/intent_bridge/main.py && echo "intent_bridge: OK"
grep -c "H1 fix" microservices/harvest_brain/harvest_brain.py && echo "harvest_brain: OK"
