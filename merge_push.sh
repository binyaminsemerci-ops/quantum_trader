#!/bin/bash
cd /home/qt/quantum_trader

echo "=== Aborting failed rebase ==="
git rebase --abort 2>&1 || true

echo ""
echo "=== Merge with ours strategy for conflicts ==="
git merge origin/main -X ours --no-edit 2>&1

echo ""
echo "=== Push ==="
git push origin main 2>&1

echo ""
echo "=== Last 5 commits ==="
git log --oneline -5
