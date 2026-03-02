#!/bin/bash
cd /home/qt/quantum_trader
echo "=== Pull + rebase ==="
git pull --rebase origin main 2>&1

echo ""
echo "=== Push ==="
git push origin main 2>&1

echo ""
echo "=== Last 5 commits ==="
git log --oneline -5
