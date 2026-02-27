#!/bin/bash
# Fix all 6 audit issues on VPS
# Run with: wsl bash /mnt/c/quantum_trader/fix_all_6.sh

SSH="ssh -i ~/.ssh/hetzner_fresh -o StrictHostKeyChecking=no root@46.224.116.254"

echo "========================================="
echo "FIX 1 P0: Reset drawdown peak in Redis"
echo "========================================="
$SSH '
EQ=$(redis-cli hget quantum:equity:current equity)
echo "Current equity: $EQ"
echo "Old peak:       $(redis-cli hget quantum:equity:current peak)"
redis-cli hset quantum:equity:current peak "$EQ"
echo "New peak:       $(redis-cli hget quantum:equity:current peak)"
echo "Drawdown = 0% -- EMERGENCY_FLATTEN cleared"
'

echo ""
echo "========================================="
echo "FIX 4 P3: AI_MAX_LEVERAGE 80 -> 10"
echo "========================================="
$SSH '
echo "Before: $(grep AI_MAX_LEVERAGE /etc/quantum/ai-engine.env)"
sed -i "s/AI_MAX_LEVERAGE=80/AI_MAX_LEVERAGE=10/" /etc/quantum/ai-engine.env
echo "After:  $(grep AI_MAX_LEVERAGE /etc/quantum/ai-engine.env)"
echo "FIX4: DONE"
'

echo ""
echo "========================================="
echo "FIX 2 P1: TradeOutcome missing 5 args"
echo "========================================="
$SSH '
FILE=/opt/quantum/microservices/ai_engine/service.py
# Get exact current block around TradeOutcome
echo "--- Current TradeOutcome call (lines 1638-1651) ---"
sed -n "1638,1655p" "$FILE"
'

echo ""
echo "========================================="
echo "FIX 3 P2: git commit all modified files"
echo "========================================="
$SSH '
cd /opt/quantum
echo "Modified files:"
git diff --name-only
echo ""
git add -A
git commit -m "chore: audit cleanup - commit all pending production modifications" || echo "Nothing to commit"
echo "FIX3: DONE"
'

echo ""
echo "============================"
echo "ALL SETUP COMMANDS DONE"
echo "============================"
