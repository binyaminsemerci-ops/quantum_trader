#!/bin/bash
echo "=== Context around line 1050-1090 in /opt/quantum xgb_agent.py ==="
sed -n '1030,1095p' /opt/quantum/ai_engine/agents/xgb_agent.py

echo ""
echo "=== [PHASE search ==="
grep -rn "\[PHASE" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|bak\|test" | head -30
