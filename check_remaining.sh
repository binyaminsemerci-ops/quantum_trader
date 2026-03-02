#!/bin/bash
echo "=== Is /opt/quantum a symlink? ==="
ls -la /opt/quantum 2>/dev/null | head -3

echo ""
echo "=== Find 'Funding check timeout' source ==="
grep -rn "Funding check timeout\|funding.*check.*timeout" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak" | head -10

echo ""
echo "=== Find 'Drift detection TIMEOUT' source ==="
grep -rn "Drift detection TIMEOUT\|drift.*TIMEOUT\|TIMEOUT.*drift" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak" | head -10

echo ""
echo "=== M2 top_coins_cache.json permissions ==="
ls -la /opt/quantum/backend/data/cache/
ls -la /home/qt/quantum_trader/backend/data/cache/ 2>/dev/null | head -5

echo ""
echo "=== H1 permit keys now ==="
redis-cli --scan --pattern "quantum:permit:*" | wc -l

echo ""
echo "=== H2 stale models home path ==="
ls /home/qt/quantum_trader/ai_engine/models/xgb_model_v*.pkl 2>/dev/null | wc -l
