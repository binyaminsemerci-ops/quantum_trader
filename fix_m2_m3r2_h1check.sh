#!/bin/bash
# M2 fix: Fix top_coins_cache.json ownership
echo "=== M2: Fix top_coins_cache.json ownership ==="
chown qt:qt /opt/quantum/backend/data/cache/top_coins_cache.json 2>/dev/null && echo "opt path: OK" || echo "opt path: not found"
chown qt:qt /home/qt/quantum_trader/backend/data/cache/top_coins_cache.json 2>/dev/null && echo "home path: OK" || echo "home path: not found"
ls -la /opt/quantum/backend/data/cache/
ls -la /home/qt/quantum_trader/backend/data/cache/

# M3 round 2
echo ""
echo "=== M3 R2: Patching remaining 5s timeouts ==="
python3 /tmp/fix_m3_r2.py

# H1: Find where permits are created (for permanent fix)
echo ""
echo "=== H1: Find permit key creation code ==="
grep -rn "permit:p33\|setex.*permit\|set.*permit.*ex\|permit.*expire\|permit.*ttl" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|.md" | head -15

echo ""
echo "=== H1: Current permit key count ==="
redis-cli --scan --pattern "quantum:permit:*" 2>/dev/null | wc -l
