#!/bin/bash
# M2: Fix top_coins_cache.json ownership
chown qt:qt /opt/quantum/backend/data/cache/top_coins_cache.json 2>/dev/null
chown qt:qt /home/qt/quantum_trader/backend/data/cache/top_coins_cache.json 2>/dev/null
echo "M2: chown done"
ls -la /opt/quantum/backend/data/cache/

# M3: Restart AI engine in background (detached)
echo "M3: restarting quantum-ai-engine (background)..."
systemctl restart quantum-ai-engine &
RESTART_PID=$!
echo "Restart PID: $RESTART_PID"

# H1: Check permit key count (use timeout)
echo ""
echo "H1: Permit key count..."
redis-cli --scan --pattern "quantum:permit:*" --count 1000 2>/dev/null | wc -l

# H1: Find permit key creation code
echo ""
echo "H1: Permit key creation in code..."
grep -rn "setnx\|set.*nx\|permit.*set\|permit.*nx" /opt/quantum/microservices/ /opt/quantum/backend/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|.md" | grep "permit" | head -10

# H1: Add TTL permanently - find the exact code that creates permit keys
echo ""
echo "H1: Full permit search..."
grep -rn "permit:" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|.md\|#" | grep "\.py:" | grep -v "test\|hget\|exists\|get\|delete" | head -15

echo ""
echo "Services:"
for svc in quantum-backend quantum-portfolio-state-publisher quantum-balance-tracker; do
  echo "$svc: $(systemctl is-active $svc)"
done

echo ""
echo "Equity USD: $(redis-cli get quantum:equity_usd)"
