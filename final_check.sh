#!/bin/bash
echo "=== M3: service.py timeout strings ==="
grep -n "timeout.*5s\|timeout.*12s\|TIMEOUT.*5s\|TIMEOUT.*12s" /opt/quantum/microservices/ai_engine/service.py | tail -15

echo ""
echo "=== M2: top_coins_cache.json owner ==="
ls -la /opt/quantum/backend/data/cache/top_coins_cache.json 2>/dev/null
ls -la /home/qt/quantum_trader/backend/data/cache/top_coins_cache.json 2>/dev/null
chown qt:qt /opt/quantum/backend/data/cache/top_coins_cache.json 2>/dev/null && echo "opt: chown OK"
chown qt:qt /home/qt/quantum_trader/backend/data/cache/top_coins_cache.json 2>/dev/null && echo "home: chown OK"

echo ""
echo "=== H1: Permit keys count ==="
redis-cli --scan --pattern "quantum:permit:*" | wc -l

echo ""
echo "=== H1: Permit key creation code ==="
grep -rn "permit.*set\|setex.*permit\|permit.*nx\|setnx.*permit\|permit.*p33" /opt/quantum/ 2>/dev/null | grep -v ".pyc\|emoji_backup\|.bak\|.md" | grep ".py:" | head -15

echo ""
echo "=== All services status ==="
for svc in quantum-backend quantum-ai-engine quantum-portfolio-state-publisher quantum-balance-tracker; do
  echo "$svc: $(systemctl is-active $svc)"
done

echo ""
echo "=== Backend health ==="
curl -s http://localhost:8000/ 2>/dev/null

echo ""
echo "=== Equity USD ==="
redis-cli get quantum:equity_usd
redis-cli get quantum:max_leverage
redis-cli get quantum:circuit_breaker
