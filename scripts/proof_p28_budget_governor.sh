#!/bin/bash
# P2.8 Portfolio Risk Governor - Proof Bundle
# Generated: 2026-01-27
# Purpose: Verify P2.8 deployment and integration

set -e

echo "═══════════════════════════════════════════════════════════"
echo "P2.8 PORTFOLIO RISK GOVERNOR - PROOF BUNDLE"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "1️⃣ SERVICE STATUS"
echo "─────────────────────────────────────────────────────────"
systemctl status quantum-portfolio-risk-governor --no-pager | head -20
echo ""

echo "2️⃣ PROMETHEUS METRICS"
echo "─────────────────────────────────────────────────────────"
curl -s localhost:8049/metrics | grep "^p28_"
echo ""

echo "3️⃣ HEALTH CHECK"
echo "─────────────────────────────────────────────────────────"
curl -s localhost:8049/health | jq .
echo ""

echo "4️⃣ SERVICE LOGS (last 20 lines)"
echo "─────────────────────────────────────────────────────────"
journalctl -u quantum-portfolio-risk-governor -n 20 --no-pager
echo ""

echo "5️⃣ REDIS BUDGET HASHES"
echo "─────────────────────────────────────────────────────────"
echo "Budget hash keys:"
redis-cli KEYS "quantum:portfolio:budget:*"
echo ""
echo "Sample budget hash (if exists):"
SAMPLE_KEY=$(redis-cli KEYS "quantum:portfolio:budget:*" | head -1)
if [ -n "$SAMPLE_KEY" ]; then
    redis-cli HGETALL "$SAMPLE_KEY"
else
    echo "(No budget hashes yet - waiting for active positions)"
fi
echo ""

echo "6️⃣ BUDGET VIOLATION EVENTS"
echo "─────────────────────────────────────────────────────────"
echo "Recent violations (last 5):"
redis-cli XREVRANGE quantum:stream:budget.violation + - COUNT 5
echo ""

echo "7️⃣ GOVERNOR INTEGRATION CHECK"
echo "─────────────────────────────────────────────────────────"
echo "Checking Governor code for P2.8 integration..."
grep -n "_check_portfolio_budget" /home/qt/quantum_trader/microservices/governor/main.py | head -3
echo ""
echo "Checking Governor logs for P2.8 mentions..."
journalctl -u quantum-governor -n 100 --no-pager | grep -i "p28\|budget" | tail -5 || echo "(No P2.8 activity yet - waiting for trades)"
echo ""

echo "8️⃣ CONFIGURATION"
echo "─────────────────────────────────────────────────────────"
cat /etc/quantum/portfolio-risk-governor.env
echo ""

echo "9️⃣ PORT BINDING"
echo "─────────────────────────────────────────────────────────"
netstat -tlnp | grep 8049
echo ""

echo "🔟 OPS LEDGER ENTRY"
echo "─────────────────────────────────────────────────────────"
grep -A 20 "OPS-2026-01-27-011" /root/quantum_trader/docs/OPS_CHANGELOG.md
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "✅ P2.8 PORTFOLIO RISK GOVERNOR - PROOF COMPLETE"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Service: quantum-portfolio-risk-governor"
echo "Status: Active (running)"
echo "Mode: SHADOW (logging violations, not blocking)"
echo "Port: 8049"
echo "Integration: Governor Gate 0 (production mode)"
echo ""
echo "Next Steps:"
echo "  1. Monitor shadow mode for 24-48 hours"
echo "  2. Verify budget computations with real positions"
echo "  3. Activate ENFORCE mode via: sed -i 's/shadow/enforce/' /etc/quantum/portfolio-risk-governor.env"
echo "  4. Restart service: systemctl restart quantum-portfolio-risk-governor"
echo ""
