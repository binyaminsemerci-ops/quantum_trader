#!/bin/bash
echo "=== M2: Data dirs ==="
ls -la /opt/quantum/backend/data/cache/ 2>/dev/null | head -4
ls -la /home/qt/quantum_trader/dashboard_v4/backend/data/cache/ 2>/dev/null | head -4

echo ""
echo "=== M3: AI engine status ==="
systemctl is-active quantum-ai-engine
journalctl -u quantum-ai-engine --since "30 min ago" --no-pager 2>/dev/null | grep -i "timeout\|started\|error\|PHASE 1\|Funding" | tail -8

echo ""
echo "=== H1: Permit keys remaining ==="
redis-cli --scan --pattern "quantum:permit:*" | wc -l

echo ""
echo "=== H2: Stale XGB models ==="
ls /opt/quantum/ai_engine/models/xgb_model_v*.pkl 2>/dev/null | wc -l
ls /home/qt/quantum_trader/ai_engine/models/xgb_model_v*.pkl 2>/dev/null | wc -l

echo ""
echo "=== H3: Slot keys ==="
redis-cli mget quantum:max_slots quantum:slot_count quantum:positions_count

echo ""
echo "=== All services ==="
for svc in quantum-backend quantum-ai-engine quantum-portfolio-state-publisher quantum-balance-tracker quantum-market-publisher quantum-rl-trainer; do
  echo "$svc: $(systemctl is-active $svc 2>/dev/null)"
done

echo ""
echo "=== Backend API health ==="
curl -s http://localhost:8000/ 2>/dev/null | head -3

echo ""
echo "=== Equity USD ==="
redis-cli get quantum:equity_usd
