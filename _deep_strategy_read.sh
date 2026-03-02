#!/bin/bash
echo "=== RISK KERNEL HARVEST (full) ==="
cat /opt/quantum/ai_engine/risk_kernel_harvest.py 2>/dev/null | head -300

echo ""
echo "=== HARVEST_BRAIN PROPOSE LOGIC ==="
sed -n '355,600p' /opt/quantum/microservices/harvest_brain/harvest_brain.py

echo ""
echo "=== HARVEST_BRAIN SCAN/MAIN LOOP ==="
sed -n '600,900p' /opt/quantum/microservices/harvest_brain/harvest_brain.py

echo ""
echo "=== RL SIZING AGENT ==="
cat /opt/quantum/microservices/rl_sizing_agent/rl_sizing_agent.py 2>/dev/null | head -200

echo ""
echo "=== CAPITAL ALLOCATION ==="
cat /opt/quantum/microservices/capital_allocation/capital_allocation.py 2>/dev/null | head -150

echo ""
echo "=== ENSEMBLE PREDICTOR (entry scoring) ==="
grep -n "score\|signal\|predict\|confidence\|threshold\|BUY\|SELL\|LONG\|SHORT\|entry\|direction\|open\|close" \
  /opt/quantum/ai_engine/services/ensemble_predictor_service.py 2>/dev/null | head -80

echo ""
echo "=== ENSEMBLE PREDICTOR CORE ==="
sed -n '1,120p' /opt/quantum/ai_engine/services/ensemble_predictor_service.py 2>/dev/null

echo ""
echo "=== HARVEST PROPOSALS LIVE IN REDIS ==="
redis-cli KEYS 'quantum:harvest:*' | sort | head -20
echo "--- sample proposal ---"
KEY=$(redis-cli KEYS 'quantum:harvest:proposal:*' | head -1)
if [ -n "$KEY" ]; then redis-cli HGETALL "$KEY"; fi

echo ""
echo "=== POSITION SIZES LIVE ==="
for k in $(redis-cli KEYS 'quantum:position:[A-Z]*'); do
    sym=$(echo $k | cut -d: -f3)
    qty=$(redis-cli HGET $k quantity 2>/dev/null)
    entry=$(redis-cli HGET $k entry_price 2>/dev/null)
    upnl=$(redis-cli HGET $k unrealized_pnl 2>/dev/null)
    echo "  $sym qty=$qty entry=$entry upnl=$upnl"
done

echo ""
echo "=== LAYER4 KELLY SIZING (active) ==="
redis-cli KEYS 'quantum:layer4:sizing:*' | head -10 | while read k; do
    sym=$(echo $k | cut -d: -f4)
    echo "--- $sym ---"
    redis-cli HGETALL "$k"
done

echo ""
echo "=== HARVEST_V2 SERVICE ==="
wc -l /opt/quantum/microservices/harvest_v2/*.py 2>/dev/null
cat /opt/quantum/microservices/harvest_v2/*.py 2>/dev/null | head -200
