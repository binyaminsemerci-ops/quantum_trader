#!/bin/bash
echo "=== P1 REVISJON: Er MANUAL_LANE_OFF en ekte blokkering? ==="
echo ""
echo "--- manual_lane:enabled nøkkel ---"
redis-cli EXISTS quantum:manual_lane:enabled 2>/dev/null
redis-cli TTL quantum:manual_lane:enabled 2>/dev/null
redis-cli GET quantum:manual_lane:enabled 2>/dev/null

echo ""
echo "--- Hva skjer med harvest-planer i intent-executor? ---"
journalctl -u quantum-intent-executor.service -n 60 --no-pager 2>/dev/null | \
    grep -E "HARVEST|harvest|EXECUTE|executed=True|P3\.|SKIP|BLOCKED|manual_lane|lane=" | tail -20

echo ""
echo "--- Intent-executor kode: er manual_lane sjekket for harvest-lane? ---"
grep -n "lane\|HARVEST\|harvest_stream\|manual_lane" \
    /home/qt/quantum_trader/microservices/intent_executor/main.py 2>/dev/null | \
    grep -v "^--" | head -30

echo ""
echo "=== P2 DIAGNOSE: quantum:stream:signals ==="
echo ""
echo "--- Signal stream nøkkel type ---"
redis-cli TYPE quantum:stream:signals 2>/dev/null
redis-cli EXISTS quantum:stream:signals 2>/dev/null

echo ""
echo "--- Alle signal-relaterte keys ---"
redis-cli KEYS 'quantum:stream:signal*' 2>/dev/null
redis-cli KEYS 'quantum:*signal*' 2>/dev/null | grep -v done | head -20

echo ""
echo "--- Ensemble predictor logg siste 30 linjer ---"
journalctl -u quantum-ensemble-predictor.service -n 30 --no-pager 2>/dev/null | tail -30

echo ""
echo "--- Ensemble consumer group på features stream ---"
redis-cli XINFO GROUPS quantum:stream:features 2>/dev/null

echo ""
echo "--- Ensemble kildekode: signal stream nøkkel ---"
grep -n "stream:signal\|signals\|XADD\|xadd" \
    /opt/quantum/ai_engine/services/ensemble_predictor_service.py 2>/dev/null | head -20

echo ""
echo "=== P3 DIAGNOSE: duplicate_plan ==="
echo ""
echo "--- Antall dedupe-nøkler ---"
redis-cli KEYS 'quantum:apply:dedupe:*' 2>/dev/null | wc -l

echo ""
echo "--- TTL på noen dedupe-nøkler ---"
redis-cli KEYS 'quantum:apply:dedupe:*' 2>/dev/null | head -5 | while read k; do
    echo "  $k TTL=$(redis-cli TTL $k)"
done

echo ""
echo "--- Harvest-v2 logg: genererer det nye plan_id'er? ---"
journalctl -u quantum-harvest-v2.service -n 20 --no-pager 2>/dev/null | tail -15

echo ""
echo "--- apply.plan stream: siste 3 entries ---"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 2>/dev/null | \
    grep -E "plan_id|symbol|decision|reason_codes|action" | head -20
