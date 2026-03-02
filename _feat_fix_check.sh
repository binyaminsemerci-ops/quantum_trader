#!/bin/bash
echo "=== XADD call in feature publisher ==="
grep -n "xadd\|XADD\|maxlen\|MAXLEN\|stream_key\|r\.xadd\|redis.*xadd" \
    /opt/quantum/ai_engine/services/feature_publisher_service.py 2>/dev/null

echo ""
echo "=== feature publisher full relevant section (lines 150-250) ==="
sed -n '150,250p' /opt/quantum/ai_engine/services/feature_publisher_service.py 2>/dev/null

echo ""
echo "=== XLEN of quantum:stream:features ==="
redis-cli XLEN quantum:stream:features

echo ""
echo "=== Last entry in quantum:stream:features ==="
redis-cli XREVRANGE quantum:stream:features + - COUNT 1

echo ""
echo "=== ensemble predictor: MAXLEN config or env ==="
grep -n "MAXLEN\|maxlen\|MAX_LEN\|stream_key\|FEATURE_STREAM\|feature_stream" \
    /opt/quantum/ai_engine/services/ensemble_predictor_service.py 2>/dev/null | head -20
cat /etc/quantum/ai-engine.env 2>/dev/null | grep -iE 'feature|maxlen|stream'

echo ""
echo "=== apply.plan stream — latest 3 entries (plan_ids) ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 3 2>/dev/null | grep -E "plan_id|symbol|action|decision|ts"
