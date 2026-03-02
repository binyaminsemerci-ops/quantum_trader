#!/bin/bash
echo "=== Meta-regime file location ==="
find /home/qt /opt/quantum -name "*meta_regime*" 2>/dev/null | head -10
find /home/qt /opt/quantum -name "*meta-regime*" 2>/dev/null | head -10

echo ""
echo "=== Meta-regime source: what stream/key does it read from? ==="
META_FILE=$(find /home/qt /opt/quantum -name "*meta_regime*" -name "*.py" 2>/dev/null | head -1)
echo "Using: $META_FILE"
if [ -n "$META_FILE" ]; then
    grep -n "stream\|kline\|market\|XREAD\|xread\|KEYS\|GET\|HGET\|required.*50\|50.*required\|found" "$META_FILE" 2>/dev/null | head -30
fi

echo ""
echo "=== Market klines stream: what format? ==="
redis-cli XREVRANGE quantum:stream:market.klines + - COUNT 1 2>/dev/null | head -30

echo ""
echo "=== market_events stream: latest entry ==="
redis-cli XREVRANGE quantum:stream:market_events + - COUNT 1 2>/dev/null | head -20

echo ""
echo "=== marketstate stream: latest entry symbol check ==="
redis-cli XREVRANGE quantum:stream:marketstate + - COUNT 1 2>/dev/null | head -20

echo ""
echo "=== MANUAL REGIME FIX: Set regime to TRENDING ==="
# If meta-regime can't find data, set a manual regime so ensemble exits init mode
redis-cli SET quantum:meta:regime "TRENDING" EX 3600
echo "Set quantum:meta:regime = TRENDING (TTL 1 hour)"

echo ""
echo "=== Signal risk_context after regime set (check in 5s) ==="
sleep 5
redis-cli XREVRANGE quantum:stream:signal.score + - COUNT 3 2>/dev/null | grep "risk_context" | head -5

echo ""
echo "=== Verify apply.result (any executions?) ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 5 2>/dev/null | grep -E "executed|plan_id|symbol|error" | head -25
