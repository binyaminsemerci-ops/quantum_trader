#!/bin/bash
# Complete state check before drastic fixes
echo "=== STREAM LENGTHS ==="
redis-cli XLEN quantum:stream:apply.plan
redis-cli XLEN quantum:stream:apply.result
redis-cli XLEN quantum:stream:harvest.v2.shadow

echo "=== LAST 2 apply.plan entries ==="
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 2

echo "=== LAST 2 apply.result entries ==="
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 2

echo "=== LAST 2 harvest.v2.shadow entries ==="
redis-cli XREVRANGE quantum:stream:harvest.v2.shadow + - COUNT 2

echo "=== HARVEST_V2 CONFIG ==="
redis-cli HGETALL quantum:config:harvest_v2

echo "=== POSITION PROVIDER keys (what harvest_v2 reads) ==="
redis-cli KEYS "quantum:position:*"

echo "=== SAMPLE POSITION ==="
redis-cli HGETALL quantum:position:BNBUSDT

echo "=== SAMPLE POSITION 2 ==="
redis-cli HGETALL quantum:position:XRPUSDT

echo "=== ATR VALUES ==="
redis-cli KEYS "quantum:atr:*"
redis-cli HGETALL quantum:atr:BNBUSDT
redis-cli HGETALL quantum:atr:XRPUSDT

echo "=== APPLY_LAYER ENV ==="
cat /etc/systemd/system/apply-layer.service 2>/dev/null | grep -E "Environment|ExecStart" | head -20

echo "=== INTENT_EXECUTOR ENV ==="
cat /etc/systemd/system/intent-executor.service 2>/dev/null | grep -E "Environment|ExecStart" | head -20

echo "=== ENSEMBLE_PREDICTOR ENV ==="
cat /etc/systemd/system/ensemble-predictor.service 2>/dev/null | grep -E "Environment|ExecStart" | head -20

echo "=== APPLY_LAYER RISK SETTINGS (live) ==="
redis-cli HGETALL quantum:config:apply_layer 2>/dev/null

echo "=== RUNNING SERVICES ==="
ps aux | grep -E "(apply_layer|intent_exec|ensemble|harvest_v2)" | grep -v grep | awk '{print $11, $12}'
