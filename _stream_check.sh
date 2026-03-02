#!/bin/bash
echo "=== Redis feature keys ==="
redis-cli KEYS 'quantum:stream:feat*' 2>/dev/null
redis-cli KEYS 'quantum:*feature*' 2>/dev/null | head -20

echo ""
echo "=== Feature publisher stream writes ==="
grep -n "xadd\|XADD\|stream_key\|stream_name\|FEATURES\|features\b" \
    /opt/quantum/ai_engine/services/feature_publisher_service.py 2>/dev/null | head -20

echo ""
echo "=== Redis: scan all streams with data ==="
redis-cli KEYS 'quantum:stream:*' 2>/dev/null | while read k; do
  len=$(redis-cli XLEN "$k" 2>/dev/null)
  if [ "$len" -gt 0 ] 2>/dev/null; then
    echo "  $k => $len"
  fi
done | sort -t= -k2 -rn | head -30

echo ""
echo "=== Ensemble predictor reads from ==="
grep -n "features\|FEATURE\|stream" \
    /opt/quantum/ai_engine/services/ensemble_predictor_service.py 2>/dev/null | \
    grep -i "read\|xread\|consume\|stream_key\|STREAM" | head -20

echo ""
echo "=== Positions open in Redis ==="
redis-cli KEYS 'quantum:position:*' 2>/dev/null | while read k; do
  val=$(redis-cli HGETALL "$k" 2>/dev/null | tr '\n' ' ')
  echo "  $k: $val"
done
