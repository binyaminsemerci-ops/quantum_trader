#!/bin/bash
echo "=== INTENT_EXECUTOR kill_score_close_ok logic ==="
grep -n "kill_score_close_ok" /opt/quantum/microservices/intent_executor/main.py

echo ""
echo "=== INTENT_EXECUTOR SKIP logic ==="
grep -n "SKIP" /opt/quantum/microservices/intent_executor/main.py | head -30

echo ""
echo "=== INTENT_EXECUTOR FULL_CLOSE logic ==="
grep -n "FULL_CLOSE" /opt/quantum/microservices/intent_executor/main.py | head -30

echo ""
echo "=== INTENT_EXECUTOR harvest_v2 check ==="
grep -n "harvest_v2\|source\b" /opt/quantum/microservices/intent_executor/main.py | head -20

echo ""
echo "=== INTENT_EXECUTOR kill_score threshold near CLOSE ==="
grep -n "kill_score\|K_CLOSE\|K_OPEN" /opt/quantum/microservices/intent_executor/main.py | head -40

echo ""
echo "=== APPLY_LAYER: how it uses kill_score for entries ==="
grep -n "kill_score\|K_OPEN_THRESHOLD\|K_CLOSE_THRESHOLD\|kelly\|layer4\|size_usdt" /opt/quantum/microservices/apply_layer/main.py | head -60
