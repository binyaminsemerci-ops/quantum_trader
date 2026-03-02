#!/bin/bash
echo "=== apply_layer: PARTIAL_25 references ==="
grep -n "PARTIAL_25\|partial_25\|partial25" /opt/quantum/microservices/apply_layer/main.py | grep -v "bak\|backup" | head -30

echo ""
echo "=== apply_layer: normalize_action function ==="
grep -n "normalize_action\|is_close_action\|action_normalized" /opt/quantum/microservices/apply_layer/main.py | head -20

echo ""
echo "=== apply_layer lines around 1190-1210 (normalize_action) ==="
sed -n '1190,1215p' /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== apply_layer lines around layer4/kelly_200usdt ==="
grep -n "kelly_200usdt\|layer4\|Layer4\|LAYER4\|layer 4" /opt/quantum/microservices/apply_layer/main.py | head -20

echo ""
echo "=== apply_layer: action_hold decision ==="
grep -n "action_hold\|action_normalized.*unknown\|unknown_variant" /opt/quantum/microservices/apply_layer/main.py | head -20
