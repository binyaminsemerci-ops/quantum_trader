#!/bin/bash
echo "=== P3.5 guard: grep for PARTIAL_25 in home/qt intent_executor ==="
grep -n "PARTIAL_25\|partial_25\|p35_guard\|P3.5\|P35\|partial" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -30

echo ""
echo "=== P3.5 guard: what is layer4_kelly_200usdt? ==="
grep -n "layer4\|kelly_200usdt\|kelly_200\|200usdt" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -20

echo ""
echo "=== P3.5 guard: action_normalized check ==="
grep -n "action_normalized\|unknown_variant\|normalize_action\|is_close_action" /home/qt/quantum_trader/microservices/intent_executor/main.py | head -20
