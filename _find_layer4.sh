#!/bin/bash
echo "=== Layer4 sizing service processes ==="
ps aux | grep layer4 | grep -v grep

echo ""
echo "=== Layer4 sizing systemd service ==="
systemctl list-units --type=service --state=active 2>/dev/null | grep layer4

echo ""
echo "=== Find layer4:sizing code ==="
find /opt/quantum /home/qt -name "*.py" 2>/dev/null | xargs grep -l "layer4:sizing" 2>/dev/null | head -10

echo ""
echo "=== Find apply_layer position sizing fallback ==="
grep -n "MIN_POSITION\|min_notional\|fallback.*size\|size.*fallback\|default.*size\|kelly.*fallback\|NO_DATA\|no_backtest\|layer4_no_edge" /home/qt/quantum_trader/microservices/apply_layer/main.py | head -30

echo ""
echo "=== Apply layer Kelly sizing code (lines 1330-1420) ==="
sed -n '1330,1420p' /home/qt/quantum_trader/microservices/apply_layer/main.py
