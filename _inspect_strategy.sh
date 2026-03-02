#!/bin/bash
# Extracts key decision logic from core files
echo "=== APPLY_LAYER FUNCTIONS ==="
grep -n "def \|cooldown\|min_hold\|stop_loss\|take_profit\|STOP\|LIMIT\|MAX_\|interval\|rate_limit\|signal\|score\|threshold" /opt/quantum/microservices/apply_layer/main.py | head -80

echo ""
echo "=== INTENT_EXECUTOR KEY LINES ==="
grep -n "def \|cooldown\|min_hold\|stop_loss\|take_profit\|LIMIT\|MAX_\|interval\|rate_limit\|score\|threshold\|reduce_only\|CLOSE\|OPEN" /opt/quantum/microservices/intent_executor/main.py | head -80

echo ""
echo "=== HARVEST_BRAIN KEY LINES ==="
grep -n "def \|min_r\|R =\|harvest\|ladder\|trail\|stop\|close\|reduce" /opt/quantum/microservices/harvest_brain/harvest_brain.py | head -60

echo ""
echo "=== RISK_KERNEL_HARVEST ==="
find /opt/quantum -name "risk_kernel_harvest.py" 2>/dev/null
cat /opt/quantum/ai_engine/risk_kernel_harvest.py 2>/dev/null | head -120

echo ""
echo "=== APPLY_LAYER HEAD ==="
head -150 /opt/quantum/microservices/apply_layer/main.py

echo ""
echo "=== INTENT_EXECUTOR APPLY/EXECUTE part ==="
grep -n "def apply\|def execute\|def send\|def place\|def open\|def close\|cooldown\|last_trade\|min_interval\|rate" /opt/quantum/microservices/intent_executor/main.py | head -40

echo ""
echo "=== SERVICES RUNNING ==="
systemctl list-units --type=service --state=running | grep quantum | grep -v "systemd"

echo ""
echo "=== ENV VARS FOR KEY SERVICES ==="
systemctl show quantum-harvest-brain.service --property=Environment 2>/dev/null | head -5
systemctl show quantum-intent-executor.service --property=Environment 2>/dev/null | head -5
systemctl show quantum-apply-layer.service --property=Environment 2>/dev/null | head -5
