#!/bin/bash
echo "=== ALL QUANTUM SERVICES (active) ==="
systemctl list-units --type=service --state=active | grep quantum | awk '{print $1}' | sort

echo ""
echo "=== EXECUTION RELATED SERVICES ==="
systemctl list-units --type=service --all | grep -iE "execut|apply|intent|trade|harvest|exit|close|order"

echo ""
echo "=== MICROSERVICES DIRECTORY ==="
ls -la /opt/quantum/microservices/

echo ""
echo "=== APPLY LAYER LOGS (last 5) ==="
journalctl -u quantum-apply-layer -n 5 --no-pager 2>/dev/null || echo "no quantum-apply-layer"

echo ""
echo "=== INTENT EXECUTOR LOGS (last 5) ==="
journalctl -u quantum-intent-executor -n 5 --no-pager 2>/dev/null || echo "no quantum-intent-executor"

echo ""
echo "=== AUTONOMOUS TRADER LOGS (last 5) ==="
journalctl -u quantum-autonomous-trader -n 5 --no-pager 2>/dev/null || echo "no quantum-autonomous-trader"

echo ""
echo "=== EXITBRAIN LOGS (last 5) ==="
journalctl -u quantum-exitbrain-v35 -n 5 --no-pager 2>/dev/null || \
journalctl -u quantum-exitbrain -n 5 --no-pager 2>/dev/null || echo "no exitbrain"

echo ""
echo "=== HARVEST BRAIN LOGS (last 5) ==="
journalctl -u quantum-harvest-brain -n 5 --no-pager 2>/dev/null || echo "no harvest-brain"
