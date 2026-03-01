#!/bin/bash
# Read env files and service unit files for key services
echo "=== quantum-apply-layer.service ==="
cat /etc/systemd/system/quantum-apply-layer.service 2>/dev/null

echo ""
echo "=== quantum-execution.service ==="
cat /etc/systemd/system/quantum-execution.service 2>/dev/null

echo ""
echo "=== quantum-harvest-v2.service ==="
cat /etc/systemd/system/quantum-harvest-v2.service 2>/dev/null

echo ""
echo "=== quantum-ensemble-predictor.service ==="
cat /etc/systemd/system/quantum-ensemble-predictor.service 2>/dev/null

echo ""
echo "=== apply-layer.env ==="
cat /opt/quantum/deployment/config/apply-layer.env 2>/dev/null

echo ""
echo "=== intent-executor related service files ==="
find /etc/systemd/system -name "*.service" | xargs grep -l "intent.exec" 2>/dev/null | head -5

echo ""
echo "=== List all quantum service files with intent/execution ==="
ls /etc/systemd/system/quantum-*.service 2>/dev/null | grep -E "intent|exec|execution"

echo ""
echo "=== harvest-v2 env ==="
find /opt/quantum /home/qt -name "harvest-v2.env" -o -name "harvest_v2.env" 2>/dev/null | xargs cat 2>/dev/null
