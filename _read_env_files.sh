#!/bin/bash
echo "=== /etc/quantum/apply-layer.env ==="
cat /etc/quantum/apply-layer.env

echo ""
echo "=== quantum-execution.service ==="
cat /etc/systemd/system/quantum-execution.service 2>/dev/null | head -25

echo ""
echo "=== quantum-harvest-v2.service ==="
cat /etc/systemd/system/quantum-harvest-v2.service 2>/dev/null | head -30

echo ""
echo "=== quantum-ensemble-predictor.service ==="
cat /etc/systemd/system/quantum-ensemble-predictor.service 2>/dev/null | head -25

echo ""
echo "=== all quantum .env files in /etc/quantum ==="
ls /etc/quantum/
echo ""
for f in /etc/quantum/*.env; do echo "--- $f ---"; cat "$f" | head -10; done
