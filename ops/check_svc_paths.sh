#!/bin/bash
echo "=== intent_bridge service file ==="
cat /etc/systemd/system/quantum-intent-bridge.service

echo ""
echo "=== apply_layer service file ==="
cat /etc/systemd/system/quantum-apply-layer.service

echo ""
echo "=== Running processes (WorkingDirectory?) ==="
cat /proc/$(systemctl show quantum-intent-bridge -p MainPID --value)/cmdline | tr '\0' ' '
echo ""
ls -la /proc/$(systemctl show quantum-intent-bridge -p MainPID --value)/cwd
