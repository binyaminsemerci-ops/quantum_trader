#!/bin/bash
echo "=== Running services (intent/apply) ==="
systemctl list-units --state=running | grep -iE "intent|apply|bridge" | head -20

echo "=== All quantum services ==="
systemctl list-units --state=running | grep quantum | head -30

echo "=== Process check ==="
ps aux | grep -iE "intent_bridge|apply_layer" | grep -v grep | head -10
