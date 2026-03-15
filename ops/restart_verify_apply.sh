#!/bin/bash
systemctl restart quantum-apply-layer
sleep 8
echo "=== apply_layer startup ==="
journalctl -u quantum-apply-layer --no-pager --since "20 seconds ago" -q 2>/dev/null | tail -30
echo ""
echo "=== Check reconcile-engine allowlist ==="
cat /etc/quantum/reconcile-engine.env 2>/dev/null | head -20
echo ""
echo "=== Position keys after cleanup ==="
redis-cli keys "quantum:state:positions:*" 2>/dev/null | sort
echo ""
echo "=== ENTRY test: active positions count ==="
redis-cli keys "quantum:state:positions:*" 2>/dev/null | wc -l | xargs echo "Active positions:"
