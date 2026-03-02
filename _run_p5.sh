#!/bin/bash
/opt/quantum/venvs/ai-client-base/bin/python3 /tmp/p5.py
echo "=== Restarting apply-layer ==="
systemctl restart quantum-apply-layer.service
sleep 3
systemctl is-active quantum-apply-layer.service
echo "=== PARTIAL_25 count ==="
grep -c "PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py
echo "=== Last 5 apply-layer logs ==="
journalctl -u quantum-apply-layer.service -n 5 --no-pager 2>/dev/null
