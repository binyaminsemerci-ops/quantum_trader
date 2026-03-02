#!/bin/bash
/opt/quantum/venvs/ai-client-base/bin/python3 /tmp/fp25s.py
echo "EXIT: $?"
grep -n "CLOSE_PARTIAL_25" /opt/quantum/microservices/apply_layer/main.py
systemctl restart quantum-apply-layer.service
sleep 2
systemctl is-active quantum-apply-layer.service
