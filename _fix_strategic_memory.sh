#!/bin/bash
mkdir -p /etc/systemd/system/quantum-strategic-memory.service.d
cat > /etc/systemd/system/quantum-strategic-memory.service.d/redis.conf << 'EOF'
[Service]
Environment=REDIS_URL=redis://127.0.0.1:6379/0
EOF
systemctl daemon-reload
systemctl restart quantum-strategic-memory
sleep 4
journalctl -u quantum-strategic-memory -n 8 --no-pager
