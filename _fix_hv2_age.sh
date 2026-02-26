#!/bin/bash
mkdir -p /etc/systemd/system/quantum-harvest-v2.service.d
cat > /etc/systemd/system/quantum-harvest-v2.service.d/max-age.conf << 'EOF'
[Service]
Environment=HARVEST_V2_MAX_AGE_SEC=604800
EOF
systemctl daemon-reload
systemctl restart quantum-harvest-v2
sleep 15
journalctl -u quantum-harvest-v2 -n 6 --no-pager
