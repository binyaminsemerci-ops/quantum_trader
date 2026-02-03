#!/bin/bash
# Security Hardening Script
set -e

echo "=== Security Hardening ==="

# UFW Firewall
echo "[1/5] Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow from 127.0.0.1 to any port 9090 comment 'Prometheus'
sudo ufw allow from 127.0.0.1 to any port 3001 comment 'Grafana'
sudo ufw --force enable

# SSH Hardening
echo "[2/5] Hardening SSH..."
sudo sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart sshd

# Docker Security
echo "[3/5] Securing Docker..."
sudo usermod -aG docker $USER
sudo chmod 660 /var/run/docker.sock

# Fail2ban
echo "[4/5] Installing fail2ban..."
sudo apt-get update
sudo apt-get install -y fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Automatic Updates
echo "[5/5] Enabling automatic security updates..."
sudo apt-get install -y unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades

echo "✅ Security hardening complete"
sudo ufw status verbose
