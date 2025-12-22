#!/bin/bash
# Direct deployment - copy Position Monitor code to VPS

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo "📝 Copying Position Monitor code directly to VPS..."

# Create backup first
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cp ~/quantum_trader/backend/main.py ~/quantum_trader/backend/main.py.before_position_monitor"

# Copy new main.py
scp -i $SSH_KEY /mnt/c/quantum_trader/backend/main.py qt@46.224.116.254:~/quantum_trader/backend/main.py

echo "🔨 Rebuilding backend..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml build backend"

echo "🔄 Restarting backend..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml up -d backend"

echo "⏳ Waiting 25 seconds..."
sleep 25

echo ""
echo "📊 Checking Position Monitor logs..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "docker logs quantum_backend 2>&1 | tail -100 | grep -E '(POSITION-MONITOR|protection)'"

echo ""
echo "✅ Done!"
