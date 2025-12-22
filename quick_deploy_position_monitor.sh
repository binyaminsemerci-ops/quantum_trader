#!/bin/bash
# Quick deploy script for Position Monitor to VPS

set -e

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo "🛡️ Deploying Position Monitor to VPS..."
echo ""

# Git pull
echo "📥 Pulling latest code..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && git pull origin main"

# Rebuild backend
echo "🔨 Rebuilding backend with Position Monitor..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml build backend"

# Restart backend
echo "🔄 Restarting backend..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml up -d backend"

# Wait for startup
echo "⏳ Waiting 20 seconds for backend to start..."
sleep 20

# Check Position Monitor logs
echo ""
echo "📊 Checking Position Monitor logs..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "docker logs quantum_backend 2>&1 | tail -50 | grep -E '(POSITION-MONITOR|TP|SL)' || echo '⚠️ No Position Monitor logs yet'"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🔍 Monitor Position Monitor activity:"
echo "   ssh -i $SSH_KEY $VPS_USER@$VPS_IP \"docker logs -f quantum_backend | grep POSITION-MONITOR\""
