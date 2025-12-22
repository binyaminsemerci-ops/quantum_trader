#!/bin/bash
# Manual deployment of Position Monitor

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo "🔍 Step 1: Check git status on VPS..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && git status"

echo ""
echo "🔧 Step 2: Fix git permissions..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && sudo chown -R qt:qt .git && chmod -R 755 .git"

echo ""
echo "📥 Step 3: Git reset and pull..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && git reset --hard origin/main && git pull origin main"

echo ""
echo "🔨 Step 4: Rebuild backend..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml build backend 2>&1 | tail -20"

echo ""
echo "🔄 Step 5: Restart backend..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "cd ~/quantum_trader && docker compose -f docker-compose.vps.yml up -d backend"

echo ""
echo "⏳ Waiting 25 seconds for startup..."
sleep 25

echo ""
echo "📊 Step 6: Check Position Monitor logs..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP "docker logs quantum_backend 2>&1 | grep -E '(POSITION-MONITOR|position_monitor|protection)' | tail -20"

echo ""
echo "✅ Done! Check above for Position Monitor activity."
