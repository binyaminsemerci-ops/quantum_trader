#!/bin/bash
# Check backend status on VPS

VPS_IP="46.224.116.254"
VPS_USER="qt"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

echo "📊 Checking backend status..."
ssh -i $SSH_KEY $VPS_USER@$VPS_IP << 'ENDSSH'
echo "Service status:"
systemctl status quantum-backend.service --no-pager | head -10

echo ""
echo "Recent logs (last 50 lines):"
journalctl -u quantum-backend.service -n 50 --no-pager

echo ""
echo "Health check:"
curl -s http://localhost:8000/health || echo "❌ Backend not responding"
ENDSSH
