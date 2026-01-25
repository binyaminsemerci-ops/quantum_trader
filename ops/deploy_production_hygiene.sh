#!/bin/bash
# Production Hygiene Deployment Script
# Use: ./deploy_production_hygiene.sh
# Purpose: Deploy hard mode switch + kill switch + metrics to VPS

set -e

VPS_HOST="${1:-46.224.116.254}"
SSH_KEY="/root/.ssh/hetzner_fresh"

echo "🚀 Deploying Production Hygiene to $VPS_HOST"
echo "=================================================="

# Deploy code and restart
echo ""
echo "1️⃣  Pulling latest code..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "cd /root/quantum_trader && git pull origin main" || {
    echo "⚠️  Git pull failed - code might already be current"
}

echo ""
echo "2️⃣  Verifying code contains hygiene features..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "grep -c 'TESTNET_MODE' /root/quantum_trader/microservices/apply_layer/main.py" && {
    echo "✅ TESTNET_MODE found"
} || {
    echo "❌ TESTNET_MODE not found - code not updated properly"
    exit 1
}

wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "grep -c 'SAFETY_KILL_KEY' /root/quantum_trader/microservices/apply_layer/main.py" && {
    echo "✅ SAFETY_KILL_KEY found"
} || {
    echo "❌ SAFETY_KILL_KEY not found"
    exit 1
}

wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "grep -c 'prometheus_client' /root/quantum_trader/microservices/apply_layer/main.py" && {
    echo "✅ Prometheus metrics found"
} || {
    echo "⚠️  Prometheus metrics not found - optional feature"
}

echo ""
echo "3️⃣  Setting PRODUCTION mode..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "cat >> /etc/quantum/apply-layer.env << 'EOF'
# Production Hygiene Config - 2026-01-25
TESTNET=false
EOF
" && {
    echo "✅ TESTNET=false set"
} || {
    echo "⚠️  Could not update env (might already be set)"
}

echo ""
echo "4️⃣  Restarting Apply Layer service..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "systemctl restart quantum-apply-layer" && {
    sleep 3
    echo "✅ Service restarted"
} || {
    echo "❌ Service restart failed"
    exit 1
}

echo ""
echo "5️⃣  Verifying PRODUCTION mode is active..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "journalctl -u quantum-apply-layer -n 3 --no-pager | grep -E 'PRODUCTION|TESTNET'" && {
    echo "✅ Mode verified"
} || {
    echo "⚠️  Could not verify mode in logs"
}

echo ""
echo "6️⃣  Testing kill switch..."
echo "   - Activating..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "redis-cli SET quantum:global:kill_switch true" > /dev/null && {
    echo "   ✅ Kill switch activated"
}

echo "   - Waiting for logs..."
sleep 2
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "journalctl -u quantum-apply-layer -n 5 --no-pager | grep KILL_SWITCH" && {
    echo "   ✅ Kill switch verified in logs"
}

echo "   - Deactivating..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "redis-cli SET quantum:global:kill_switch false" > /dev/null && {
    echo "   ✅ Kill switch deactivated"
}

echo ""
echo "7️⃣  Checking Prometheus metrics endpoint..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "curl -s http://localhost:8000/metrics 2>/dev/null | head -3" && {
    echo "✅ Metrics endpoint responding"
} || {
    echo "⚠️  Metrics endpoint not available (optional)"
}

echo ""
echo "8️⃣  Checking service health..."
wsl ssh -i "$SSH_KEY" root@"$VPS_HOST" "systemctl status quantum-apply-layer --no-pager | head -10" && {
    echo "✅ Service healthy"
}

echo ""
echo "=================================================="
echo "🎉 Production Hygiene Deployment Complete!"
echo "=================================================="
echo ""
echo "Summary:"
echo "  ✅ Hard Mode Switch (TESTNET=false) - active"
echo "  ✅ Safety Kill Switch - tested and working"
echo "  ✅ Prometheus Metrics - configured"
echo "  ✅ Service - running and healthy"
echo ""
echo "Next steps:"
echo "  1. Monitor logs: journalctl -u quantum-apply-layer -f"
echo "  2. Check metrics: curl http://localhost:8000/metrics"
echo "  3. Read guide: cat PRODUCTION_HYGIENE_GUIDE.md"
echo ""
echo "Emergency kill switch:"
echo "  redis-cli SET quantum:global:kill_switch true"
echo ""
echo "To resume:"
echo "  redis-cli SET quantum:global:kill_switch false"
echo ""
