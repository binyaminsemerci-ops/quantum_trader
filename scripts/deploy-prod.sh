#!/bin/bash

# Quantum Trader - Production Deployment Script
# This script automates the deployment process to a VPS

set -e  # Exit on any error

echo "🚀 Quantum Trader - Production Deployment"
echo "=========================================="

# Configuration
VPS_HOST="${VPS_HOST:-}"
VPS_USER="${VPS_USER:-ubuntu}"
DEPLOY_DIR="/home/$VPS_USER/quantum_trader"
BACKUP_DIR="/home/$VPS_USER/backups"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if VPS_HOST is set
if [ -z "$VPS_HOST" ]; then
    log_error "VPS_HOST environment variable not set"
    echo "Usage: VPS_HOST=your-server.com VPS_USER=ubuntu ./deploy-prod.sh"
    exit 1
fi

# Check SSH connection
log_info "Testing SSH connection to $VPS_USER@$VPS_HOST..."
if ! ssh -o ConnectTimeout=5 "$VPS_USER@$VPS_HOST" "echo 'SSH connection successful'" &>/dev/null; then
    log_error "Cannot connect to VPS via SSH"
    exit 1
fi
log_info "SSH connection successful"

# Create backup
log_info "Creating backup on VPS..."
ssh "$VPS_USER@$VPS_HOST" << 'EOF'
    if [ -d ~/quantum_trader ]; then
        BACKUP_NAME="quantum_trader_backup_$(date +%Y%m%d_%H%M%S)"
        mkdir -p ~/backups
        cp -r ~/quantum_trader ~/backups/$BACKUP_NAME
        echo "Backup created: $BACKUP_NAME"
    fi
EOF

# Upload code
log_info "Uploading code to VPS..."
rsync -avz --exclude='node_modules' \
           --exclude='.venv' \
           --exclude='*.pyc' \
           --exclude='__pycache__' \
           --exclude='.git' \
           --exclude='backend/trades.db' \
           ./ "$VPS_USER@$VPS_HOST:$DEPLOY_DIR/"

# Deploy
log_info "Running deployment on VPS..."
ssh "$VPS_USER@$VPS_HOST" << 'EOF'
    set -e
    cd ~/quantum_trader
    
    echo "� Restarting systemd services..."
    sudo systemctl daemon-reload
    
    echo "🛑 Stopping services..."
    sudo systemctl stop 'quantum-*.service'
    
    echo "🚀 Starting services..."
    sudo systemctl start 'quantum-*.service'
    
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    echo "✅ Checking health..."
    HEALTH=$(curl -s http://localhost:8000/health || echo "failed")
    echo "Backend health: $HEALTH"
    
    echo "📊 Service status:"
    systemctl list-units 'quantum-*.service' --no-pager --no-legend | head -10
EOF

# Verify deployment
log_info "Verifying deployment..."
HEALTH_CHECK=$(curl -s "http://$VPS_HOST/health" || echo "failed")

if [[ $HEALTH_CHECK == *"healthy"* ]]; then
    log_info "✅ Deployment successful!"
    log_info "🌐 Dashboard: https://$VPS_HOST"
    log_info "📊 API: https://$VPS_HOST/api"
    log_info "❤️ Health: https://$VPS_HOST/health"
else
    log_error "❌ Deployment verification failed"
    log_warn "Check logs with: ssh $VPS_USER@$VPS_HOST 'journalctl -u quantum-backend.service -n 100'"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 Deployment Complete!"
echo "=========================================="
