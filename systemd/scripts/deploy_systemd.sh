#!/bin/bash
# Deploy Quantum Trader with systemd
# Run as root on target Linux system

set -euo pipefail

echo "🚀 Quantum Trader - Systemd Deployment"
echo "======================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run as root"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "📂 Project root: $PROJECT_ROOT"
echo ""

# STEP 1: Create directory structure
echo "📁 [STEP 1/10] Creating directory structure..."
mkdir -p /opt/quantum/{scripts,venvs,microservices}
mkdir -p /data/quantum/{models,runtime,learning,checkpoints}
mkdir -p /etc/quantum
mkdir -p /var/log/quantum
mkdir -p /run/quantum
mkdir -p /var/lib/quantum/redis

# STEP 2: Create system users
echo "👤 [STEP 2/10] Creating system users..."
bash "$SCRIPT_DIR/create_users.sh"

# STEP 3: Setup Python venvs
echo "🐍 [STEP 3/10] Setting up Python virtual environments..."
bash "$SCRIPT_DIR/setup_venvs.sh"

# STEP 4: Deploy code
echo "📦 [STEP 4/10] Deploying application code..."
rsync -av --delete \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='node_modules' \
    --exclude='.venv' \
    --exclude='venv' \
    "$PROJECT_ROOT/" /opt/quantum/

# STEP 5: Copy models
echo "🧠 [STEP 5/10] Copying ML models..."
if [ -d "$PROJECT_ROOT/models" ]; then
    rsync -av "$PROJECT_ROOT/models/" /data/quantum/models/
    echo "   ✅ Models copied"
else
    echo "   ⚠️  No models directory found - will use defaults"
fi

# STEP 6: Install systemd units
echo "⚙️ [STEP 6/10] Installing systemd units..."
cp "$PROJECT_ROOT/systemd/units/"*.service /etc/systemd/system/
cp "$PROJECT_ROOT/systemd/units/"*.target /etc/systemd/system/
echo "   ✅ Systemd units installed"

# STEP 7: Deploy configs
echo "🔧 [STEP 7/10] Deploying configs..."
cp "$PROJECT_ROOT/systemd/configs/"* /etc/quantum/ 2>/dev/null || echo "   ⚠️  No configs to copy"
cp "$PROJECT_ROOT/systemd/env-templates/"*.env /etc/quantum/ 2>/dev/null || echo "   ⚠️  No env templates found"

# STEP 8: Set permissions
echo "🔐 [STEP 8/10] Setting permissions..."
chown -R quantum-redis:quantum-redis /var/lib/quantum/redis
chown -R root:root /opt/quantum
chmod -R 755 /opt/quantum/scripts
chmod -R 755 /opt/quantum/venvs
chmod 755 /opt/quantum/systemd/scripts/*.sh 2>/dev/null || true
chmod -R 750 /etc/quantum
chmod 640 /etc/quantum/*.env 2>/dev/null || true

# Allow services to write logs
chmod 777 /var/log/quantum

echo "   ✅ Permissions set"

# STEP 9: Reload systemd
echo "🔄 [STEP 9/10] Reloading systemd daemon..."
systemctl daemon-reload
echo "   ✅ Systemd reloaded"

# STEP 10: Enable services
echo "✅ [STEP 10/10] Enabling services..."
systemctl enable quantum-redis.service
systemctl enable quantum-ceo-brain.service
systemctl enable quantum-strategy-brain.service
systemctl enable quantum-risk-brain.service
systemctl enable quantum-ai-engine.service
systemctl enable quantum-trader.target

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "   1. Edit /etc/quantum/*.env files with your API keys"
echo "   2. Run: ./start_all.sh"
echo "   3. Verify: ./verify_health.sh"
echo ""
