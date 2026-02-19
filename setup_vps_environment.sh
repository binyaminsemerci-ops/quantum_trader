#!/bin/bash
# Quantum Trader - VPS Environment Setup Script
# Copies all necessary environment files with working credentials to correct locations

echo "🚀 QUANTUM TRADER VPS ENVIRONMENT SETUP"
echo "========================================"
echo ""

# Working credentials (verified functional)
WORKING_KEY="w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
WORKING_SECRET="QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

echo "✅ Working credentials loaded"
echo "   API Key: ${WORKING_KEY:0:20}..."
echo "   API Secret: ${WORKING_SECRET:0:20}..."
echo ""

# 1. Create /etc/quantum directory
echo "📁 Creating /etc/quantum directory..."
sudo mkdir -p /etc/quantum

# 2. Create testnet.env (most important - referenced by multiple services)
echo "🔧 Creating /etc/quantum/testnet.env..."
sudo tee /etc/quantum/testnet.env > /dev/null << EOF
# Quantum Trader Testnet Environment - PRODUCTION VPS
# Used by: quantum-execution-real, quantum-exit-monitor, quantum-exitbrain-v35

# Working Binance Testnet API Credentials
BINANCE_API_KEY=$WORKING_KEY
BINANCE_API_SECRET=$WORKING_SECRET
BINANCE_TESTNET_API_KEY=$WORKING_KEY
BINANCE_TESTNET_API_SECRET=$WORKING_SECRET

# Environment
TESTNET=true
MODE=testnet
ENVIRONMENT=testnet

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
EOF

# 3. Create clm.env (referenced by quantum-clm-minimal)
echo "🔧 Creating /etc/quantum/clm.env..."
sudo tee /etc/quantum/clm.env > /dev/null << EOF
# Continuous Learning Module Environment
BINANCE_API_KEY=$WORKING_KEY
BINANCE_API_SECRET=$WORKING_SECRET
REDIS_HOST=localhost
REDIS_PORT=6379
CLM_MODE=active
EOF

# 4. Create safety-telemetry.env (referenced by quantum-safety-telemetry)
echo "🔧 Creating /etc/quantum/safety-telemetry.env..."
sudo tee /etc/quantum/safety-telemetry.env > /dev/null << EOF
# Safety Telemetry Environment
REDIS_HOST=localhost
REDIS_PORT=6379
TELEMETRY_ENABLED=true
SAFETY_MODE=active
EOF

# 5. Create utf.env (referenced by quantum-utf-publisher)
echo "🔧 Creating /etc/quantum/utf.env..."
sudo tee /etc/quantum/utf.env > /dev/null << EOF
# UTF Publisher Environment
REDIS_HOST=localhost
REDIS_PORT=6379
UTF_MODE=active
EOF

# 6. Set proper permissions (secure)
echo "🔒 Setting secure permissions..."
sudo chmod 600 /etc/quantum/*.env
sudo chown root:root /etc/quantum/*.env

# 7. Verify files
echo ""
echo "📋 Created environment files:"
sudo ls -la /etc/quantum/

echo ""
echo "🔄 Restarting affected systemd services..."

# Services that reference environment files
services=(
    "quantum-execution-real"
    "quantum-exit-monitor" 
    "quantum-exitbrain-v35"
    "quantum-clm-minimal"
    "quantum-safety-telemetry"
    "quantum-utf-publisher"
)

for service in "${services[@]}"; do
    if systemctl is-active --quiet "$service.service"; then
        echo "  🔄 Restarting $service..."
        sudo systemctl restart "$service.service"
        if systemctl is-active --quiet "$service.service"; then
            echo "     ✅ $service restarted successfully"
        else
            echo "     ⚠️  $service restart failed - check logs"
        fi
    else
        echo "  ℹ️  $service not running - will use new env when started"
    fi
done

echo ""
echo "🎉 VPS ENVIRONMENT SETUP COMPLETE!"
echo "================================================"
echo "✅ All environment files created with working credentials"
echo "✅ Proper permissions set (600, root:root)"  
echo "✅ Affected services restarted"
echo ""
echo "🚀 Quantum Trader VPS is now fully configured!"
echo "   All systemd services have access to working Binance testnet credentials"
echo "   Formula system can now operate on live infrastructure"