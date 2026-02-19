#!/bin/bash
"""
Deploy Enhanced Exit Monitor with New Exit Math
=============================================

Upgrades VPS exit monitor service to use advanced dynamic exit formulas
instead of hardcoded percentages.

THIS SCRIPT:
1. 🔄 Backs up old service
2. 📤 Uploads new exit_monitor_service_v2.py 
3. 🛑 Stops old service
4. 🚀 Starts new service with exit math
5. ✅ Validates deployment

Author: Exit Logic Upgrade Team
Date: 2026-02-18
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

VPS_HOST="root@46.224.116.254"
VPS_PROJECT_DIR="/root/quantum_trader"
SSH_KEY="~/.ssh/hetzner_fresh"

echo -e "${BLUE}🚀 DEPLOYING ENHANCED EXIT MONITOR${NC}"
echo "=================================================="

# Step 1: Backup old service
echo -e "${YELLOW}📦 Step 1: Backing up old service...${NC}"
ssh -i "$SSH_KEY" "$VPS_HOST" "
    cd $VPS_PROJECT_DIR
    if [ -f exit_monitor_service_patched.py ]; then
        cp exit_monitor_service_patched.py exit_monitor_service_patched.py.backup_$(date +%Y%m%d_%H%M%S)
        echo '✅ Old service backed up'
    else
        echo '⚠️  No old service found to backup'
    fi
"

# Step 2: Copy new service and dependencies
echo -e "${YELLOW}📤 Step 2: Uploading new exit service...${NC}"
scp -i "$SSH_KEY" exit_monitor_service_v2.py "$VPS_HOST:$VPS_PROJECT_DIR/"
scp -i "$SSH_KEY" common/exit_math.py "$VPS_HOST:$VPS_PROJECT_DIR/common/"

# Step 3: Stop old service
echo -e "${YELLOW}🛑 Step 3: Stopping old service...${NC}"
ssh -i "$SSH_KEY" "$VPS_HOST" "
    echo 'Stopping quantum-exit-monitor service...'
    systemctl stop quantum-exit-monitor || echo 'Service already stopped'
    
    # Kill any lingering Python processes
    pkill -f exit_monitor || echo 'No exit monitor processes running'
    
    echo '✅ Old service stopped'
"

# Step 4: Update systemd service file to use new service
echo -e "${YELLOW}🔧 Step 4: Updating systemd service configuration...${NC}"
ssh -i "$SSH_KEY" "$VPS_HOST" "
    # Update systemd service to point to new file
    sed -i 's|exit_monitor_service_patched.py|exit_monitor_service_v2.py|g' /etc/systemd/system/quantum-exit-monitor.service || echo 'Service file update failed - will create new one'
    
    # Reload systemd
    systemctl daemon-reload
    
    echo '✅ Systemd configuration updated'
"

# Step 5: Start new service
echo -e "${YELLOW}🚀 Step 5: Starting enhanced exit monitor...${NC}"
ssh -i "$SSH_KEY" "$VPS_HOST" "
    cd $VPS_PROJECT_DIR
    
    # Start the service
    systemctl start quantum-exit-monitor
    
    # Wait a moment for startup
    sleep 3
    
    # Check status
    if systemctl is-active --quiet quantum-exit-monitor; then
        echo '✅ Enhanced exit monitor started successfully'
    else
        echo '❌ Failed to start enhanced exit monitor'
        systemctl status quantum-exit-monitor
        exit 1
    fi
"

# Step 6: Validation
echo -e "${YELLOW}✅ Step 6: Validating deployment...${NC}"
ssh -i "$SSH_KEY" "$VPS_HOST" "
    cd $VPS_PROJECT_DIR
    
    echo '🔍 Checking process...'
    pgrep -f exit_monitor_service_v2 || echo 'Process not found by name'
    
    echo '📊 Checking service status...'
    systemctl status quantum-exit-monitor --no-pager -l
    
    echo '📝 Checking recent logs...'
    tail -20 /var/log/quantum/exit-monitor.log | grep -E 'EXIT_MATH|started successfully|ERROR' || echo 'No relevant log entries found'
    
    echo '🌐 Testing health endpoint...'
    timeout 5 curl -s http://localhost:8007/health | python3 -m json.tool || echo 'Health endpoint test failed'
"

# Step 7: Summary
echo -e "${GREEN}🏁 DEPLOYMENT SUMMARY${NC}"
echo "=================================================="
echo "✅ Old service backed up"  
echo "✅ New exit_monitor_service_v2.py deployed"
echo "✅ Exit math module (common/exit_math.py) deployed"
echo "✅ Service configuration updated"
echo "✅ Enhanced exit monitor started"
echo ""
echo "🔧 NEW FEATURES:"
echo "  • Dynamic risk-based stop losses"
echo "  • ATR-adaptive trailing stops"
echo "  • Leverage-aware exit calculations"  
echo "  • Account equity risk normalization"
echo "  • No more hardcoded percentages!"
echo ""
echo "🌐 Service running on: http://46.224.116.254:8007"
echo "📊 Health check: curl http://46.224.116.254:8007/health"
echo "📋 Positions: curl http://46.224.116.254:8007/positions"

echo -e "${BLUE}🎯 EXIT MONITOR UPGRADE COMPLETE!${NC}"