#!/bin/bash
# Activate ExitBrain v3.5 on VPS

set -e
cd ~/quantum_trader

echo "🚀 ExitBrain v3.5 Activation Script"
echo "===================================="
echo ""

# 1️⃣ Enable ExitBrain v3.5 via environment variable
echo "📝 Step 1: Enabling ExitBrain v3.5..."
if ! grep -q "EXIT_BRAIN_V35_ENABLED" docker-compose.vps.yml; then
    echo "Adding EXIT_BRAIN_V35_ENABLED=true to docker-compose.vps.yml..."
    # This would need to be done manually or via sed
    echo "⚠️  Please add to position-monitor service environment:"
    echo "    EXIT_BRAIN_V35_ENABLED: 'true'"
else
    echo "✅ EXIT_BRAIN_V35_ENABLED already in docker-compose.vps.yml"
fi

# 2️⃣ Rebuild position-monitor with updated code
echo ""
echo "🔧 Step 2: Rebuilding position-monitor..."
docker compose -f docker-compose.vps.yml build --no-cache position-monitor

# 3️⃣ Restart position-monitor
echo ""
echo "🔄 Step 3: Restarting position-monitor..."
docker compose -f docker-compose.vps.yml stop position-monitor
docker compose -f docker-compose.vps.yml up -d position-monitor

# 4️⃣ Wait for startup
echo ""
echo "⏳ Step 4: Waiting for container startup..."
sleep 12

# 5️⃣ Validate ExitBrain v3.5 activation
echo ""
echo "🔍 Step 5: Validating ExitBrain v3.5..."
echo "========================================"

# Check for v3.5 initialization logs
echo ""
echo "📊 Checking initialization logs..."
if docker logs --tail=50 quantum_position_monitor 2>&1 | grep -q "EXIT_BRAIN_V3.5.*ACTIVE"; then
    echo "✅ ExitBrain v3.5 ACTIVE confirmed!"
    docker logs --tail=50 quantum_position_monitor 2>&1 | grep "EXIT_BRAIN_V3.5"
else
    echo "⚠️  ExitBrain v3.5 not found in logs - checking availability..."
    docker logs --tail=100 quantum_position_monitor 2>&1 | grep -i "exit.*brain" | head -20
fi

# Check container status
echo ""
echo "📦 Container status:"
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep position_monitor

echo ""
echo "✅ Deployment complete!"
echo ""
echo "To verify ExitBrain v3.5 is processing positions:"
echo "  docker logs -f quantum_position_monitor | grep EXIT_BRAIN_V3.5"
echo ""
echo "To test with injected position:"
echo "  bash test_exitbrain_v3_5_integration.sh"
