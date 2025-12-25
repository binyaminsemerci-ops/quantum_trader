#!/bin/bash
set -e
echo "🔨 Rebuilding Position Monitor with ExitBrain v3.5"
echo "Timestamp: $(date)"
echo ""

cd ~/quantum_trader

# 1. Stop the current container
echo "=== [Step 1/4] Stopping position_monitor ==="
docker stop quantum_position_monitor || true
docker rm quantum_position_monitor || true
echo "✅ Container stopped and removed"
echo ""

# 2. Rebuild the image
echo "=== [Step 2/4] Rebuilding image ==="
docker compose -f docker-compose.vps.yml build --no-cache position-monitor
echo "✅ Image rebuilt"
echo ""

# 3. Start the container
echo "=== [Step 3/4] Starting new container ==="
docker compose -f docker-compose.vps.yml up -d position-monitor
sleep 5
echo "✅ Container started"
echo ""

# 4. Verify ExitBrain is now available
echo "=== [Step 4/4] Verifying ExitBrain module ==="
if docker exec quantum_position_monitor ls /app/microservices/exitbrain_v3_5 > /dev/null 2>&1; then
  echo "✅ ExitBrain v3.5 module found in container!"
  docker exec quantum_position_monitor ls -la /app/microservices/exitbrain_v3_5/
else
  echo "❌ ExitBrain v3.5 still missing in container"
  echo "Checking what's available:"
  docker exec quantum_position_monitor ls -la /app/microservices/
fi
echo ""

echo "=== [Complete] ==="
echo "Position Monitor rebuilt and running"
echo "ExitBrain v3.5 should now be integrated"
