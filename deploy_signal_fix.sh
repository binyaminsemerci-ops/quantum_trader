#!/bin/bash
set -e

echo "🚀 Deploying signal reading fix to auto executor..."
echo ""

# Copy updated executor service
echo "📤 Uploading executor_service.py..."
docker cp /tmp/executor_service_fixed.py quantum_auto_executor:/app/executor_service.py

echo "♻️  Restarting auto executor..."
docker restart quantum_auto_executor

echo "⏳ Waiting for startup..."
sleep 5

echo "📋 Checking logs..."
docker logs quantum_auto_executor --tail 30

echo ""
echo "✅ Deployment complete!"
echo "Monitor with: docker logs quantum_auto_executor -f"
