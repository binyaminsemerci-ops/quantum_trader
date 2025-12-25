#!/bin/bash
echo "🔧 REVERSING MY INCORRECT CHANGES ON VPS..."
echo ""

cd /home/qt/quantum_trader

echo "1️⃣ Reverting executor_service.py from git..."
git checkout backend/microservices/auto_executor/executor_service.py
echo "✅ Reverted from git"
echo ""

echo "2️⃣ Copying back to container..."
docker cp backend/microservices/auto_executor/executor_service.py quantum_auto_executor:/app/
echo "✅ Copied to container"
echo ""

echo "3️⃣ Restarting quantum_auto_executor..."
docker restart quantum_auto_executor
echo "✅ Restarted"
echo ""

echo "⏳ Waiting 5 seconds for startup..."
sleep 5
echo ""

echo "📋 Checking logs (last 20 lines)..."
docker logs quantum_auto_executor --tail 20
echo ""

echo "✅ DONE! Original code restored!"
