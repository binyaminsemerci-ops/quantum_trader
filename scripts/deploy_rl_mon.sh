cd /home/qt/quantum_trader
echo "📥 Pulling latest changes..."
git pull origin main
echo ""
echo "🔨 Building RL monitor..."
docker compose -f docker-compose.vps.yml build rl-monitor
echo ""
echo "▶️  Starting service..."
docker compose -f docker-compose.vps.yml up -d rl-monitor
echo ""
echo "Waiting 5 seconds..."
sleep 5
echo ""
echo "=== Container Status ==="
docker ps --format 'table {{.Names}}\t{{.Status}}' | grep -E 'rl_monitor|rl_sizing'
echo ""
echo "=== RL Monitor Logs ==="
docker logs quantum_rl_monitor --tail 15 2>&1
echo ""
echo "✅ Done!"
