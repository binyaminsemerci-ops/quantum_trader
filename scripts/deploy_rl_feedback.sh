#!/bin/bash
set -e

echo "🧠 Deploying RL Feedback Bridge..."

# Check if service already exists in docker-compose
if ! grep -q "rl-feedback:" docker-compose.vps.yml; then
    echo "Adding rl-feedback service to docker-compose.vps.yml..."
    cat >> docker-compose.vps.yml <<EOF

  rl-feedback:
    build: ./microservices/rl_feedback_bridge
    container_name: quantum_rl_feedback_bridge
    environment:
      - REDIS_HOST=redis
    depends_on:
      - redis
      - ai-engine
    restart: always
EOF
else
    echo "rl-feedback service already exists in docker-compose.vps.yml"
fi

echo "Building rl-feedback container..."
docker compose -f docker-compose.vps.yml build rl-feedback

echo "Starting rl-feedback container..."
docker compose -f docker-compose.vps.yml up -d rl-feedback

echo "Waiting for container to start..."
sleep 10

echo ""
echo "=== Container Status ==="
docker ps --filter name=quantum_rl_feedback_bridge --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

echo ""
echo "=== Recent Logs ==="
docker logs quantum_rl_feedback_bridge --tail 20

echo ""
echo "✅ RL Feedback Bridge deployed successfully!"
echo "🧠 Listening to: quantum:stream:exitbrain.pnl"
echo "📊 Updating actor.pth & critic.pth in real-time"
