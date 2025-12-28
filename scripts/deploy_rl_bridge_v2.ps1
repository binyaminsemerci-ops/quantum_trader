# Deploy RL Feedback Bridge v2
$ErrorActionPreference = "Stop"

Write-Host "🧠 Deploying RL Feedback Bridge v2..." -ForegroundColor Cyan

# Create directory on VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'mkdir -p /home/qt/quantum_trader/microservices/rl_feedback_bridge_v2'

# Copy files
wsl bash -c 'scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/microservices/rl_feedback_bridge_v2/bridge_v2.py root@46.224.116.254:/home/qt/quantum_trader/microservices/rl_feedback_bridge_v2/'
wsl bash -c 'scp -i ~/.ssh/hetzner_fresh /mnt/c/quantum_trader/microservices/rl_feedback_bridge_v2/Dockerfile root@46.224.116.254:/home/qt/quantum_trader/microservices/rl_feedback_bridge_v2/'

# Add to docker-compose if not present
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 @'
cd /home/qt/quantum_trader
if ! grep -q "rl-feedback-v2" docker-compose.vps.yml; then
  cat >> docker-compose.vps.yml <<'YAML'

  rl-feedback-v2:
    build: ./microservices/rl_feedback_bridge_v2
    container_name: quantum_rl_feedback_v2
    environment:
      - REDIS_HOST=redis
    volumes:
      - ./models:/models
    depends_on:
      - redis
      - ai-engine
      - model-federation
    networks:
      - quantum_trader
    restart: always
YAML
fi
'@

# Build and start
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml build rl-feedback-v2'
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cd /home/qt/quantum_trader && docker compose -f docker-compose.vps.yml up -d rl-feedback-v2'

Start-Sleep -Seconds 10

# Verify
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker ps --filter name=quantum_rl_feedback_v2'
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'docker logs quantum_rl_feedback_v2 --tail 15'

Write-Host "`n✅ RL Feedback Bridge v2 deployed!" -ForegroundColor Green
