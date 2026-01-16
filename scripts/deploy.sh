#!/bin/bash
# Deploy Script - Quantum Trader Production Deployment
set -e

SERVICE=${1:-all}
VPS_HOST="qt@46.224.116.254"
VPS_PATH="/home/qt/quantum_trader"
SSH_KEY="$HOME/.ssh/hetzner_fresh"

deploy() {
  local service=$1
  echo "🚀 Deploying $service..."
  
  # Sync code
  rsync -avz --delete \
    -e "ssh -i $SSH_KEY" \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'node_modules' \
    --exclude '.env*' \
    ./ $VPS_HOST:$VPS_PATH/
  
  # Rebuild and restart service
  ssh -i $SSH_KEY $VPS_HOST "cd $VPS_PATH && \
    sudo systemctl daemon-reload && \
    sudo systemctl restart quantum-$service.service"
  
  echo "✅ $service deployed"
}

case $SERVICE in
  ai-engine)
    deploy ai-engine
    ;;
  execution)
    deploy execution
    ;;
  all)
    deploy ai-engine
    deploy execution
    ;;
  *)
    echo "Usage: $0 [ai-engine|execution|all]"
    exit 1
    ;;
esac

echo "🎉 Deployment complete!"
