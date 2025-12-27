#!/bin/bash
# QuantumFond VPS Docker Deployment
# Deploys using Docker Compose to Hetzner VPS

set -e

VPS_HOST="46.224.116.254"
VPS_USER="root"
SSH_KEY="~/.ssh/hetzner_fresh"
DEPLOY_DIR="/opt/quantumfond"

echo "🐳 QuantumFond Docker Deployment to VPS"
echo "Target: $VPS_USER@$VPS_HOST"
echo "─────────────────────────────────"

# Test SSH connection
echo "→ Testing SSH connection..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "echo 'Connection successful'" || {
    echo "✗ SSH connection failed"
    exit 1
}
echo "✓ SSH connection OK"

# Install Docker on VPS if needed
echo ""
echo "→ Checking Docker installation..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
if ! command -v docker &> /dev/null; then
    echo "→ Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl start docker
    systemctl enable docker
    
    echo "→ Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    echo "✓ Docker installed"
else
    echo "✓ Docker already installed"
fi

docker --version
docker-compose --version
ENDSSH

# Create deployment directory
echo ""
echo "→ Creating deployment directory..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "mkdir -p $DEPLOY_DIR"

# Copy all necessary files
echo ""
echo "📦 Uploading files to VPS..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    --exclude '.git' \
    --exclude 'node_modules' \
    --exclude '__pycache__' \
    --exclude 'venv' \
    --exclude 'dist' \
    ./quantumfond_backend/ $VPS_USER@$VPS_HOST:$DEPLOY_DIR/quantumfond_backend/

rsync -avz --progress -e "ssh -i $SSH_KEY" \
    --exclude 'node_modules' \
    --exclude 'dist' \
    ./quantumfond_frontend/ $VPS_USER@$VPS_HOST:$DEPLOY_DIR/quantumfond_frontend/

scp -i $SSH_KEY docker-compose.quantumfond.yml $VPS_USER@$VPS_HOST:$DEPLOY_DIR/
scp -i $SSH_KEY .env.quantumfond $VPS_USER@$VPS_HOST:$DEPLOY_DIR/.env

# Deploy with Docker Compose
echo ""
echo "🚀 Starting Docker deployment on VPS..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
set -e

cd /opt/quantumfond

echo "→ Stopping existing containers..."
docker-compose -f docker-compose.quantumfond.yml down || true

echo "→ Building images..."
docker-compose -f docker-compose.quantumfond.yml build

echo "→ Starting containers..."
docker-compose -f docker-compose.quantumfond.yml up -d

echo "→ Waiting for services to start..."
sleep 10

echo "→ Checking container status..."
docker-compose -f docker-compose.quantumfond.yml ps

echo ""
echo "→ Backend logs:"
docker-compose -f docker-compose.quantumfond.yml logs backend | tail -20

echo ""
echo "→ Frontend logs:"
docker-compose -f docker-compose.quantumfond.yml logs frontend | tail -10

ENDSSH

# Verify deployment
echo ""
echo "🔍 Verifying deployment..."
sleep 5

echo "→ Testing backend health..."
curl -s http://$VPS_HOST:8000/health | jq . || echo "Backend check: $(curl -s -o /dev/null -w '%{http_code}' http://$VPS_HOST:8000/health)"

echo ""
echo "→ Testing frontend..."
curl -s -o /dev/null -w "Frontend HTTP Status: %{http_code}\n" http://$VPS_HOST/ || echo "Frontend not responding"

# Show container status
echo ""
echo "📊 Container Status:"
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "cd $DEPLOY_DIR && docker-compose -f docker-compose.quantumfond.yml ps"

echo ""
echo "🎉 Docker Deployment Complete!"
echo "─────────────────────────────────"
echo "Services running on VPS:"
echo "  • Backend API: http://$VPS_HOST:8000"
echo "  • Frontend: http://$VPS_HOST"
echo "  • API Docs: http://$VPS_HOST:8000/docs"
echo "  • Database: localhost:5432"
echo "  • Redis: localhost:6379"
echo ""
echo "📝 Useful commands:"
echo "ssh -i $SSH_KEY $VPS_USER@$VPS_HOST 'cd $DEPLOY_DIR && docker-compose -f docker-compose.quantumfond.yml logs -f'"
echo "ssh -i $SSH_KEY $VPS_USER@$VPS_HOST 'cd $DEPLOY_DIR && docker-compose -f docker-compose.quantumfond.yml ps'"
echo "ssh -i $SSH_KEY $VPS_USER@$VPS_HOST 'cd $DEPLOY_DIR && docker-compose -f docker-compose.quantumfond.yml restart backend'"
