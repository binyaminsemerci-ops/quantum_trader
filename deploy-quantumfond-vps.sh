#!/bin/bash
# QuantumFond VPS Deployment Script
# Deploys to Hetzner VPS: 46.224.116.254

set -e

VPS_HOST="46.224.116.254"
VPS_USER="root"
SSH_KEY="~/.ssh/hetzner_fresh"
DEPLOY_DIR="/opt/quantumfond"
BACKUP_DIR="/opt/quantumfond/backups"

echo "🚀 QuantumFond VPS Deployment"
echo "Target: $VPS_USER@$VPS_HOST"
echo "─────────────────────────────────"

# Test SSH connection
echo "→ Testing SSH connection..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "echo 'Connection successful'" || {
    echo "✗ SSH connection failed"
    exit 1
}
echo "✓ SSH connection OK"

# Create deployment directory on VPS
echo ""
echo "→ Creating deployment directories..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "mkdir -p $DEPLOY_DIR/{backend,frontend,backups}"

# Copy backend files
echo ""
echo "📦 Uploading backend files..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    ./quantumfond_backend/ $VPS_USER@$VPS_HOST:$DEPLOY_DIR/backend/

# Copy frontend files
echo ""
echo "📦 Uploading frontend files..."
rsync -avz --progress -e "ssh -i $SSH_KEY" \
    --exclude 'node_modules' \
    --exclude 'dist' \
    ./quantumfond_frontend/ $VPS_USER@$VPS_HOST:$DEPLOY_DIR/frontend/

# Copy deployment configs
echo ""
echo "📦 Uploading deployment configs..."
scp -i $SSH_KEY nginx-quantumfond.conf $VPS_USER@$VPS_HOST:$DEPLOY_DIR/
scp -i $SSH_KEY quantumfond-backend.service $VPS_USER@$VPS_HOST:$DEPLOY_DIR/
scp -i $SSH_KEY .env.quantumfond $VPS_USER@$VPS_HOST:$DEPLOY_DIR/backend/.env

# Run deployment commands on VPS
echo ""
echo "🔧 Running deployment on VPS..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
set -e

DEPLOY_DIR="/opt/quantumfond"
cd $DEPLOY_DIR

echo "→ Installing system dependencies..."
apt-get update
apt-get install -y python3-pip python3-venv nodejs npm postgresql postgresql-contrib nginx certbot python3-certbot-nginx

echo "→ Setting up PostgreSQL..."
systemctl start postgresql
systemctl enable postgresql

# Create database and user if not exists
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = 'quantumdb'" | grep -q 1 || \
sudo -u postgres psql -c "CREATE DATABASE quantumdb;"

sudo -u postgres psql -tc "SELECT 1 FROM pg_roles WHERE rolname='quantumfond'" | grep -q 1 || \
sudo -u postgres psql -c "CREATE USER quantumfond WITH PASSWORD 'QuantumFond2025!';"

sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE quantumdb TO quantumfond;"

echo "→ Setting up backend..."
cd $DEPLOY_DIR/backend

# Create virtual environment
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Install dependencies
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

# Initialize database
venv/bin/python -c "from db.connection import init_db; init_db()" || echo "Database already initialized"

# Install systemd service
cp $DEPLOY_DIR/quantumfond-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable quantumfond-backend
systemctl restart quantumfond-backend

echo "→ Setting up frontend..."
cd $DEPLOY_DIR/frontend

# Install dependencies and build
npm install
npm run build

# Copy to nginx directory
mkdir -p /var/www/quantumfond/frontend
cp -r dist/* /var/www/quantumfond/frontend/
chown -R www-data:www-data /var/www/quantumfond

echo "→ Configuring nginx..."
cp $DEPLOY_DIR/nginx-quantumfond.conf /etc/nginx/sites-available/quantumfond

# Enable site
if [ ! -L /etc/nginx/sites-enabled/quantumfond ]; then
    ln -s /etc/nginx/sites-available/quantumfond /etc/nginx/sites-enabled/
fi

# Remove default site if exists
rm -f /etc/nginx/sites-enabled/default

# Test nginx config
nginx -t

# Reload nginx
systemctl reload nginx

echo ""
echo "✅ VPS Deployment Complete!"
echo "─────────────────────────────────"

# Check service status
echo "→ Backend service status:"
systemctl status quantumfond-backend --no-pager | head -10

echo ""
echo "→ Testing backend health..."
sleep 3
curl -s http://localhost:8000/health || echo "Backend not responding yet"

echo ""
echo "→ Testing frontend..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost/ || echo "Frontend not responding yet"

ENDSSH

# Final verification from local machine
echo ""
echo "🔍 Verifying deployment..."
echo "→ Testing backend from external..."
curl -s http://$VPS_HOST:8000/health || echo "Backend not accessible externally yet"

echo ""
echo "→ Testing frontend from external..."
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://$VPS_HOST/ || echo "Frontend not accessible externally yet"

echo ""
echo "🎉 Deployment Complete!"
echo "─────────────────────────────────"
echo "Backend API: http://$VPS_HOST:8000"
echo "Frontend: http://$VPS_HOST"
echo "API Docs: http://$VPS_HOST:8000/docs"
echo ""
echo "📝 Next steps:"
echo "1. Configure domain DNS to point to $VPS_HOST"
echo "2. Setup SSL: ssh -i $SSH_KEY $VPS_USER@$VPS_HOST"
echo "   sudo certbot --nginx -d api.quantumfond.com -d app.quantumfond.com"
echo "3. Update .env file on VPS with production settings"
echo "4. Test the application"
echo ""
echo "📊 View logs:"
echo "ssh -i $SSH_KEY $VPS_USER@$VPS_HOST 'journalctl -u quantumfond-backend -f'"
