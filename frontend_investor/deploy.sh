#!/bin/bash

# QuantumFond Investor Portal Deployment Script
# Deploys to investor.quantumfond.com on VPS

set -e

echo "🚀 QuantumFond Investor Portal Deployment"
echo "=========================================="

# Configuration
VPS_HOST="46.224.116.254"
VPS_USER="root"
SSH_KEY="~/.ssh/hetzner_fresh"
REMOTE_DIR="/home/qt/quantum_trader/frontend_investor"
DOMAIN="investor.quantumfond.com"

echo "📦 Step 1: Building Next.js application..."
npm install
npm run build

echo "📤 Step 2: Uploading to VPS..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST "mkdir -p $REMOTE_DIR"

# Upload build files
tar -czf investor_build.tar.gz .next package.json package-lock.json next.config.js
scp -i $SSH_KEY investor_build.tar.gz $VPS_USER@$VPS_HOST:$REMOTE_DIR/

# Extract and setup
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
cd /home/qt/quantum_trader/frontend_investor
tar -xzf investor_build.tar.gz
npm install --production
pm2 delete quantumfond-investor 2>/dev/null || true
pm2 start npm --name "quantumfond-investor" -- start
pm2 save
ENDSSH

echo "🌐 Step 3: Configuring Nginx..."
ssh -i $SSH_KEY $VPS_USER@$VPS_HOST << 'ENDSSH'
cat > /etc/nginx/sites-available/investor.quantumfond.com << 'EOF'
server {
    listen 80;
    listen [::]:80;
    server_name investor.quantumfond.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name investor.quantumfond.com;

    ssl_certificate /etc/letsencrypt/live/quantumfond.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/quantumfond.com/privkey.pem;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

ln -sf /etc/nginx/sites-available/investor.quantumfond.com /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx
ENDSSH

echo "✅ Deployment Complete!"
echo "🌐 Investor Portal: https://$DOMAIN"
echo ""
echo "📋 Post-deployment checklist:"
echo "  - Test login at https://$DOMAIN/login"
echo "  - Verify API connectivity"
echo "  - Check all pages load correctly"
echo "  - Test report downloads"

# Cleanup
rm -f investor_build.tar.gz

echo ""
echo ">>> [Phase 22 Complete – Investor Portal & Reporting Layer Operational on investor.quantumfond.com]"
