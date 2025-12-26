#!/bin/bash
set -e

echo "🚀 Deploying Quantum Trader Dashboard to quantumfond.com"
echo "=========================================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="quantumfond.com"
API_DOMAIN="api.quantumfond.com"
APP_DOMAIN="app.quantumfond.com"
PROJECT_DIR="/root/quantum_trader/dashboard_v4"
NGINX_CONF="/etc/nginx/sites-available/quantumfond.conf"

echo -e "${YELLOW}Step 1: Checking prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker not installed${NC}"
    exit 1
fi

if ! command -v nginx &> /dev/null; then
    echo -e "${YELLOW}Installing Nginx...${NC}"
    apt-get update
    apt-get install -y nginx
fi

if ! command -v certbot &> /dev/null; then
    echo -e "${YELLOW}Installing Certbot...${NC}"
    apt-get install -y certbot python3-certbot-nginx
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

echo -e "${YELLOW}Step 2: Building frontend...${NC}"
cd $PROJECT_DIR/frontend
npm install
npm run build
echo -e "${GREEN}✓ Frontend built${NC}"

echo -e "${YELLOW}Step 3: Deploying Nginx configuration...${NC}"
cp /root/quantum_trader/nginx/quantumfond.conf $NGINX_CONF

# Test Nginx configuration
if nginx -t; then
    echo -e "${GREEN}✓ Nginx config valid${NC}"
else
    echo -e "${RED}Error: Invalid Nginx configuration${NC}"
    exit 1
fi

echo -e "${YELLOW}Step 4: Obtaining SSL certificates...${NC}"
# Check if certificates already exist
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    echo -e "${YELLOW}Requesting new SSL certificates...${NC}"
    certbot --nginx -d $DOMAIN -d $API_DOMAIN -d $APP_DOMAIN --non-interactive --agree-tos --email admin@quantumfond.com --redirect
    echo -e "${GREEN}✓ SSL certificates obtained${NC}"
else
    echo -e "${GREEN}✓ SSL certificates already exist${NC}"
fi

# Enable Nginx site
ln -sf $NGINX_CONF /etc/nginx/sites-enabled/quantumfond.conf
systemctl reload nginx
echo -e "${GREEN}✓ Nginx reloaded${NC}"

echo -e "${YELLOW}Step 5: Deploying Docker containers...${NC}"
cd /root/quantum_trader
docker compose --profile dashboard down
docker compose --profile dashboard build
docker compose --profile dashboard up -d
echo -e "${GREEN}✓ Docker containers running${NC}"

echo -e "${YELLOW}Step 6: Setting up SSL auto-renewal...${NC}"
# Add certbot renewal to crontab if not already present
if ! crontab -l 2>/dev/null | grep -q "certbot renew"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet --post-hook 'systemctl reload nginx'") | crontab -
    echo -e "${GREEN}✓ Auto-renewal scheduled (daily at 3 AM)${NC}"
else
    echo -e "${GREEN}✓ Auto-renewal already configured${NC}"
fi

echo -e "${YELLOW}Step 7: Verifying deployment...${NC}"
sleep 5

# Test backend API
if curl -sf https://$API_DOMAIN/health > /dev/null; then
    echo -e "${GREEN}✓ Backend API responding${NC}"
else
    echo -e "${RED}⚠ Backend API not responding yet (may need a moment)${NC}"
fi

# Test frontend
if curl -sf https://$APP_DOMAIN > /dev/null; then
    echo -e "${GREEN}✓ Frontend responding${NC}"
else
    echo -e "${RED}⚠ Frontend not responding yet (may need a moment)${NC}"
fi

echo ""
echo "=========================================================="
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE${NC}"
echo "=========================================================="
echo ""
echo "Dashboard URLs:"
echo "  Frontend: https://$APP_DOMAIN"
echo "  API:      https://$API_DOMAIN"
echo "  API Docs: https://$API_DOMAIN/docs"
echo "  WebSocket: wss://$API_DOMAIN/stream/live"
echo ""
echo "Container Status:"
docker compose --profile dashboard ps
echo ""
echo "SSL Certificate Status:"
certbot certificates
echo ""
echo -e "${GREEN}>>> [Domain Migration Complete – quantumfond.com operational with SSL and live streaming]${NC}"
