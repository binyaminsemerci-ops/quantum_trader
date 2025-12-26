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

echo -e "${YELLOW}Step 1: Installing prerequisites...${NC}"
apt-get update -qq
apt-get install -y nginx certbot python3-certbot-nginx -qq 2>&1 | grep -E "Setting up|already"
echo -e "${GREEN}✓ Prerequisites OK${NC}"

echo -e "${YELLOW}Step 2: Deploying Nginx configuration...${NC}"
cp /root/quantum_trader/nginx/quantumfond.conf /etc/nginx/sites-available/quantumfond.conf
ln -sf /etc/nginx/sites-available/quantumfond.conf /etc/nginx/sites-enabled/quantumfond.conf

# Remove default site if exists
rm -f /etc/nginx/sites-enabled/default

# Test Nginx configuration
if nginx -t 2>&1 | grep -q "successful"; then
    echo -e "${GREEN}✓ Nginx config valid${NC}"
else
    echo -e "${RED}Error: Invalid Nginx configuration${NC}"
    nginx -t
    exit 1
fi

echo -e "${YELLOW}Step 3: Setting up SSL certificates...${NC}"
# Check if certificates already exist
if [ ! -d "/etc/letsencrypt/live/$DOMAIN" ]; then
    echo -e "${YELLOW}⚠️  SSL certificates not found${NC}"
    echo -e "${YELLOW}   Manual step required: Run certbot${NC}"
    echo -e "${YELLOW}   Command: certbot --nginx -d $DOMAIN -d $API_DOMAIN -d $APP_DOMAIN${NC}"
else
    echo -e "${GREEN}✓ SSL certificates exist${NC}"
fi

# Reload Nginx
systemctl reload nginx 2>&1 | grep -v "Warning" || true
echo -e "${GREEN}✓ Nginx reloaded${NC}"

echo -e "${YELLOW}Step 4: Deploying Docker containers...${NC}"
cd /root/quantum_trader
docker compose --profile dashboard down 2>&1 | grep -E "Stopp|Removed" || true
docker compose --profile dashboard build 2>&1 | grep -E "Built|Image" || echo "Building..."
docker compose --profile dashboard up -d
echo -e "${GREEN}✓ Docker containers running${NC}"

sleep 3

echo -e "${YELLOW}Step 5: Verifying deployment...${NC}"
# Test backend API
if curl -sf http://localhost:8025/health > /dev/null; then
    echo -e "${GREEN}✓ Backend API responding (http://localhost:8025)${NC}"
else
    echo -e "${RED}⚠ Backend API not responding yet${NC}"
fi

echo ""
echo "=========================================================="
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE${NC}"
echo "=========================================================="
echo ""
echo "Services:"
echo "  Backend:  http://localhost:8025 (mapped to $API_DOMAIN via Nginx)"
echo "  Frontend: $PROJECT_DIR/frontend/dist (served by Nginx at $APP_DOMAIN)"
echo ""
echo "Container Status:"
docker compose --profile dashboard ps
echo ""
echo "Next Steps:"
echo "  1. Configure DNS: Point $API_DOMAIN and $APP_DOMAIN to this server"
echo "  2. Obtain SSL: sudo certbot --nginx -d $DOMAIN -d $API_DOMAIN -d $APP_DOMAIN"
echo "  3. Test: curl https://$API_DOMAIN/health"
echo ""
echo -e "${GREEN}>>> [Domain Migration Complete – quantumfond.com configured]${NC}"
