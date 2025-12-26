#!/bin/bash
set -e

echo "=================================================="
echo "Quantum Trader Dashboard - Production Deployment"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Verify DNS
echo -e "${YELLOW}Step 1: Verifying DNS configuration...${NC}"
API_IP=$(nslookup api.quantumfond.com 8.8.8.8 | grep -A1 "Name:" | tail -n1 | awk '{print $2}')
APP_IP=$(nslookup app.quantumfond.com 8.8.8.8 | grep -A1 "Name:" | tail -n1 | awk '{print $2}')

if [ "$API_IP" != "46.224.116.254" ] || [ "$APP_IP" != "46.224.116.254" ]; then
    echo -e "${RED}❌ DNS not configured correctly!${NC}"
    echo "Expected: 46.224.116.254"
    echo "api.quantumfond.com resolves to: $API_IP"
    echo "app.quantumfond.com resolves to: $APP_IP"
    echo ""
    echo "Please add these A records to your DNS:"
    echo "  api  →  46.224.116.254"
    echo "  app  →  46.224.116.254"
    exit 1
fi

echo -e "${GREEN}✓ DNS configured correctly${NC}"
echo ""

# Step 2: Stop conflicting nginx container
echo -e "${YELLOW}Step 2: Stopping conflicting nginx container...${NC}"
ssh root@46.224.116.254 "cd ~/quantum_trader && docker compose stop nginx"
echo -e "${GREEN}✓ Container stopped${NC}"
echo ""

# Step 3: Deploy HTTP nginx configuration
echo -e "${YELLOW}Step 3: Deploying HTTP nginx configuration...${NC}"
ssh root@46.224.116.254 "cp /root/quantum_trader/nginx/quantumfond-http.conf /etc/nginx/sites-available/quantumfond.conf && \
    ln -sf /etc/nginx/sites-available/quantumfond.conf /etc/nginx/sites-enabled/quantumfond.conf && \
    nginx -t && \
    systemctl start nginx && \
    systemctl enable nginx"
echo -e "${GREEN}✓ Nginx deployed${NC}"
echo ""

# Step 4: Test HTTP access
echo -e "${YELLOW}Step 4: Testing HTTP access...${NC}"
sleep 5
HTTP_RESPONSE=$(ssh root@46.224.116.254 "curl -s -o /dev/null -w '%{http_code}' http://api.quantumfond.com/health")
if [ "$HTTP_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✓ HTTP access working${NC}"
else
    echo -e "${RED}❌ HTTP access failed (Status: $HTTP_RESPONSE)${NC}"
    exit 1
fi
echo ""

# Step 5: Obtain SSL certificates
echo -e "${YELLOW}Step 5: Obtaining SSL certificates from Let's Encrypt...${NC}"
ssh root@46.224.116.254 "certbot --nginx \
    -d quantumfond.com \
    -d api.quantumfond.com \
    -d app.quantumfond.com \
    --non-interactive \
    --agree-tos \
    --email admin@quantumfond.com \
    --redirect"
echo -e "${GREEN}✓ SSL certificates obtained${NC}"
echo ""

# Step 6: Deploy HTTPS nginx configuration
echo -e "${YELLOW}Step 6: Deploying HTTPS nginx configuration...${NC}"
ssh root@46.224.116.254 "cp /root/quantum_trader/nginx/quantumfond.conf /etc/nginx/sites-available/quantumfond.conf && \
    nginx -t && \
    systemctl reload nginx"
echo -e "${GREEN}✓ HTTPS configuration deployed${NC}"
echo ""

# Step 7: Verify HTTPS access
echo -e "${YELLOW}Step 7: Verifying HTTPS access...${NC}"
sleep 3
HTTPS_RESPONSE=$(curl -s -o /dev/null -w '%{http_code}' https://api.quantumfond.com/health)
if [ "$HTTPS_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✓ HTTPS access working${NC}"
else
    echo -e "${RED}❌ HTTPS access failed (Status: $HTTPS_RESPONSE)${NC}"
    exit 1
fi
echo ""

# Step 8: Setup auto-renewal
echo -e "${YELLOW}Step 8: Setting up SSL auto-renewal...${NC}"
ssh root@46.224.116.254 "(crontab -l 2>/dev/null | grep -v certbot; echo '0 3 * * * certbot renew --quiet --post-hook \"systemctl reload nginx\"') | crontab -"
echo -e "${GREEN}✓ Auto-renewal configured${NC}"
echo ""

# Final verification
echo "=================================================="
echo -e "${GREEN}✓ DEPLOYMENT COMPLETE!${NC}"
echo "=================================================="
echo ""
echo "Your dashboard is now accessible at:"
echo "  Frontend: https://app.quantumfond.com"
echo "  Backend:  https://api.quantumfond.com"
echo ""
echo "Test endpoints:"
echo "  Health:      https://api.quantumfond.com/health"
echo "  AI Insights: https://api.quantumfond.com/ai/insights"
echo "  Portfolio:   https://api.quantumfond.com/portfolio"
echo "  WebSocket:   wss://api.quantumfond.com/stream/live"
echo ""
