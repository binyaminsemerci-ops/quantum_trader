#!/bin/bash
# Setup Nginx Reverse Proxy with SSL for quantum.quantumfond.com

set -e

echo "🚀 Setting up Nginx Reverse Proxy for quantum.quantumfond.com"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Variables
DOMAIN="quantum.quantumfond.com"
EMAIL="admin@quantumfond.com"  # Change this to your email
NGINX_CONF="/etc/nginx/sites-available/${DOMAIN}"
NGINX_ENABLED="/etc/nginx/sites-enabled/${DOMAIN}"

# Step 1: Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}❌ Please run as root (use sudo)${NC}"
    exit 1
fi

# Step 2: Install prerequisites
echo -e "\n${YELLOW}📦 Installing prerequisites...${NC}"
apt-get update
apt-get install -y nginx certbot python3-certbot-nginx

# Step 3: Copy nginx configuration
echo -e "\n${YELLOW}📝 Copying nginx configuration...${NC}"
cp /home/qt/quantum_trader/nginx/quantum.quantumfond.com.conf ${NGINX_CONF}

# Step 4: Create symlink if not exists
if [ ! -L ${NGINX_ENABLED} ]; then
    echo -e "${YELLOW}🔗 Creating symlink...${NC}"
    ln -s ${NGINX_CONF} ${NGINX_ENABLED}
fi

# Step 5: Create certbot webroot directory
echo -e "\n${YELLOW}📁 Creating certbot directory...${NC}"
mkdir -p /var/www/certbot

# Step 6: Test nginx configuration
echo -e "\n${YELLOW}🧪 Testing nginx configuration...${NC}"
nginx -t

# Step 7: Reload nginx
echo -e "\n${YELLOW}🔄 Reloading nginx...${NC}"
systemctl reload nginx

# Step 8: Check DNS
echo -e "\n${YELLOW}🌐 Checking DNS for ${DOMAIN}...${NC}"
if host ${DOMAIN} > /dev/null 2>&1; then
    RESOLVED_IP=$(host ${DOMAIN} | grep "has address" | awk '{print $4}')
    SERVER_IP=$(curl -s ifconfig.me)
    echo -e "  Resolved IP: ${RESOLVED_IP}"
    echo -e "  Server IP:   ${SERVER_IP}"
    
    if [ "${RESOLVED_IP}" = "${SERVER_IP}" ]; then
        echo -e "${GREEN}✅ DNS correctly points to this server${NC}"
    else
        echo -e "${RED}⚠️  DNS does not point to this server!${NC}"
        echo -e "${YELLOW}Please update your DNS:${NC}"
        echo -e "  Add A record: ${DOMAIN} → ${SERVER_IP}"
        echo -e "\n${YELLOW}After DNS propagation, run:${NC}"
        echo -e "  sudo certbot --nginx -d ${DOMAIN} --email ${EMAIL} --agree-tos --no-eff-email"
        exit 1
    fi
else
    echo -e "${RED}❌ DNS not configured for ${DOMAIN}${NC}"
    SERVER_IP=$(curl -s ifconfig.me)
    echo -e "${YELLOW}Please add DNS record:${NC}"
    echo -e "  Type: A"
    echo -e "  Name: quantum (or quantum.quantumfond)"
    echo -e "  Value: ${SERVER_IP}"
    exit 1
fi

# Step 9: Get SSL certificate with certbot
echo -e "\n${YELLOW}🔐 Obtaining SSL certificate...${NC}"
certbot --nginx -d ${DOMAIN} --email ${EMAIL} --agree-tos --no-eff-email --redirect

# Step 10: Test SSL renewal
echo -e "\n${YELLOW}🧪 Testing SSL renewal...${NC}"
certbot renew --dry-run

# Step 11: Setup auto-renewal (if not already)
echo -e "\n${YELLOW}⏰ Setting up auto-renewal...${NC}"
if ! crontab -l | grep -q "certbot renew"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * certbot renew --quiet --post-hook 'systemctl reload nginx'") | crontab -
    echo -e "${GREEN}✅ Auto-renewal configured${NC}"
else
    echo -e "${GREEN}✅ Auto-renewal already configured${NC}"
fi

# Step 12: Verify everything
echo -e "\n${YELLOW}🔍 Verifying setup...${NC}"
echo -e "  Nginx status:"
systemctl status nginx --no-pager | head -3
echo -e "\n  SSL certificate:"
certbot certificates | grep -A2 "Certificate Name: ${DOMAIN}" || echo "Certificate check pending..."

# Final summary
echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "\n${GREEN}✅ QuantumFond Dashboard:${NC}"
echo -e "   https://${DOMAIN}"
echo -e "\n${GREEN}✅ RL Intelligence:${NC}"
echo -e "   https://${DOMAIN}/rl-intelligence"
echo -e "\n${GREEN}✅ RL Dashboard API:${NC}"
echo -e "   https://${DOMAIN}/api/rl-dashboard/data"
echo -e "\n${YELLOW}📝 Configuration files:${NC}"
echo -e "   Nginx: ${NGINX_CONF}"
echo -e "   SSL:   /etc/letsencrypt/live/${DOMAIN}/"
echo -e "\n${YELLOW}🔄 Auto-renewal:${NC}"
echo -e "   Certbot will auto-renew at 3 AM daily"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
