#!/bin/bash
# ============================================================================
# NGINX + SSL SETUP SCRIPT (VPS)
# ============================================================================

set -e

echo "=== Quantum Trader - Nginx Production Setup ==="

# Variables
DOMAIN="${1:-quantum-trader.example.com}"
EMAIL="${2:-admin@example.com}"

if [ "$DOMAIN" = "quantum-trader.example.com" ]; then
    echo "ERROR: You must provide a real domain name"
    echo "Usage: ./setup_nginx.sh your-domain.com your-email@example.com"
    exit 1
fi

echo "Domain: $DOMAIN"
echo "Email: $EMAIL"

# 1. Install Nginx and Certbot
echo ""
echo "[1/6] Installing Nginx and Certbot..."
sudo apt update
sudo apt install -y nginx certbot python3-certbot-nginx

# 2. Create certbot webroot
echo ""
echo "[2/6] Creating certbot webroot..."
sudo mkdir -p /var/www/certbot

# 3. Update docker-compose to expose only internal ports
echo ""
echo "[3/6] Updating Docker Compose..."
cd ~/quantum_trader

# Backup current compose file
cp docker-compose.vps.yml docker-compose.vps.yml.backup

# Update ports to localhost only
sed -i 's/- "8001:8001"/- "127.0.0.1:8001:8001"/' docker-compose.vps.yml
sed -i 's/- "3000:3000"/- "127.0.0.1:3000:3000"/' docker-compose.vps.yml
sed -i 's/- "6379:6379"/- "127.0.0.1:6379:6379"/' docker-compose.vps.yml

echo "✅ Ports now bound to localhost only"

# Restart containers with new config
docker compose -f docker-compose.vps.yml up -d

# 4. Configure Nginx
echo ""
echo "[4/6] Configuring Nginx..."

# Copy nginx config and replace domain
sudo cp nginx.conf /etc/nginx/sites-available/quantum-trader
sudo sed -i "s/YOUR_DOMAIN_HERE/$DOMAIN/g" /etc/nginx/sites-available/quantum-trader

# Enable site
sudo ln -sf /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t

# 5. Obtain SSL certificate
echo ""
echo "[5/6] Obtaining SSL certificate..."
sudo systemctl reload nginx

sudo certbot --nginx \
    -d "$DOMAIN" \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    --redirect

# 6. Configure UFW firewall
echo ""
echo "[6/6] Configuring firewall..."
sudo ufw allow 'Nginx Full'
sudo ufw delete allow 8001/tcp 2>/dev/null || true
sudo ufw delete allow 3000/tcp 2>/dev/null || true
sudo ufw delete allow 6379/tcp 2>/dev/null || true

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "✅ Nginx configured and running"
echo "✅ SSL certificate obtained"
echo "✅ Firewall updated"
echo "✅ Services accessible via:"
echo "   - Frontend: https://$DOMAIN"
echo "   - API: https://$DOMAIN/api/"
echo "   - Health: https://$DOMAIN/api/health"
echo ""
echo "⚠️  Direct port access (8001, 3000, 6379) now blocked externally"
echo "✅ All traffic routed through Nginx HTTPS"
