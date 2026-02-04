#!/bin/bash
# Quantum Trader - Let's Encrypt SSL Setup Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NGINX_SSL_DIR="${PROJECT_ROOT}/nginx/ssl"

echo "================================================"
echo "   Quantum Trader Let's Encrypt SSL Setup"
echo "================================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script must be run as root (for certbot)"
    echo "   Run with: sudo bash $0"
    exit 1
fi

# Prompt for domain
echo "⚠️  Prerequisites:"
echo "   1. Domain name must point to this server's IP"
echo "   2. DNS A record must be configured"
echo "   3. Ports 80 and 443 must be accessible"
echo ""
read -p "Enter your domain name (e.g., trade.example.com): " domain

if [ -z "$domain" ]; then
    echo "❌ Domain name required"
    exit 1
fi

# Verify DNS
echo ""
echo "=== Verifying DNS ==="
server_ip=$(curl -s ifconfig.me)
domain_ip=$(dig +short "$domain" | head -1)

echo "Server IP:  $server_ip"
echo "Domain IP:  $domain_ip"

if [ "$server_ip" != "$domain_ip" ]; then
    echo ""
    echo "⚠️  WARNING: Domain DNS does not point to this server!"
    read -p "Continue anyway? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        echo "Setup cancelled"
        exit 1
    fi
fi

# Install certbot if not present
echo ""
echo "=== Installing Certbot ==="
if ! command -v certbot &> /dev/null; then
    apt-get update
    apt-get install -y certbot python3-certbot-nginx
    echo "✅ Certbot installed"
else
    echo "✅ Certbot already installed"
fi

# Stop Nginx container to free port 80
echo ""
echo "=== Stopping Nginx Container ==="
cd "$PROJECT_ROOT"
docker stop quantum_nginx || true
echo "✅ Nginx stopped"

# Get certificate
echo ""
echo "=== Obtaining SSL Certificate ==="
certbot certonly --standalone \
    -d "$domain" \
    --non-interactive \
    --agree-tos \
    --email "admin@${domain}" \
    --no-eff-email

if [ $? -eq 0 ]; then
    echo "✅ SSL certificate obtained"
else
    echo "❌ Failed to obtain certificate"
    docker start quantum_nginx
    exit 1
fi

# Update Nginx configuration
echo ""
echo "=== Updating Nginx Configuration ==="
NGINX_CONF="${PROJECT_ROOT}/nginx/nginx.conf"
NGINX_CONF_BACKUP="${NGINX_CONF}.backup-$(date +%Y%m%d_%H%M%S)"

# Backup current config
cp "$NGINX_CONF" "$NGINX_CONF_BACKUP"
echo "✅ Backup created: $NGINX_CONF_BACKUP"

# Replace SSL certificate paths
sed -i "s|ssl_certificate /etc/nginx/ssl/nginx.crt;|ssl_certificate /etc/letsencrypt/live/${domain}/fullchain.pem;|" "$NGINX_CONF"
sed -i "s|ssl_certificate_key /etc/nginx/ssl/nginx.key;|ssl_certificate_key /etc/letsencrypt/live/${domain}/privkey.pem;|" "$NGINX_CONF"

echo "✅ Nginx configuration updated"

# Update docker-compose to mount Let's Encrypt certs
echo ""
echo "=== Updating Docker Compose ==="
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.wsl.yml"
COMPOSE_BACKUP="${COMPOSE_FILE}.backup-$(date +%Y%m%d_%H%M%S)"

cp "$COMPOSE_FILE" "$COMPOSE_BACKUP"
echo "✅ Backup created: $COMPOSE_BACKUP"

# Check if letsencrypt volume already exists
if ! grep -q "/etc/letsencrypt:/etc/letsencrypt:ro" "$COMPOSE_FILE"; then
    # Add letsencrypt volume to nginx service
    sed -i "/nginx:$/,/^  [a-z]/ {
        /volumes:/a\        - /etc/letsencrypt:/etc/letsencrypt:ro
    }" "$COMPOSE_FILE"
    echo "✅ Let's Encrypt volume added to docker-compose"
else
    echo "✅ Let's Encrypt volume already configured"
fi

# Restart Nginx with new configuration
echo ""
echo "=== Restarting Nginx ==="
cd "$PROJECT_ROOT"
docker compose -f docker-compose.wsl.yml up -d nginx
sleep 3

# Test HTTPS
echo ""
echo "=== Testing HTTPS ==="
if curl -sk "https://${domain}/health" > /dev/null 2>&1; then
    echo "✅ HTTPS working!"
else
    echo "⚠️  HTTPS test failed, check logs:"
    echo "   docker logs quantum_nginx"
fi

# Setup auto-renewal
echo ""
echo "=== Setting Up Auto-Renewal ==="
RENEWAL_SCRIPT="${SCRIPT_DIR}/renew_ssl.sh"

cat > "$RENEWAL_SCRIPT" << 'EOF'
#!/bin/bash
# Auto-renewal script for Let's Encrypt certificates

certbot renew --quiet --pre-hook "docker stop quantum_nginx" --post-hook "docker start quantum_nginx"

if [ $? -eq 0 ]; then
    echo "$(date): SSL certificates renewed successfully" >> /var/log/letsencrypt-renewal.log
else
    echo "$(date): SSL renewal failed" >> /var/log/letsencrypt-renewal.log
fi
EOF

chmod +x "$RENEWAL_SCRIPT"

# Add to crontab (run daily at 3 AM)
if ! crontab -l 2>/dev/null | grep -q "renew_ssl.sh"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * ${RENEWAL_SCRIPT}") | crontab -
    echo "✅ Auto-renewal cron job added (daily at 3 AM)"
else
    echo "✅ Auto-renewal cron job already configured"
fi

echo ""
echo "================================================"
echo "   ✅ Let's Encrypt SSL Setup Complete!"
echo "================================================"
echo ""
echo "Your site is now accessible at:"
echo "  https://${domain}"
echo ""
echo "Certificate location:"
echo "  /etc/letsencrypt/live/${domain}/fullchain.pem"
echo "  /etc/letsencrypt/live/${domain}/privkey.pem"
echo ""
echo "Auto-renewal:"
echo "  Runs daily at 3 AM"
echo "  Certificates valid for 90 days"
echo ""
echo "Test renewal:"
echo "  certbot renew --dry-run"
echo ""
