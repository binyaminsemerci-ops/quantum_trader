#!/bin/bash
# Quantum Trader - Complete Deployment Script
# Deploys: Postgres, Nginx, Alertmanager notifications

set -e

echo "=== Quantum Trader Production Deployment ==="
echo "Date: $(date)"
echo ""

# Step 1: Generate SSL certificates (self-signed for testing)
echo "=== STEP 1: Generate SSL Certificates ==="
mkdir -p ~/quantum_trader/nginx/ssl
if [ ! -f ~/quantum_trader/nginx/ssl/cert.pem ]; then
    echo "Generating self-signed SSL certificate..."
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ~/quantum_trader/nginx/ssl/key.pem \
        -out ~/quantum_trader/nginx/ssl/cert.pem \
        -subj "/C=NO/ST=Oslo/L=Oslo/O=QuantumTrader/OU=IT/CN=quantumtrader.local"
    echo "✅ SSL certificates generated"
else
    echo "✅ SSL certificates already exist"
fi

# Step 2: Create backup directories
echo ""
echo "=== STEP 2: Create Backup Directories ==="
mkdir -p ~/quantum_trader/backups/postgres
mkdir -p ~/quantum_trader/nginx/logs
echo "✅ Directories created"

# Step 3: Set up Postgres password
echo ""
echo "=== STEP 3: Configure Postgres Password ==="
if ! grep -q "POSTGRES_PASSWORD" ~/quantum_trader/.env; then
    echo "Adding POSTGRES_PASSWORD to .env..."
    POSTGRES_PASS=$(openssl rand -base64 32)
    echo "" >> ~/quantum_trader/.env
    echo "# PostgreSQL Configuration" >> ~/quantum_trader/.env
    echo "POSTGRES_PASSWORD=$POSTGRES_PASS" >> ~/quantum_trader/.env
    echo "✅ Postgres password generated"
else
    echo "✅ Postgres password already configured"
fi

# Step 4: Deploy services
echo ""
echo "=== STEP 4: Deploy Services ==="
cd ~/quantum_trader
docker compose -f docker-compose.wsl.yml up -d postgres nginx
echo "✅ Postgres and Nginx deployed"

# Step 5: Wait for Postgres to be healthy
echo ""
echo "=== STEP 5: Wait for Postgres ==="
echo "Waiting for Postgres to be healthy..."
for i in {1..30}; do
    if docker exec quantum_postgres pg_isready -U quantum > /dev/null 2>&1; then
        echo "✅ Postgres is healthy"
        break
    fi
    echo -n "."
    sleep 2
done

# Step 6: Set up backup script
echo ""
echo "=== STEP 6: Configure Automated Backups ==="
chmod +x ~/quantum_trader/scripts/backup_postgres.sh
echo "Testing backup script..."
~/quantum_trader/scripts/backup_postgres.sh
echo "✅ Backup script tested successfully"

# Step 7: Add cron job for daily backups
echo ""
echo "=== STEP 7: Configure Cron Job ==="
CRON_JOB="0 2 * * * /home/qt/quantum_trader/scripts/backup_postgres.sh >> /home/qt/quantum_trader/logs/backup.log 2>&1"
(crontab -l 2>/dev/null | grep -v "backup_postgres.sh"; echo "$CRON_JOB") | crontab -
echo "✅ Cron job configured (daily at 2 AM)"

# Step 8: Configure Alertmanager
echo ""
echo "=== STEP 8: Alertmanager Configuration ==="
echo "Current Alertmanager status:"
docker ps | grep alertmanager
echo ""
echo "📝 To enable notifications, update ~/quantum_trader/monitoring/alertmanager.yml with:"
echo "   - Slack: Add SLACK_WEBHOOK_URL to .env"
echo "   - Email: Add SMTP settings to .env"
echo ""
echo "Then restart: docker compose -f docker-compose.alerting.yml restart alertmanager"

# Step 9: Final status check
echo ""
echo "=== STEP 9: Final Status Check ==="
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "quantum_postgres|quantum_nginx|quantum_alertmanager"

echo ""
echo "=== Deployment Summary ==="
echo "✅ Postgres: Running with automated daily backups"
echo "✅ Nginx: Running with SSL (HTTPS on port 443)"
echo "✅ Alertmanager: Running (configure notifications in .env)"
echo ""
echo "📊 Access Points:"
echo "  - HTTPS Health: https://$(hostname -I | awk '{print $1}')/health"
echo "  - HTTP → HTTPS redirect: http://$(hostname -I | awk '{print $1}')/"
echo "  - Postgres: localhost:5432 (internal only)"
echo ""
echo "⚠️  Next Steps:"
echo "  1. Configure Alertmanager notifications (see Step 8)"
echo "  2. For production SSL: Install certbot and get Let's Encrypt certificate"
echo "  3. Test backup restore: cat backup.sql.gz | gunzip | docker exec -i quantum_postgres psql -U quantum quantum_trader"
echo ""
echo "=== Deployment Complete ==="
