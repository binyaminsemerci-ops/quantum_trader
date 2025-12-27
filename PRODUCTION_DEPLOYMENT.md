# 🚀 Production Deployment - Quick Start Guide

## Prerequisites
- VPS with Ubuntu 20.04+ (minimum 4GB RAM, 20GB disk)
- Domain name pointing to VPS IP
- Docker and Docker Compose installed
- SSH access to VPS

## 1. Initial VPS Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx (for reverse proxy and SSL)
sudo apt install nginx certbot python3-certbot-nginx -y
```

## 2. Clone and Configure

```bash
# Clone repository
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
cd quantum_trader

# Create environment file
cp .env.example .env
nano .env  # Edit with your settings
```

### Required Environment Variables

```env
# API Keys (CRITICAL - Keep Secret!)
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_secret

# Trading Configuration
QT_ALLOWED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT  # Add your symbols
QT_EXECUTION_MODE=live  # or 'testnet' for testing
QT_DRY_RUN=false  # Set to true for paper trading

# Risk Management
QT_MAX_NOTIONAL=100  # Maximum position size in USD
QT_MAX_DAILY_LOSS=-50  # Maximum daily loss in USD

# Database
DATABASE_URL=sqlite:///./backend/trades.db

# Scheduler
QUANTUM_TRADER_SCHEDULER_ENABLED=true
QUANTUM_TRADER_REFRESH_SECONDS=180
QUANTUM_TRADER_LIQUIDITY_SECONDS=900
QUANTUM_TRADER_EXECUTION_SECONDS=300
```

## 3. SSL Certificate Setup

```bash
# Replace your-domain.com with your actual domain
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

## 4. Nginx Configuration

```bash
sudo nano /etc/nginx/sites-available/quantum-trader
```

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://localhost:8000/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # Health check
    location /health {
        proxy_pass http://localhost:8000/health;
    }
}
```

```bash
# Enable site and restart Nginx
sudo ln -s /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 5. Build and Deploy

```bash
# Build Docker images
docker-compose build --no-cache

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f backend
```

## 6. Monitoring Setup

### Create systemd service for monitoring

```bash
sudo nano /etc/systemd/system/quantum-monitor.service
```

```ini
[Unit]
Description=Quantum Trader Monitoring
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/docker-compose -f /home/ubuntu/quantum_trader/docker-compose.yml ps
WorkingDirectory=/home/ubuntu/quantum_trader

[Install]
WantedBy=multi-user.target
```

### Create monitoring script

```bash
nano ~/monitor-quantum.sh
```

```bash
#!/bin/bash

LOG_FILE="/var/log/quantum-trader/monitor.log"
mkdir -p /var/log/quantum-trader

check_health() {
    echo "[$(date)] Checking health..." >> $LOG_FILE
    
    # Check backend health
    HEALTH=$(curl -s http://localhost:8000/health)
    echo "Backend: $HEALTH" >> $LOG_FILE
    
    # Check if containers are running
    CONTAINERS=$(docker-compose ps --services --filter "status=running")
    echo "Running containers: $CONTAINERS" >> $LOG_FILE
    
    # Check disk space
    DISK=$(df -h / | tail -1 | awk '{print $5}')
    echo "Disk usage: $DISK" >> $LOG_FILE
    
    # Check memory
    MEM=$(free -h | grep Mem | awk '{print $3 "/" $2}')
    echo "Memory usage: $MEM" >> $LOG_FILE
    
    echo "---" >> $LOG_FILE
}

check_health

# Send alert if health check fails
if [ $? -ne 0 ]; then
    # Add your alerting logic here (email, Slack, Discord, etc.)
    echo "ALERT: Health check failed!" >> $LOG_FILE
fi
```

```bash
chmod +x ~/monitor-quantum.sh

# Add to crontab (run every 5 minutes)
crontab -e
```

Add line:
```
*/5 * * * * /home/ubuntu/monitor-quantum.sh
```

## 7. Backup Strategy

```bash
# Create backup script
nano ~/backup-quantum.sh
```

```bash
#!/bin/bash

BACKUP_DIR="/home/ubuntu/backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup database
cp /home/ubuntu/quantum_trader/backend/trades.db $BACKUP_DIR/trades_$DATE.db

# Backup environment
cp /home/ubuntu/quantum_trader/.env $BACKUP_DIR/env_$DATE.bak

# Backup models
cp -r /home/ubuntu/quantum_trader/ai_engine/models $BACKUP_DIR/models_$DATE

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.bak" -mtime +7 -delete

echo "Backup completed: $DATE"
```

```bash
chmod +x ~/backup-quantum.sh

# Add to crontab (daily at 2 AM)
crontab -e
```

Add line:
```
0 2 * * * /home/ubuntu/backup-quantum.sh
```

## 8. Security Checklist

- [ ] Change default passwords
- [ ] Enable firewall: `sudo ufw enable`
- [ ] Allow only necessary ports:
  ```bash
  sudo ufw allow 22/tcp    # SSH
  sudo ufw allow 80/tcp    # HTTP
  sudo ufw allow 443/tcp   # HTTPS
  ```
- [ ] Disable root login
- [ ] Set up fail2ban: `sudo apt install fail2ban -y`
- [ ] Regular security updates: `sudo apt update && sudo apt upgrade -y`
- [ ] Store API keys in secure vault (not in code)
- [ ] Enable 2FA on exchange account
- [ ] Set up IP whitelist on exchange
- [ ] Monitor unusual activity

## 9. Post-Deployment Checks

```bash
# Check backend health
curl https://your-domain.com/health

# Check API
curl https://your-domain.com/api/metrics

# Check WebSocket
wscat -c wss://your-domain.com/ws/dashboard

# View logs
docker-compose logs -f backend

# Check scheduler
curl https://your-domain.com/health/scheduler
```

## 10. Maintenance Commands

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart backend

# View logs
docker-compose logs -f

# Update code
git pull origin main
docker-compose build --no-cache
docker-compose up -d

# Stop all services
docker-compose down

# Clean up old images
docker system prune -a
```

## 11. Troubleshooting

### Backend not starting
```bash
# Check logs
docker-compose logs backend

# Check environment
docker-compose exec backend env

# Rebuild
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Database issues
```bash
# Check database
docker-compose exec backend python -c "from database import get_db; next(get_db())"

# Reset database (CAUTION: deletes data)
rm backend/trades.db
docker-compose restart backend
```

### Memory issues
```bash
# Check memory
free -h

# Restart services
docker-compose restart

# Reduce refresh intervals in .env
```

## 12. Performance Tuning

```env
# In .env file

# Reduce refresh frequency for lower resource usage
QUANTUM_TRADER_REFRESH_SECONDS=300  # 5 minutes instead of 3

# Limit symbols
QT_ALLOWED_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT  # Focus on top coins

# Adjust worker threads
UVICORN_WORKERS=2  # Match CPU cores
```

## Support

- 📧 Issues: https://github.com/binyaminsemerci-ops/quantum_trader/issues
- 📖 Docs: See `/docs` folder
- 💬 Community: [Add your Discord/Slack]

---

**⚠️ IMPORTANT:** Always test in paper trading mode (`QT_DRY_RUN=true`) before going live!
