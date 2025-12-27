# QuantumFond Deployment Guide

## 🚀 Deployment Options

### Option 1: Docker (Recommended)
Easiest deployment using Docker Compose.

### Option 2: Traditional
Manual deployment on Linux/Ubuntu server.

### Option 3: Cloud Platforms
Deploy to AWS, Azure, or Google Cloud.

---

## 🐳 Docker Deployment (Recommended)

### Prerequisites
- Docker & Docker Compose installed
- Domain names configured (api.quantumfond.com, app.quantumfond.com)
- Ports 80, 443, 5432, 6379, 8000 available

### Quick Start

```bash
# 1. Configure environment
cp .env.quantumfond .env
nano .env  # Edit with production settings

# 2. Deploy with Docker Compose
docker-compose -f docker-compose.quantumfond.yml up -d

# 3. Check status
docker-compose -f docker-compose.quantumfond.yml ps

# 4. View logs
docker-compose -f docker-compose.quantumfond.yml logs -f
```

### What Gets Deployed
- ✅ PostgreSQL database (port 5432)
- ✅ Redis cache (port 6379)
- ✅ Backend API (port 8000)
- ✅ Frontend app (port 80)

### Verify Deployment

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost/health

# API docs
curl http://localhost:8000/docs
```

---

## 🖥️ Traditional Deployment (Linux/Ubuntu)

### Prerequisites
- Ubuntu 20.04+ or similar Linux distribution
- Python 3.11+
- Node.js 20+
- PostgreSQL 15+
- Nginx
- Domain names with SSL certificates

### 1. Prepare Server

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv nodejs npm postgresql nginx certbot python3-certbot-nginx

# Create user
sudo useradd -m -s /bin/bash quantumfond
sudo mkdir -p /opt/quantumfond
sudo chown quantumfond:quantumfond /opt/quantumfond
```

### 2. Setup Database

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database and user
CREATE DATABASE quantumdb;
CREATE USER quantumfond WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE quantumdb TO quantumfond;
\q
```

### 3. Deploy Backend

```bash
# Copy backend files
sudo cp -r quantumfond_backend /opt/quantumfond/backend
cd /opt/quantumfond/backend

# Create virtual environment
sudo -u quantumfond python3 -m venv venv

# Install dependencies
sudo -u quantumfond venv/bin/pip install -r requirements.txt

# Configure environment
sudo cp .env.example .env
sudo nano .env  # Edit with production settings

# Initialize database
sudo -u quantumfond venv/bin/python -c "from db.connection import init_db; init_db()"

# Install systemd service
sudo cp quantumfond-backend.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable quantumfond-backend
sudo systemctl start quantumfond-backend

# Check status
sudo systemctl status quantumfond-backend
```

### 4. Deploy Frontend

```bash
# Build frontend
cd quantumfond_frontend
npm install
npm run build

# Copy to web directory
sudo mkdir -p /var/www/quantumfond/frontend
sudo cp -r dist/* /var/www/quantumfond/frontend/
sudo chown -R www-data:www-data /var/www/quantumfond
```

### 5. Configure Nginx

```bash
# Copy nginx configuration
sudo cp nginx-quantumfond.conf /etc/nginx/sites-available/quantumfond

# Enable site
sudo ln -s /etc/nginx/sites-available/quantumfond /etc/nginx/sites-enabled/

# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx
```

### 6. Setup SSL

```bash
# Obtain SSL certificates
sudo certbot --nginx -d api.quantumfond.com
sudo certbot --nginx -d app.quantumfond.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### 7. Automated Deployment Script

```bash
# Make script executable
chmod +x deploy-quantumfond.sh

# Run deployment
./deploy-quantumfond.sh production
```

---

## ☁️ Cloud Platform Deployment

### AWS Deployment

**Architecture:**
- EC2 instance (t3.medium or larger)
- RDS PostgreSQL
- ElastiCache Redis
- Application Load Balancer
- Route 53 for DNS
- Certificate Manager for SSL

**Steps:**
1. Launch EC2 instance with Ubuntu
2. Create RDS PostgreSQL instance
3. Create ElastiCache Redis cluster
4. Configure security groups
5. Follow traditional deployment steps
6. Configure ALB with target groups
7. Setup Route 53 DNS records

### Azure Deployment

**Architecture:**
- Azure App Service (Backend)
- Azure Static Web Apps (Frontend)
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Azure Front Door
- Azure DNS

### Google Cloud Deployment

**Architecture:**
- Google Compute Engine or Cloud Run
- Cloud SQL (PostgreSQL)
- Memorystore (Redis)
- Cloud Load Balancing
- Cloud DNS
- Cloud CDN

---

## 🔒 Security Configuration

### Environment Variables

**Required Variables:**
```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/quantumdb

# JWT
JWT_SECRET_KEY=your-super-secret-key-at-least-32-chars

# Environment
ENVIRONMENT=production

# CORS
CORS_ORIGINS=https://app.quantumfond.com
```

### SSL/TLS

**Generate strong SSL:**
```bash
sudo certbot certonly --nginx \
  -d api.quantumfond.com \
  -d app.quantumfond.com \
  --email admin@quantumfond.com \
  --agree-tos
```

### Firewall Rules

```bash
# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow SSH (if needed)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

---

## 📊 Monitoring & Logging

### Application Logs

```bash
# Backend logs
journalctl -u quantumfond-backend -f

# Nginx access logs
tail -f /var/log/nginx/quantumfond-app-access.log

# Nginx error logs
tail -f /var/log/nginx/quantumfond-app-error.log
```

### Health Checks

```bash
# Backend health
curl https://api.quantumfond.com/health

# Frontend health
curl https://app.quantumfond.com/health

# Database connection
sudo -u postgres psql -c "SELECT version();"
```

### Docker Monitoring

```bash
# Container status
docker-compose -f docker-compose.quantumfond.yml ps

# Container logs
docker-compose -f docker-compose.quantumfond.yml logs backend
docker-compose -f docker-compose.quantumfond.yml logs frontend

# Resource usage
docker stats
```

---

## 🔄 Updates & Maintenance

### Update Backend

```bash
# Pull latest code
git pull origin main

# Restart service
sudo systemctl restart quantumfond-backend

# Or with Docker
docker-compose -f docker-compose.quantumfond.yml build backend
docker-compose -f docker-compose.quantumfond.yml up -d backend
```

### Update Frontend

```bash
# Build new version
cd quantumfond_frontend
npm run build

# Deploy
sudo cp -r dist/* /var/www/quantumfond/frontend/

# Or with Docker
docker-compose -f docker-compose.quantumfond.yml build frontend
docker-compose -f docker-compose.quantumfond.yml up -d frontend
```

### Database Backup

```bash
# Manual backup
pg_dump -U quantumfond quantumdb > backup_$(date +%Y%m%d).sql

# Automated backup (add to crontab)
0 2 * * * pg_dump -U quantumfond quantumdb > /backups/quantumdb_$(date +\%Y\%m\%d).sql
```

---

## 🐛 Troubleshooting

### Backend Not Starting

```bash
# Check logs
journalctl -u quantumfond-backend -n 50

# Check if port is in use
sudo lsof -i :8000

# Test database connection
psql -U quantumfond -d quantumdb -h localhost
```

### Frontend 404 Errors

```bash
# Check nginx configuration
sudo nginx -t

# Verify file permissions
ls -la /var/www/quantumfond/frontend/

# Check nginx error log
tail -f /var/log/nginx/error.log
```

### CORS Issues

1. Check backend CORS settings in `.env`
2. Verify nginx proxy headers
3. Check browser console for specific CORS errors
4. Ensure credentials are allowed

### Database Connection Issues

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connection
psql -U quantumfond -h localhost -d quantumdb

# Verify credentials in .env
cat /opt/quantumfond/backend/.env | grep DATABASE_URL
```

---

## 📈 Performance Optimization

### Backend Optimization

1. **Increase workers:**
   ```bash
   uvicorn main:app --workers 4
   ```

2. **Enable connection pooling** (already configured in code)

3. **Add Redis caching:**
   - Install Redis
   - Configure caching in backend

### Frontend Optimization

1. **Enable CDN** for static assets
2. **Enable Brotli compression** in nginx
3. **Optimize images** before deployment
4. **Enable HTTP/2** in nginx

### Database Optimization

1. **Add indexes** on frequently queried fields
2. **Configure connection pooling**
3. **Regular VACUUM** operations
4. **Monitor slow queries**

---

## ✅ Post-Deployment Checklist

- [ ] Backend health check returns 200
- [ ] Frontend loads correctly
- [ ] API documentation accessible
- [ ] Authentication working
- [ ] Database connections active
- [ ] SSL certificates valid
- [ ] Logs are being collected
- [ ] Backups configured
- [ ] Monitoring setup
- [ ] Firewall configured
- [ ] DNS records correct
- [ ] Test all major features

---

## 🎯 Production URLs

**After deployment, your services will be available at:**

- **Frontend:** https://app.quantumfond.com
- **Backend API:** https://api.quantumfond.com
- **API Documentation:** https://api.quantumfond.com/docs
- **Health Checks:**
  - Backend: https://api.quantumfond.com/health
  - Frontend: https://app.quantumfond.com/health

---

## 📞 Support

**Documentation:**
- Setup Guide: `QUANTUMFOND_SETUP_GUIDE.md`
- Architecture: `QUANTUMFOND_ARCHITECTURE.md`
- Quick Start: `QUANTUMFOND_QUICKSTART.md`

**Logs Location:**
- Backend: `/var/log/quantumfond/` or `journalctl -u quantumfond-backend`
- Nginx: `/var/log/nginx/quantumfond-*.log`
- Database: `/var/log/postgresql/`

---

>>> **Ready for production deployment on quantumfond.com** <<<
