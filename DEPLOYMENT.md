# Quantum Trader Deployment Guide

This guide provides step-by-step instructions for deploying Quantum Trader in various environments, from local development to production clusters.

## Table of Contents

1. [Quick Start (Docker Compose)](#quick-start-docker-compose)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [Environment Configuration](#environment-configuration)
6. [Troubleshooting](#troubleshooting)

## Quick Start (Docker Compose)

**Prerequisites**: Docker and Docker Compose installed

1. **Clone the repository**:
   ```bash
   git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
   cd quantum_trader
   ```

2. **Start all services**:
   ```bash
   docker-compose up --build
   ```

3. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

4. **Initialize database** (first time only):
   ```bash
   docker-compose exec backend alembic upgrade head
   docker-compose exec backend python scripts/seed_demo_data.py
   ```

## Local Development Setup

### Backend Development

1. **Set up Python environment**:
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development tools
   ```

3. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up database**:
   ```bash
   # For SQLite (development)
   alembic upgrade head
   python scripts/seed_demo_data.py

   # For PostgreSQL (production-like)
   export QUANTUM_TRADER_DATABASE_URL="postgresql://user:pass@localhost/quantum_trader"  # pragma: allowlist secret
   alembic upgrade head
   python scripts/seed_demo_data.py
   ```

5. **Run development server**:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Development

1. **Set up Node.js environment**:
   ```bash
   cd frontend
   npm install
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Run development server**:
   ```bash
   npm run dev
   ```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/ -v

# Frontend tests
cd frontend
npm run test
```

## Production Deployment

### Option 1: Docker Swarm

1. **Prepare production configuration**:
   ```bash
   cp docker-compose.yml docker-compose.prod.yml
   # Edit docker-compose.prod.yml for production settings
   ```

2. **Initialize swarm**:
   ```bash
   docker swarm init
   ```

3. **Deploy stack**:
   ```bash
   docker stack deploy -c docker-compose.prod.yml quantum-trader
   ```

### Option 2: Kubernetes

1. **Create namespace**:
   ```bash
   kubectl create namespace quantum-trader
   ```

2. **Apply configurations**:
   ```bash
   kubectl apply -f k8s/ -n quantum-trader
   ```

3. **Monitor deployment**:
   ```bash
   kubectl get pods -n quantum-trader
   kubectl logs -f deployment/quantum-trader-backend -n quantum-trader
   ```

### Option 3: Traditional Server Setup

1. **Set up reverse proxy** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location /api/ {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location / {
           proxy_pass http://localhost:3000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Set up process management** (systemd example):
   ```ini
   [Unit]
   Description=Quantum Trader Backend
   After=network.target

   [Service]
   Type=simple
   User=quantum-trader
   WorkingDirectory=/opt/quantum-trader/backend
   Environment=PATH=/opt/quantum-trader/backend/.venv/bin
   ExecStart=/opt/quantum-trader/backend/.venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

## Cloud Deployment Options

### AWS (Amazon Web Services)

**Option A: ECS with Fargate**
- Container-based deployment with managed infrastructure
- Auto-scaling and load balancing
- Integrated with RDS for PostgreSQL

**Option B: Elastic Beanstalk**
- Platform-as-a-Service approach
- Automatic capacity provisioning and load balancing
- Easy deployment via git or CLI

**Option C: EC2 with Auto Scaling**
- Full control over infrastructure
- Custom AMIs with pre-configured environment
- CloudWatch integration for monitoring

### Google Cloud Platform

**Option A: Cloud Run**
- Serverless container platform
- Auto-scaling to zero
- Pay-per-request pricing

**Option B: Google Kubernetes Engine (GKE)**
- Managed Kubernetes service
- Integrated with Cloud SQL for PostgreSQL
- Stackdriver for logging and monitoring

### Microsoft Azure

**Option A: Container Instances**
- Simple container deployment
- Per-second billing
- Integrated with Azure Database for PostgreSQL

**Option B: App Service**
- Platform-as-a-Service for web applications
- Built-in CI/CD integration
- Auto-scaling capabilities

### Digital Ocean

**Option A: App Platform**
- Simple git-based deployment
- Managed databases available
- Automatic HTTPS and CDN

**Option B: Kubernetes**
- Managed Kubernetes service
- Integrated load balancers
- Block and object storage options

## Environment Configuration

### Required Environment Variables

```bash
# Database Configuration
QUANTUM_TRADER_DATABASE_URL=postgresql://user:pass@host:5432/quantum_trader  # pragma: allowlist secret

# API Keys (for live trading)
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# External Data Sources
CRYPTOPANIC_API_KEY=your_cryptopanic_key
TWITTER_BEARER_TOKEN=your_twitter_token

# Application Settings
LOG_LEVEL=INFO
ENVIRONMENT=production
SECRET_KEY=your-secret-key-for-sessions

# Performance Monitoring
ENABLE_METRICS=true
METRICS_RETENTION_HOURS=24
```

### Optional Configuration

```bash
# Redis for Caching (optional)
REDIS_URL=redis://localhost:6379/0

# SMTP for Notifications (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=notifications@yourdomain.com
SMTP_PASSWORD=your_app_password

# Monitoring & Alerting
SENTRY_DSN=your_sentry_dsn
DATADOG_API_KEY=your_datadog_key
```

## Monitoring and Maintenance

### Health Checks

```bash
# Application health
curl http://localhost:8000/api/health

# Database connectivity
curl http://localhost:8000/api/health/db

# Performance metrics
curl http://localhost:8000/api/metrics/system
```

### Log Management

```bash
# View application logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Database logs
docker-compose logs -f postgres

# Performance metrics logs
grep "performance" logs/*.log
```

### Backup Strategy

```bash
# Database backup
pg_dump $QUANTUM_TRADER_DATABASE_URL > backup_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf config_backup.tar.gz .env docker-compose.yml k8s/

# Application data backup
rsync -av --exclude='.git' --exclude='node_modules' . backup/quantum_trader/
```

## Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check database status
docker-compose ps postgres

# Test connection
psql $QUANTUM_TRADER_DATABASE_URL -c "SELECT 1"

# Check migrations
alembic current
alembic history
```

**Frontend Build Issues**
```bash
# Clear node modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check for TypeScript errors
npm run typecheck

# Build production bundle
npm run build
```

**API Performance Issues**
```bash
# Check metrics endpoints
curl http://localhost:8000/api/metrics/requests
curl http://localhost:8000/api/metrics/database

# Monitor resource usage
docker stats

# Check logs for slow queries
grep "duration_ms.*[5-9][0-9][0-9]" logs/performance.log
```

### Performance Tuning

**Database Optimization**
- Enable connection pooling
- Add database indexes for frequently queried columns
- Configure PostgreSQL shared_buffers and work_mem
- Monitor slow query log

**Application Optimization**
- Enable Redis caching for API responses
- Implement request rate limiting
- Use CDN for static assets
- Configure appropriate worker counts for uvicorn

**Frontend Optimization**
- Enable code splitting and lazy loading
- Compress and cache static assets
- Implement service worker for offline functionality
- Optimize bundle size with tree shaking

## Security Considerations

### Production Checklist

- [ ] Use HTTPS everywhere (TLS 1.2+)
- [ ] Implement API rate limiting
- [ ] Secure environment variables (never commit to git)
- [ ] Enable database connection encryption
- [ ] Regular security updates for all dependencies
- [ ] Implement proper CORS policies
- [ ] Use strong, unique passwords for all services
- [ ] Enable audit logging for all API access
- [ ] Implement proper session management
- [ ] Regular backup testing and recovery procedures

### Monitoring Setup

```bash
# Set up alerts for key metrics
- Response time > 1 second
- Error rate > 5%
- Database connection failures
- High memory usage (>80%)
- Disk space usage (>90%)
```

For additional support, visit our [GitHub repository](https://github.com/binyaminsemerci-ops/quantum_trader) or check the [API documentation](http://localhost:8000/api/docs).
