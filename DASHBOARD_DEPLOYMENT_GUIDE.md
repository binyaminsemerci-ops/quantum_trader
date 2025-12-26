# 🚀 QUANTUM TRADER DASHBOARD - DEPLOYMENT GUIDE

## 📋 Deployment Steps (VPS)

### Option 1: Manual Deployment via SSH

```bash
# 1. SSH into VPS
ssh root@46.224.116.254

# 2. Navigate to project
cd /root/quantum_trader

# 3. Check if dashboard_v4 exists (should be synced from local)
ls -la dashboard_v4/

# 4. Stop existing dashboard (if running)
docker compose --profile dashboard down

# 5. Build dashboard containers
docker compose --profile dashboard build dashboard-backend
docker compose --profile dashboard build dashboard-frontend

# 6. Start dashboard
docker compose --profile dashboard up -d dashboard-backend dashboard-frontend

# 7. Check status
docker compose --profile dashboard ps

# 8. View logs
docker compose --profile dashboard logs -f dashboard-backend
docker compose --profile dashboard logs -f dashboard-frontend

# 9. Test endpoints
curl http://localhost:8025/health
curl http://localhost:8025/api/ai/status
curl -I http://localhost:8888
```

### Option 2: Using Deployment Script

```bash
# On VPS:
cd /root/quantum_trader
bash deploy_dashboard_vps.sh
```

## 🌐 Access URLs

Once deployed:
- **Backend API**: http://46.224.116.254:8025
- **Frontend**: http://46.224.116.254:8888
- **API Docs**: http://46.224.116.254:8025/docs
- **Domain** (when DNS works): http://quantumtrader.com:8888

## 🔧 Port Configuration

Make sure these ports are open in Hetzner Cloud Firewall:
- **8025** - Dashboard Backend API
- **8888** - Dashboard Frontend

## 📦 What Gets Deployed

### Backend (dashboard-backend)
- FastAPI application
- Connected to PostgreSQL (existing quantum_trader DB)
- Connected to Redis (existing quantum_trader Redis)
- Endpoints:
  - `/health` - Health check
  - `/version` - API version
  - `/ai/status` - AI model metrics
  - `/portfolio/status` - Portfolio PnL
  - `/risk/metrics` - Risk analytics
  - `/system/health` - System resources

### Frontend (dashboard-frontend)
- React + TypeScript + Vite
- Nginx serving static files
- API calls proxied to backend via `/api` prefix
- Auto-refreshes data every 5 seconds

## 🐛 Troubleshooting

### Backend not starting
```bash
# Check logs
docker logs quantum_dashboard_backend

# Check PostgreSQL connection
docker exec quantum_dashboard_backend python -c "from db.connection import engine; engine.connect()"
```

### Frontend not accessible
```bash
# Check if nginx is running
docker exec quantum_dashboard_frontend nginx -t

# Check logs
docker logs quantum_dashboard_frontend
```

### Port conflicts
```bash
# Check what's using port 8888
netstat -tulpn | grep 8888

# Stop conflicting container
docker stop <container_name>
```

## 🔄 Updating Dashboard

```bash
# On local machine, sync changes
scp -r dashboard_v4/ root@46.224.116.254:/root/quantum_trader/
scp docker-compose.yml root@46.224.116.254:/root/quantum_trader/

# On VPS, rebuild and restart
ssh root@46.224.116.254
cd /root/quantum_trader
docker compose --profile dashboard down
docker compose --profile dashboard build
docker compose --profile dashboard up -d
```

## 📊 Monitoring

```bash
# Check container health
docker compose --profile dashboard ps

# View real-time logs
docker compose --profile dashboard logs -f

# Check resource usage
docker stats quantum_dashboard_backend quantum_dashboard_frontend
```

## 🔐 Security Notes

- Backend uses environment variables for DB credentials
- Frontend served over HTTP (add HTTPS/SSL later)
- API endpoints have CORS enabled for development
- Production: Add authentication, rate limiting, SSL
