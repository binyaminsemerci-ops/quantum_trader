# Quantum Trader Dashboard - Production Deployment Guide

## Current Status

✅ **Completed**:
- Backend and frontend Docker containers running
- Services accessible via IP: http://46.224.116.254:8025 (backend) and :8889 (frontend)
- Production configurations created (.env files, CORS, nginx configs)
- Frontend built (149.43 kB production bundle)
- Root domain DNS configured: quantumfond.com → 46.224.116.254

⏳ **Required Actions**:
1. Add subdomain DNS records (see below)
2. Wait for DNS propagation (5-60 minutes)
3. Run automated deployment script

---

## Step 1: Add Missing DNS Records

**In your DNS management panel, add these two A records:**

```
Type    Name    Content          TTL
A       api     46.224.116.254   600
A       app     46.224.116.254   600
```

This creates:
- `api.quantumfond.com` → 46.224.116.254 (backend)
- `app.quantumfond.com` → 46.224.116.254 (frontend)

---

## Step 2: Verify DNS Propagation

**Wait 5-60 minutes**, then verify:

```powershell
# Should return 46.224.116.254
nslookup api.quantumfond.com 8.8.8.8
nslookup app.quantumfond.com 8.8.8.8
```

Or use online tools:
- https://dnschecker.org/#A/api.quantumfond.com
- https://dnschecker.org/#A/app.quantumfond.com

---

## Step 3: Run Automated Deployment

Once DNS propagates, pull the latest code and run:

```bash
# On VPS
ssh root@46.224.116.254
cd ~/quantum_trader
git pull origin main
chmod +x deploy_production.sh
./deploy_production.sh
```

The script will:
1. ✓ Verify DNS configuration
2. ✓ Stop conflicting nginx container
3. ✓ Deploy HTTP nginx configuration
4. ✓ Test HTTP access
5. ✓ Obtain SSL certificates from Let's Encrypt
6. ✓ Deploy HTTPS nginx configuration
7. ✓ Verify HTTPS access
8. ✓ Setup SSL auto-renewal

---

## Step 4: Access Your Dashboard

After deployment completes:

**Production URLs:**
- Frontend: https://app.quantumfond.com
- Backend: https://api.quantumfond.com
- WebSocket: wss://api.quantumfond.com/stream/live

**Test Endpoints:**
```bash
# Health check
curl https://api.quantumfond.com/health

# AI insights
curl https://api.quantumfond.com/ai/insights

# Portfolio status
curl https://api.quantumfond.com/portfolio

# Real-time metrics
curl https://api.quantumfond.com/metrics/realtime
```

---

## Troubleshooting

### DNS Not Resolving

**Issue**: `nslookup` times out or returns wrong IP

**Solution**:
1. Verify A records in DNS management panel
2. Check DNS propagation: https://dnschecker.org
3. Wait longer (propagation can take up to 24 hours)
4. Try different DNS server: `nslookup api.quantumfond.com 1.1.1.1`

### SSL Certificate Fails

**Issue**: Certbot cannot obtain certificates

**Causes**:
- DNS not propagated yet → Wait and retry
- Port 80/443 not accessible → Check firewall
- Domain already has certificates → Use `--force-renewal`

**Retry**:
```bash
ssh root@46.224.116.254
certbot --nginx \
  -d quantumfond.com \
  -d api.quantumfond.com \
  -d app.quantumfond.com \
  --non-interactive \
  --agree-tos \
  --email admin@quantumfond.com \
  --force-renewal
```

### Port 80 Still Occupied

**Issue**: Nginx fails to bind to port 80

**Solution**:
```bash
# Check what's using port 80
ssh root@46.224.116.254 "lsof -i :80"

# Force stop all Docker containers using port 80
ssh root@46.224.116.254 "cd ~/quantum_trader && docker compose stop nginx nginx-proxy"

# Restart system nginx
ssh root@46.224.116.254 "systemctl restart nginx"
```

### Backend Not Responding

**Issue**: https://api.quantumfond.com returns 502 Bad Gateway

**Check containers**:
```bash
ssh root@46.224.116.254 "cd ~/quantum_trader && docker compose --profile dashboard ps"
```

**Restart backend**:
```bash
ssh root@46.224.116.254 "cd ~/quantum_trader && docker compose --profile dashboard restart dashboard-backend"
```

---

## Manual Deployment (Alternative)

If the automated script fails, you can deploy manually:

```bash
# 1. Stop conflicting container
ssh root@46.224.116.254 "cd ~/quantum_trader && docker compose stop nginx"

# 2. Deploy HTTP nginx
ssh root@46.224.116.254 "cp /root/quantum_trader/nginx/quantumfond-http.conf /etc/nginx/sites-available/quantumfond.conf && \
  ln -sf /etc/nginx/sites-available/quantumfond.conf /etc/nginx/sites-enabled/ && \
  nginx -t && \
  systemctl start nginx && \
  systemctl enable nginx"

# 3. Test HTTP
curl http://api.quantumfond.com/health

# 4. Obtain SSL
ssh root@46.224.116.254 "certbot --nginx \
  -d quantumfond.com \
  -d api.quantumfond.com \
  -d app.quantumfond.com \
  --non-interactive \
  --agree-tos \
  --email admin@quantumfond.com"

# 5. Deploy HTTPS
ssh root@46.224.116.254 "cp /root/quantum_trader/nginx/quantumfond.conf /etc/nginx/sites-available/quantumfond.conf && \
  systemctl reload nginx"

# 6. Test HTTPS
curl https://api.quantumfond.com/health
```

---

## Architecture Overview

```
Internet
    ↓
quantumfond.com (DNS: 46.224.116.254)
    ↓
System Nginx (Port 80/443) + SSL Termination
    ↓
    ├─→ api.quantumfond.com → Docker Backend (Port 8025)
    └─→ app.quantumfond.com → Frontend Static Files (dist/)
```

**Services:**
- System Nginx: Handles SSL, routes requests
- Dashboard Backend: FastAPI on port 8025
- Dashboard Frontend: React SPA served as static files
- PostgreSQL: Database on port 5432 (internal)
- Redis: Cache on port 6379 (internal)

---

## Configuration Files

All configurations are already in place:

| File | Purpose |
|------|---------|
| `dashboard_v4/.env` | Backend production environment |
| `dashboard_v4/frontend/.env.production` | Frontend production URLs |
| `nginx/quantumfond.conf` | HTTPS nginx config (full SSL) |
| `nginx/quantumfond-http.conf` | HTTP nginx config (initial setup) |
| `deploy_production.sh` | Automated deployment script |

---

## Next Steps After Deployment

1. **Monitor Logs**:
   ```bash
   # Backend logs
   ssh root@46.224.116.254 "docker logs -f quantum_dashboard_backend"
   
   # Nginx logs
   ssh root@46.224.116.254 "tail -f /var/log/nginx/error.log"
   ```

2. **Setup Monitoring** (optional):
   - Grafana: http://46.224.116.254:3001
   - Prometheus: http://46.224.116.254:9090

3. **Configure Alerts** (optional):
   - Add uptime monitoring (e.g., UptimeRobot)
   - Setup email notifications for errors

4. **Backup Strategy**:
   - Database backups: `pg_dump`
   - SSL certificates: `/etc/letsencrypt/`
   - Configuration files: Git repository

---

## Support

- **Dashboard GitHub**: https://github.com/binyaminsemerci-ops/quantum_trader
- **Issues**: Create issue in repository
- **Logs Location**: `/var/log/nginx/`, Docker container logs

---

**Last Updated**: December 26, 2025
**Version**: 1.0.0
