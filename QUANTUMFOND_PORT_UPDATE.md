# QuantumFond Port Configuration Update

**Updated:** 26. desember 2025  
**Purpose:** Align QuantumFond with existing quantum_trader dashboard ports and domains

---

## ⚙️ Port Changes

### Previous Configuration
| Service | Old Port | New Port |
|---------|----------|----------|
| Backend API | 8000 | **8025** |
| Frontend | 9000 | **3000** |
| PostgreSQL | 5432 | 5432 (unchanged) |
| Redis | 6380 | 6380 (unchanged) |

---

## 🌐 Domain Configuration

### Domains in Use
```
api.quantumfond.com     → Backend API (Port 8025)
app.quantumfond.com     → Frontend App (Port 3000)
quantumfond.com         → Redirect to app.quantumfond.com
www.quantumfond.com     → Redirect to app.quantumfond.com
```

### Matching quantum_trader Pattern
This aligns with the existing dashboard configuration where:
- Backend API uses port **8025**
- Frontend uses port **3000** (standard Next.js/React port)
- Root domain redirects to app subdomain

---

## 📝 Updated Files

### 1. systemctl.quantumfond.yml
```yaml
services:
  backend:
    ports:
      - "8025:8025"  # Changed from 8000
    environment:
      - API_PORT=8025
      - CORS_ORIGINS=https://app.quantumfond.com,https://quantumfond.com,http://localhost:3000

  frontend:
    ports:
      - "3000:80"  # Changed from 9000
    environment:
      - VITE_API_URL=https://api.quantumfond.com
      - VITE_WS_URL=wss://api.quantumfond.com
```

### 2. quantumfond_backend/Dockerfile
```dockerfile
EXPOSE 8025  # Changed from 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8025", "--workers", "4"]
```

### 3. nginx-quantumfond.conf
```nginx
# Backend API (api.quantumfond.com)
location / {
    proxy_pass http://localhost:8025;  # Changed from 8000
}

# Frontend (app.quantumfond.com)
location / {
    proxy_pass http://localhost:3000;  # Changed from 9000
}

# Root domain redirect
server {
    server_name quantumfond.com www.quantumfond.com;
    return 301 https://app.quantumfond.com$request_uri;
}
```

### 4. .env.quantumfond
```env
CORS_ORIGINS=https://app.quantumfond.com,https://quantumfond.com,http://localhost:3000
```

---

## 🚀 Deployment Commands

### Deploy to VPS
```bash
# Upload updated files
wsl rsync -avz --exclude='node_modules' --exclude='dist' \
  -e "ssh -i ~/.ssh/hetzner_fresh" \
  /mnt/c/quantum_trader/quantumfond_backend/ \
  root@46.224.116.254:/opt/quantumfond/quantumfond_backend/

wsl rsync -avz --exclude='node_modules' --exclude='dist' \
  -e "ssh -i ~/.ssh/hetzner_fresh" \
  /mnt/c/quantum_trader/quantumfond_frontend/ \
  root@46.224.116.254:/opt/quantumfond/quantumfond_frontend/

wsl scp -i ~/.ssh/hetzner_fresh \
  /mnt/c/quantum_trader/systemctl.quantumfond.yml \
  root@46.224.116.254:/opt/quantumfond/

wsl scp -i ~/.ssh/hetzner_fresh \
  /mnt/c/quantum_trader/nginx-quantumfond.conf \
  root@46.224.116.254:/opt/quantumfond/

# Rebuild and restart
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  cd /opt/quantumfond && \
  systemctl -f systemctl.quantumfond.yml down && \
  systemctl -f systemctl.quantumfond.yml build && \
  systemctl -f systemctl.quantumfond.yml up -d
'
```

### Verify Deployment
```bash
# Check backend health
curl http://46.224.116.254:8025/health

# Check frontend
curl -I http://46.224.116.254:3000
```

---

## 🔐 SSL & Nginx Setup

### Install Nginx Configuration
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Copy nginx config
cp /opt/quantumfond/nginx-quantumfond.conf /etc/nginx/sites-available/quantumfond

# Enable site
ln -sf /etc/nginx/sites-available/quantumfond /etc/nginx/sites-enabled/

# Test configuration
nginx -t

# Obtain SSL certificates
certbot --nginx -d api.quantumfond.com -d app.quantumfond.com -d quantumfond.com -d www.quantumfond.com

# Reload nginx
systemctl reload nginx
```

---

## ✅ Access URLs (After DNS & SSL)

### Production URLs
```
https://quantumfond.com              → Redirects to app
https://www.quantumfond.com          → Redirects to app
https://app.quantumfond.com          → Frontend Dashboard
https://api.quantumfond.com          → Backend API
https://api.quantumfond.com/docs     → API Documentation
https://api.quantumfond.com/health   → Health Check
```

### Development URLs (Direct IP)
```
http://46.224.116.254:3000           → Frontend
http://46.224.116.254:8025           → Backend API
http://46.224.116.254:8025/docs      → API Docs
http://46.224.116.254:8025/health    → Health Check
```

---

## 📊 Benefits of This Configuration

### 1. Consistency
- Matches existing quantum_trader dashboard ports
- Familiar structure for development and deployment

### 2. Standard Ports
- **3000** - Industry standard for React/Next.js apps
- **8025** - Matches quantum_trader backend API

### 3. Domain Strategy
- Clean subdomains for API and app
- Root domain redirects to primary app
- Supports www subdomain

### 4. Scalability
- Separate ports allow independent scaling
- Nginx reverse proxy enables load balancing
- Standard ports work with monitoring tools

---

## 🔄 Migration Notes

### From Previous Setup
```
Old: http://46.224.116.254:8000 → New: http://46.224.116.254:8025
Old: http://46.224.116.254:9000 → New: http://46.224.116.254:3000
```

### Update Frontend API Calls
If frontend has hardcoded API URLs:
```javascript
// Old
const API_URL = 'http://46.224.116.254:8000'

// New  
const API_URL = import.meta.env.VITE_API_URL || 'http://46.224.116.254:8025'
```

---

## 🛠️ Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8025
lsof -i :8025

# Check what's using port 3000
lsof -i :3000

# Stop conflicting services if needed
docker stop <container_name>
```

### Backend Not Responding on 8025
```bash
# Check container logs
journalctl -u quantumfond_backend.service

# Verify port mapping
systemctl list-units --format "table {{.Names}}\t{{.Ports}}"

# Test internal container
docker exec quantumfond_backend curl localhost:8025/health
```

### Frontend Not Accessible on 3000
```bash
# Check container logs
journalctl -u quantumfond_frontend.service

# Verify nginx is proxying correctly
curl -v http://localhost:3000
```

---

>>> **QuantumFond now aligned with quantum_trader port configuration** <<<

Ready for production deployment with SSL on quantumfond.com domains! 🚀

