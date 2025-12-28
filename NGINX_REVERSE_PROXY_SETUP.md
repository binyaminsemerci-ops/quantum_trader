# 🌐 Nginx Reverse Proxy Setup for quantum.quantumfond.com

## 📋 Prerequisites

**Before running the setup script, you MUST configure DNS:**

### Step 1: Get Server IP
```bash
# On VPS, run:
curl ifconfig.me
# Example output: 46.224.116.254
```

### Step 2: Configure DNS

Go to your DNS provider (Cloudflare, Namecheap, etc.) and add:

**DNS Record:**
- **Type:** A
- **Name:** `quantum` (or `quantum.quantumfond` depending on provider)
- **Value:** `46.224.116.254` (your VPS IP)
- **TTL:** Auto or 3600

**Full domain will be:** `quantum.quantumfond.com`

### Step 3: Wait for DNS Propagation (5-30 minutes)

Check DNS propagation:
```bash
# From your local machine:
nslookup quantum.quantumfond.com

# Or check online:
# https://dnschecker.org/#A/quantum.quantumfond.com
```

---

## 🚀 Deployment Steps

### 1. Commit and Push Nginx Config
```powershell
git add nginx/quantum.quantumfond.com.conf scripts/setup_quantum_nginx.sh
git commit -m "Add nginx reverse proxy config for quantum.quantumfond.com"
git push origin main
```

### 2. Deploy to VPS
```bash
# SSH into VPS
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254

# Pull latest code
cd /home/qt/quantum_trader
git pull origin main

# Make script executable
chmod +x scripts/setup_quantum_nginx.sh

# Run setup script
sudo ./scripts/setup_quantum_nginx.sh
```

### 3. If DNS Not Ready

The script will check DNS automatically. If DNS is not ready, you'll see:

```
⚠️  DNS does not point to this server!
Please update your DNS:
  Add A record: quantum.quantumfond.com → 46.224.116.254

After DNS propagation, run:
  sudo certbot --nginx -d quantum.quantumfond.com --email admin@quantumfond.com --agree-tos --no-eff-email
```

Wait for DNS propagation and run the certbot command.

---

## ✅ Expected Result

After successful setup:

### Main Dashboard
**URL:** https://quantum.quantumfond.com
- Full QuantumFond interface
- All pages accessible
- Auto-redirect from HTTP to HTTPS

### RL Intelligence
**URL:** https://quantum.quantumfond.com/rl-intelligence
- Real-time RL charts
- Performance heatmap
- Correlation matrix

### RL Dashboard API
**URL:** https://quantum.quantumfond.com/api/rl-dashboard/data
- Direct API access
- Returns JSON with rewards data

---

## 🔧 Configuration Details

### Nginx Setup
- **Config file:** `/etc/nginx/sites-available/quantum.quantumfond.com`
- **Symlink:** `/etc/nginx/sites-enabled/quantum.quantumfond.com`
- **Proxy target:** `http://localhost:3002` (QuantumFond frontend)
- **API proxy:** `/api/rl-dashboard/` → `http://localhost:8027/`

### SSL Certificate
- **Provider:** Let's Encrypt
- **Auto-renewal:** Daily at 3 AM (via cron)
- **Certificate path:** `/etc/letsencrypt/live/quantum.quantumfond.com/`

### Security Features
- ✅ HTTPS redirect (HTTP → HTTPS)
- ✅ HSTS header (max-age: 1 year)
- ✅ X-Frame-Options: SAMEORIGIN
- ✅ X-Content-Type-Options: nosniff
- ✅ X-XSS-Protection: enabled
- ✅ WebSocket support (for SocketIO)

---

## 🧪 Testing

### Test HTTP → HTTPS Redirect
```bash
curl -I http://quantum.quantumfond.com
# Should show: 301 Moved Permanently
# Location: https://quantum.quantumfond.com
```

### Test HTTPS
```bash
curl -I https://quantum.quantumfond.com
# Should show: 200 OK
```

### Test RL Dashboard API
```bash
curl https://quantum.quantumfond.com/api/rl-dashboard/data
# Should return JSON with rewards data
```

### Test from Browser
1. Open: https://quantum.quantumfond.com
2. Click "🧠 RL Intelligence" in sidebar
3. Verify charts are loading real-time data

---

## 🔍 Troubleshooting

### Check Nginx Status
```bash
sudo systemctl status nginx
sudo nginx -t  # Test configuration
```

### Check SSL Certificate
```bash
sudo certbot certificates
```

### View Nginx Logs
```bash
sudo tail -f /var/log/nginx/quantum.quantumfond.com.access.log
sudo tail -f /var/log/nginx/quantum.quantumfond.com.error.log
```

### Reload Nginx After Changes
```bash
sudo nginx -t && sudo systemctl reload nginx
```

### Manual SSL Renewal
```bash
sudo certbot renew --force-renewal
sudo systemctl reload nginx
```

### Check Container Status
```bash
docker ps --filter name=quantum_quantumfond_frontend
docker logs quantum_quantumfond_frontend
```

---

## 🔄 Updates

### Update Nginx Config
```bash
cd /home/qt/quantum_trader
git pull origin main
sudo cp nginx/quantum.quantumfond.com.conf /etc/nginx/sites-available/quantum.quantumfond.com
sudo nginx -t && sudo systemctl reload nginx
```

### Update QuantumFond Frontend
```bash
cd /home/qt/quantum_trader
git pull origin main
docker compose -f docker-compose.vps.yml build quantumfond-frontend
docker compose -f docker-compose.vps.yml up -d quantumfond-frontend
```

---

## 📊 Architecture

```
Internet
   ↓
HTTPS (443)
   ↓
Nginx Reverse Proxy (quantum.quantumfond.com)
   ↓
┌─────────────────┬────────────────────┐
│  / (root)       │  /api/rl-dashboard │
│  ↓              │  ↓                 │
│  localhost:3002 │  localhost:8027    │
│  (Frontend)     │  (RL Dashboard)    │
└─────────────────┴────────────────────┘
```

---

## 📝 Notes

- **Port 80/443** must be open in firewall (standard for web traffic)
- **Port 3002** can remain closed (internal only)
- **Port 8027** can remain closed (internal only)
- DNS changes take 5-30 minutes to propagate
- SSL certificate auto-renews every 60 days
- Nginx logs rotate automatically

---

## 🎯 Quick Commands

```bash
# Full deployment (after DNS is ready)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  "cd /home/qt/quantum_trader && \
   git pull && \
   chmod +x scripts/setup_quantum_nginx.sh && \
   ./scripts/setup_quantum_nginx.sh"

# Check everything
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 \
  "systemctl status nginx && \
   docker ps --filter name=quantumfond && \
   certbot certificates"
```

---

**Ready to deploy? Run the setup script after DNS is configured! 🚀**
