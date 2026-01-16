# GRAFANA INTEGRATION INTO QUANTUMFOND.COM DASHBOARD

**Date**: January 3, 2026  
**Status**: ✅ READY FOR DEPLOYMENT  
**URL**: https://app.quantumfond.com/grafana

---

## 📋 OVERVIEW

Grafana is now integrated directly into your quantumfond.com dashboard instead of requiring SSH tunneling to localhost. This provides:

- **Public Access**: No SSH tunnel needed - access via web browser
- **Iframe Embedding**: Embed dashboards in your frontend pages
- **Anonymous Viewing**: Metrics visible to all users (read-only)
- **Admin Control**: Full admin access for configuration

---

## 🎯 IMPLEMENTATION SUMMARY

### **1. Docker Compose Changes** (`systemctl.monitoring.yml`)

```yaml
grafana:
  environment:
    - GF_SERVER_ROOT_URL=https://app.quantumfond.com/grafana
    - GF_SERVER_SERVE_FROM_SUB_PATH=true
    - GF_AUTH_ANONYMOUS_ENABLED=true          # Allow public viewing
    - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer       # Read-only access
    - GF_SECURITY_ALLOW_EMBEDDING=true        # Enable iframe
    - GF_SECURITY_COOKIE_SAMESITE=none        # Cross-origin support
```

**Key Features**:
- ✅ Runs at `/grafana` subpath on quantumfond.com
- ✅ Anonymous users can view dashboards (no login required)
- ✅ Embeddable in iframe for frontend integration
- ✅ Admin access preserved (admin/quantum2026secure)

### **2. Nginx Reverse Proxy** (`nginx/app.quantumfond.com.conf`)

```nginx
location /grafana/ {
    proxy_pass http://127.0.0.1:3001/;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

**Routes**:
- `https://app.quantumfond.com/grafana` → Grafana UI
- `https://app.quantumfond.com/api` → Dashboard Backend
- `https://app.quantumfond.com/api/rl-dashboard` → RL Monitor

---

## 🚀 DEPLOYMENT STEPS

### **Quick Deploy**:

```bash
wsl bash scripts/deploy_grafana_to_quantumfond.sh
```

### **Manual Steps**:

1. **Commit and Push**:
   ```bash
   git add systemctl.monitoring.yml nginx/app.quantumfond.com.conf
   git commit -m "feat: Integrate Grafana into quantumfond.com"
   git push origin main
   ```

2. **Deploy on VPS**:
   ```bash
   ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
   cd /home/qt/quantum_trader
   git pull origin main
   docker compose -f systemctl.monitoring.yml up -d grafana
   nginx -t && nginx -s reload
   ```

3. **Verify**:
   ```bash
   journalctl -u quantum_grafana.service --tail 20
   curl -I https://app.quantumfond.com/grafana/
   ```

---

## 🌐 ACCESS URLS

### **For Users**:
- **Main URL**: https://app.quantumfond.com/grafana
- **P1-B Logs Dashboard**: https://app.quantumfond.com/grafana/d/p1b-logs
- **P1-C Baseline Dashboard**: https://app.quantumfond.com/grafana/d/p1c-baseline
- **P1-C Full Dashboard**: https://app.quantumfond.com/grafana/d/p1c-performance-baseline

### **For Admin**:
- **Login URL**: https://app.quantumfond.com/grafana/login
- **Credentials**: admin / quantum2026secure

---

## 📊 IFRAME EMBEDDING

### **Standard Embed**:

```html
<iframe 
  src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv" 
  width="100%" 
  height="600px" 
  frameborder="0">
</iframe>
```

### **Kiosk Modes**:

- `?kiosk=tv` - Full screen, no controls (best for embedding)
- `?kiosk` - Full screen with time picker
- No param - Full Grafana UI

### **Time Range**:

```html
<!-- Last 6 hours -->
<iframe src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv&from=now-6h&to=now"></iframe>

<!-- Last 24 hours -->
<iframe src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv&from=now-24h&to=now"></iframe>
```

### **Auto-Refresh**:

```html
<!-- Refresh every 30 seconds -->
<iframe src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv&refresh=30s"></iframe>
```

---

## 🔐 SECURITY NOTES

### **Anonymous Access**:
- ✅ **Viewer role only** - Can view, cannot edit
- ✅ **No data modification** - Read-only access
- ✅ **No config changes** - Cannot add data sources or dashboards
- ✅ **No alerts** - Cannot create or modify alerts

### **Admin Access**:
- 🔒 **Password protected** - admin/quantum2026secure
- 🔒 **Full control** - Can edit dashboards, configure data sources
- 🔒 **User management** - Can create/modify users

### **HTTPS Only**:
- ✅ SSL/TLS via Let's Encrypt certificate
- ✅ Secure cookie transmission
- ✅ No plain HTTP access

---

## 📱 FRONTEND INTEGRATION EXAMPLE

Create a new page in your dashboard: `dashboard_v4/frontend/src/pages/Metrics.jsx`

```jsx
import React from 'react';

const MetricsPage = () => {
  return (
    <div className="metrics-page">
      <h1>System Performance Metrics</h1>
      
      {/* P1-C Performance Baseline */}
      <section className="dashboard-section">
        <h2>Performance Baseline</h2>
        <iframe 
          src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv&refresh=30s"
          width="100%"
          height="600px"
          frameBorder="0"
          title="P1-C Performance Baseline"
        />
      </section>

      {/* P1-B Logs */}
      <section className="dashboard-section">
        <h2>System Logs</h2>
        <iframe 
          src="https://app.quantumfond.com/grafana/d/p1b-logs?kiosk=tv&refresh=30s"
          width="100%"
          height="600px"
          frameBorder="0"
          title="P1-B Log Analysis"
        />
      </section>
    </div>
  );
};

export default MetricsPage;
```

**Add to router** in `App.jsx`:

```jsx
import MetricsPage from './pages/Metrics';

// In your routes
<Route path="/metrics" element={<MetricsPage />} />
```

---

## 🧪 TESTING CHECKLIST

After deployment, verify:

### **1. External Access**:
```bash
curl -I https://app.quantumfond.com/grafana/
# Expected: HTTP/1.1 200 OK
```

### **2. Anonymous Viewing**:
1. Open incognito window
2. Navigate to https://app.quantumfond.com/grafana/d/p1c-baseline
3. Should see dashboard WITHOUT login prompt

### **3. Admin Access**:
1. Navigate to https://app.quantumfond.com/grafana/login
2. Login with admin/quantum2026secure
3. Should have full edit permissions

### **4. Iframe Embedding**:
```html
<!-- Create test.html -->
<!DOCTYPE html>
<html>
<body>
  <iframe src="https://app.quantumfond.com/grafana/d/p1c-baseline?kiosk=tv" 
          width="100%" height="600px"></iframe>
</body>
</html>
```

### **5. WebSocket Live Updates**:
1. Open dashboard
2. Check browser console for WebSocket errors
3. Verify metrics update in real-time

---

## 📊 AVAILABLE DASHBOARDS

### **P1-B: Log Analysis**
- **UID**: `p1b-logs`
- **URL**: `/grafana/d/p1b-logs`
- **Panels**: 12 (Loki queries, log trends, error analysis)

### **P1-C: Performance Baseline**
- **UID**: `p1c-baseline` or `p1c-performance-baseline`
- **URL**: `/grafana/d/p1c-baseline`
- **Panels**: 11 (CPU, Memory, Disk, Containers, Redis, Network)

---

## 🔧 TROUBLESHOOTING

### **Problem**: 502 Bad Gateway at /grafana
**Solution**:
```bash
ssh root@46.224.116.254
systemctl list-units | grep grafana  # Check if running
journalctl -u quantum_grafana.service --tail 50  # Check logs
docker restart quantum_grafana
```

### **Problem**: "Origin not allowed" in iframe
**Solution**: Already configured via:
```yaml
GF_SECURITY_COOKIE_SAMESITE=none
GF_SECURITY_ALLOW_EMBEDDING=true
```

### **Problem**: Anonymous user cannot see dashboards
**Solution**:
1. Login as admin
2. Go to Configuration → Data Sources
3. Set datasources as "Default" and "Public"
4. Go to Dashboard Settings → Permissions
5. Add "Viewer" role with "View" permission

---

## 🎯 NEXT STEPS

### **1. Deploy** (5 minutes):
```bash
wsl bash scripts/deploy_grafana_to_quantumfond.sh
```

### **2. Test Access** (2 minutes):
```bash
# From local machine
curl -I https://app.quantumfond.com/grafana/

# From browser
open https://app.quantumfond.com/grafana/d/p1c-baseline
```

### **3. Integrate in Dashboard** (15 minutes):
- Create Metrics page in React dashboard
- Add iframe embeds for P1-B and P1-C dashboards
- Add navigation link to /metrics

### **4. Optional - Custom Dashboards** (ongoing):
- Create role-specific dashboards (trader, admin, observer)
- Add RL-specific metrics (reward, loss, exploration rate)
- Build trade-specific views (PnL, positions, signals)

---

## 📈 BENEFITS

### **Before** (SSH Tunnel):
- ❌ Requires SSH tunnel setup
- ❌ localhost:3000 only accessible from local machine
- ❌ Cannot share with team or embed
- ❌ Manual tunnel management

### **After** (Quantumfond.com):
- ✅ **Public HTTPS access** - No tunnel needed
- ✅ **Team sharing** - Anyone can view metrics
- ✅ **Iframe embedding** - Integrate in React dashboard
- ✅ **Always available** - No manual startup
- ✅ **Anonymous viewing** - No login required for basic access
- ✅ **Professional** - Same domain as main app

---

## 🚀 DEPLOYMENT COMMAND

```bash
wsl bash scripts/deploy_grafana_to_quantumfond.sh
```

**Expected Output**:
```
🚀 Deploying Grafana to quantumfond.com
📤 Step 1: Committing and pushing changes...
📥 Step 2: Pulling latest code on VPS...
🔨 Step 3: Rebuilding Grafana with new config...
♻️  Step 4: Restarting Grafana service...
🔄 Step 5: Reloading Nginx configuration...
✅ Step 6: Verifying Grafana...

✅ GRAFANA DEPLOYMENT COMPLETE!

🌐 Access URLs:
   Admin:  https://app.quantumfond.com/grafana
   Viewer: Anonymous access enabled

📊 Dashboard URLs:
   P1-C:   /grafana/d/p1c-baseline
```

---

## ✅ COMPLETION CHECKLIST

- [x] Grafana configured for subpath serving
- [x] Anonymous viewer access enabled
- [x] Iframe embedding enabled
- [x] Nginx reverse proxy configured
- [x] HTTPS and WebSocket support
- [x] Deployment script created
- [ ] **DEPLOY TO VPS** ← NEXT ACTION
- [ ] Test external access
- [ ] Test iframe embedding
- [ ] Create Metrics page in React dashboard

---

**STATUS**: ✅ READY TO DEPLOY  
**NEXT ACTION**: Run `wsl bash scripts/deploy_grafana_to_quantumfond.sh`


