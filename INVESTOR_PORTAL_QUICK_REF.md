# 🚀 INVESTOR PORTAL - QUICK REFERENCE CARD

**Domain:** investor.quantumfond.com  
**VPS:** 46.224.116.254  
**Status:** ✅ DEPLOYED (HTTP) | ⏳ HTTPS PENDING

---

## 📍 CURRENT ACCESS

```
HTTP (Now):  http://46.224.116.254
HTTPS (Soon): https://investor.quantumfond.com
```

---

## ⚡ QUICK COMMANDS

### **SSH to VPS**
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
```

### **PM2 Management**
```bash
pm2 list                              # View all processes
pm2 logs quantumfond-investor         # Live logs
pm2 restart quantumfond-investor      # Restart app
pm2 stop quantumfond-investor         # Stop app
pm2 start quantumfond-investor        # Start app
pm2 monit                             # Resource monitor
```

### **Nginx Management**
```bash
systemctl status nginx                # Check status
systemctl reload nginx                # Reload config
nginx -t                              # Test config
tail -f /var/log/nginx/*.log          # Watch logs
```

### **Application Checks**
```bash
curl -I http://localhost:3001         # Test app directly
curl -I http://localhost              # Test via Nginx
pm2 logs quantumfond-investor --lines 50  # Last 50 logs
```

---

## 🔧 NEXT STEPS (DO THESE NOW)

### **1. Configure DNS** (5 min)
```
DNS Provider: [Your DNS Provider]
Type: A
Name: investor.quantumfond.com
Value: 46.224.116.254
TTL: 300

Wait: 5-30 minutes for propagation
Test: nslookup investor.quantumfond.com
```

### **2. Install SSL** (5 min - after DNS)
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254
certbot --nginx -d investor.quantumfond.com
# Follow prompts (choose redirect HTTP→HTTPS)
```

### **3. Verify HTTPS** (2 min)
```bash
curl -I https://investor.quantumfond.com
# Should return: HTTP/2 200

# Test SSL grade
# Visit: https://www.ssllabs.com/ssltest/
```

---

## 🎯 TEST CHECKLIST

### **Login Test**
- [ ] Open https://investor.quantumfond.com/login
- [ ] Enter: username: `investor`, password: `demo123`
- [ ] Should redirect to dashboard

### **Pages Test**
- [ ] Dashboard (6 KPI cards display)
- [ ] Portfolio (positions table loads)
- [ ] Performance (equity curve displays)
- [ ] Risk (metrics with color coding)
- [ ] Models (ensemble table shows)
- [ ] Reports (download buttons work)

### **Downloads Test**
- [ ] Click "Download JSON Report" → file downloads
- [ ] Click "Download CSV Report" → file downloads
- [ ] Click "Download PDF Report" → file downloads

---

## 🔐 SECURITY CHECKS

### **After SSL Installation**
```bash
# 1. Check HTTPS redirect
curl -I http://investor.quantumfond.com
# Should return: 301 → https://

# 2. Check SSL certificate
curl -vI https://investor.quantumfond.com 2>&1 | grep "SSL certificate"
# Should show Let's Encrypt

# 3. Test security headers
curl -I https://investor.quantumfond.com | grep -E "(Strict-Transport|X-Frame|X-Content)"
# Should show security headers
```

---

## 📊 MONITORING

### **Health Check**
```bash
# Application status
pm2 list | grep investor
# Should show: online, 0 restarts

# Memory usage
pm2 show quantumfond-investor | grep memory
# Should be < 100 MB

# Response time
time curl -I http://localhost:3001
# Should be < 1 second
```

### **Error Checking**
```bash
# PM2 errors
pm2 logs quantumfond-investor --err --lines 20

# Nginx errors
tail -20 /var/log/nginx/investor.quantumfond.com.error.log

# System resources
top -bn1 | grep node
# Check CPU and memory
```

---

## 🚨 TROUBLESHOOTING

### **App Not Responding**
```bash
pm2 restart quantumfond-investor
pm2 logs quantumfond-investor --lines 50
```

### **Nginx 502 Bad Gateway**
```bash
# Check if app is running
pm2 list | grep investor

# Check app port
netstat -tlnp | grep 3001

# Restart both
pm2 restart quantumfond-investor
systemctl reload nginx
```

### **SSL Not Working**
```bash
# Check certificate
certbot certificates

# Renew if needed
certbot renew --nginx

# Check Nginx config
nginx -t
```

---

## 📱 DEMO CREDENTIALS

```
Username: investor
Password: demo123
```

---

## 🔗 USEFUL LINKS

- **Application:** https://investor.quantumfond.com
- **SSL Test:** https://www.ssllabs.com/ssltest/
- **Security Headers:** https://securityheaders.com/
- **Backend API:** https://api.quantumfond.com

---

## 💾 BACKUP & RESTORE

### **Backup**
```bash
cd /home/qt/quantum_trader/frontend_investor
tar -czf backup-$(date +%Y%m%d).tar.gz .next package.json
```

### **Restore**
```bash
tar -xzf backup-YYYYMMDD.tar.gz
pm2 restart quantumfond-investor
```

---

## 🔄 UPDATE PROCEDURE

```bash
# 1. Build locally (Windows)
cd C:\quantum_trader\frontend_investor
npm run build

# 2. Deploy
.\deploy.ps1

# 3. Verify
ssh root@46.224.116.254
pm2 list
curl -I http://localhost:3001
```

---

## 📞 EMERGENCY CONTACTS

- **DevOps:** devops@quantumfond.com
- **Security:** security@quantumfond.com
- **Hotline:** +47 XXX XX XXX

---

**Quick Reference Version:** 1.0  
**Last Updated:** December 27, 2025  
**Print this card and keep it handy!**
