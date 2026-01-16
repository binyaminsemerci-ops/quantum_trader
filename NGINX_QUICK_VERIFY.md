# NGINX FIX - QUICK MANUAL VERIFICATION

Run these commands to verify the nginx fix:

## 1. Check nginx config is valid
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker exec quantum_nginx nginx -t"
```
✅ Expected: `nginx: configuration file /etc/nginx/nginx.conf test is successful`

---

## 2. Verify upstream backend exists
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker exec quantum_nginx grep -A 3 'upstream backend' /etc/nginx/nginx.conf"
```
✅ Expected:
```nginx
upstream backend {
    server quantum_backend:8000;
    keepalive 32;
}
```

---

## 3. Test backend health directly
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "curl -s http://localhost:8000/health"
```
✅ Expected: `{"status":"healthy",...}`

---

## 4. Test nginx proxy to backend (HTTPS)
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "curl -k -s https://localhost:443/health"
```
✅ Expected: `{"status":"healthy",...}`

---

## 5. Check nginx container health status
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker inspect quantum_nginx --format='{{.State.Health.Status}}'"
```
✅ Expected: `healthy` (wait 30s if shows "starting")

---

## 6. Check nginx container in systemctl list-units
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "systemctl list-units | grep quantum_nginx"
```
✅ Expected: Status column should show "healthy"

---

## ONE-LINER FULL CHECK
```bash
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "echo '=== Nginx Config ===' && docker exec quantum_nginx nginx -t 2>&1 | grep successful && echo '=== Backend Health ===' && curl -s http://localhost:8000/health | grep -o '\"status\":\"[^\"]*\"' && echo '=== Nginx Proxy ===' && curl -k -s https://localhost:443/health | grep -o '\"status\":\"[^\"]*\"' && echo '=== Container Status ===' && systemctl list-units | grep quantum_nginx | grep -o 'healthy'"
```

✅ **Success looks like:**
```
=== Nginx Config ===
nginx: configuration file /etc/nginx/nginx.conf test is successful
=== Backend Health ===
"status":"healthy"
=== Nginx Proxy ===
"status":"healthy"
=== Container Status ===
healthy
```

---

## ROOT CAUSE SUMMARY

**Problem:**
- Nginx config had: `proxy_pass http://quantum_backend:8000/health;`
- Missing upstream backend definition
- Healthcheck was failing → container unhealthy

**Fix:**
1. Added `upstream backend` block
2. Changed proxy_pass to: `http://backend/health`
3. Reloaded nginx config

**Result:** Container should be healthy within 30 seconds

---

## IF STILL UNHEALTHY

```bash
# Check logs
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker logs --tail 50 quantum_nginx"

# Restart container
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "docker restart quantum_nginx"

# Wait 35 seconds
sleep 35

# Check status
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 "systemctl list-units | grep quantum_nginx"
```

