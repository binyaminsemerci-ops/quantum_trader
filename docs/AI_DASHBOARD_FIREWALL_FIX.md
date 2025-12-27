# 🔥 Dashboard Connection Timeout - RESOLVED

**Issue:** ERR_CONNECTION_TIMED_OUT når man forsøker å nå http://46.224.116.254:8080  
**Root Cause:** Brannmur blokkerer port 8080  
**Status:** ✅ Fix klar, krever sudo for å kjøre

---

## 🔍 Diagnostikk Utført

### 1. Container Status
```bash
✅ quantum_dashboard: Up 22 minutes
✅ Port 8080 mappet: 0.0.0.0:8080->8080/tcp
✅ Dashboard serves innhold lokalt (curl localhost:8080 fungerer)
```

### 2. Port Listening Status
```bash
✅ Port 8080 lytter på alle interfaces (0.0.0.0:8080)
✅ IPv6 også aktiv ([::]:8080)
```

### 3. Lokal Test
```bash
✅ Fra VPS: curl http://localhost:8080/ fungerer
❌ Fra ekstern: curl http://46.224.116.254:8080/ = Connection timeout
```

**Konklusjon:** Brannmur blokkerer ekstern tilgang til port 8080

---

## ✅ Løsning

### Metode 1: Automatisk Script (Anbefalt)

SSH til VPS og kjør:

```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254

# Kjør firewall script med sudo
sudo bash ~/quantum_trader/scripts/open_dashboard_port.sh
```

**Scriptet gjør:**
- ✅ Detekterer firewall type (UFW/firewalld/iptables)
- ✅ Åpner port 8080 for TCP trafikk
- ✅ Lagrer regler permanent
- ✅ Verifiserer konfigurasjonen

---

### Metode 2: Manuelle Kommandoer

#### Hvis UFW (Ubuntu/Debian):
```bash
sudo ufw allow 8080/tcp comment 'Quantum Trader Dashboard'
sudo ufw status | grep 8080  # Verifiser
```

#### Hvis firewalld (CentOS/RHEL):
```bash
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports | grep 8080  # Verifiser
```

#### Hvis iptables (direkte):
```bash
sudo iptables -I INPUT -p tcp --dport 8080 -j ACCEPT
sudo iptables-save > /etc/iptables/rules.v4  # Lagre permanent
sudo iptables -L INPUT -n -v | grep 8080  # Verifiser
```

---

## 🔍 Verifisering

### 1. Test fra VPS (skal fungere allerede)
```bash
curl http://localhost:8080/
# Skal returnere HTML
```

### 2. Test fra lokal maskin (etter firewall fix)
```bash
curl http://46.224.116.254:8080/
# Skal returnere HTML (ikke timeout)
```

### 3. Test i nettleser
```
http://46.224.116.254:8080
```

**Forventet resultat:**
- Dashboard vises med Quantum Trader V3 interface
- Audit log tab tilgjengelig
- WebSocket forbindelse etableres
- Real-time updates fungerer

---

## 🐛 Troubleshooting

### Problem: "sudo: a password is required"
**Løsning:** 
- Du må ha sudo-tilgang på VPS
- Kontakt VPS administrator for tilgang
- Alternativt: Legg til din bruker i sudoers

### Problem: Fortsatt timeout etter firewall fix
**Sjekk:**
```bash
# 1. Verifiser container kjører
docker ps | grep quantum_dashboard

# 2. Sjekk container logs
docker logs quantum_dashboard --tail 50

# 3. Test lokalt på VPS
curl -v http://localhost:8080/

# 4. Sjekk firewall regler
sudo iptables -L INPUT -n -v | grep 8080

# 5. Restart container hvis nødvendig
docker restart quantum_dashboard
```

### Problem: Dashboard viser "Not Found" eller 404
**Løsning:**
```bash
# Dashboard serverer på root path
http://46.224.116.254:8080/          # ✅ Riktig
http://46.224.116.254:8080/health    # ❌ Ikke implementert

# API endpoints:
http://46.224.116.254:8080/api/audit
http://46.224.116.254:8080/api/reports

# WebSocket:
ws://46.224.116.254:8080/ws/audit
```

---

## 📋 Andre Porter som Må Være Åpne

For full Quantum Trader funksjonalitet:

| Port | Service | Status | Firewall |
|------|---------|--------|----------|
| 8080 | Dashboard | ✅ Kjører | ❌ Blokkert |
| 8001 | AI Engine | ✅ Kjører | Sjekk |
| 8003 | Trading Bot | ✅ Kjører | Sjekk |
| 6379 | Redis | ✅ Kjører | Intern (OK) |
| 5432 | PostgreSQL | ✅ Kjører | Intern (OK) |
| 80/443 | Nginx | ✅ Kjører | Intern (OK) |
| 3001 | Grafana | ✅ Kjører | Intern (OK) |
| 9090 | Prometheus | ✅ Kjører | Intern (OK) |
| 9093 | Alertmanager | ✅ Kjører | Intern (OK) |

**Anbefaling:** 
- Port 8080 (Dashboard): Åpne for ekstern tilgang
- Port 8001 (AI Engine): Vurder API tilgang hvis nødvendig
- Andre porter: Hold interne (kun localhost) for sikkerhet

---

## 🔐 Sikkerhetsanbefalinger

### 1. Begrens Tilgang til Spesifikke IP-er (Valgfritt)

Hvis du bare vil gi tilgang fra din IP:

```bash
# UFW
sudo ufw allow from YOUR_IP to any port 8080 proto tcp

# iptables
sudo iptables -I INPUT -p tcp -s YOUR_IP --dport 8080 -j ACCEPT
```

### 2. Legg til HTTP Basic Auth

For ekstra sikkerhet, vurder å legge til autentisering i `dashboard/app.py`:

```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "admin" or credentials.password != "your_password":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials
```

### 3. Bruk HTTPS med Nginx Reverse Proxy

Configure Nginx som reverse proxy med SSL:
- Sett opp Let's Encrypt sertifikat
- Proxy pass til dashboard på localhost:8080
- Force HTTPS redirect

---

## 📊 Forventet Resultat Etter Fix

### Dashboard Tilgjengelig:
```
✅ http://46.224.116.254:8080/
   - Quantum Trader V3 Dashboard loads
   - Metrics visible
   - Charts rendered

✅ Audit Log Tab
   - Shows AUTO_REPAIR_AUDIT.log content
   - Search/filter working
   - Yellow highlights on matches

✅ WebSocket Connection
   - ws://46.224.116.254:8080/ws/audit connected
   - Real-time updates <3 seconds
   - Auto-reconnect on disconnect
```

### Test Scenario:
1. Åpne dashboard i nettleser
2. Gå til "Audit Log" tab
3. Skriv søkeord i search box (f.eks. "database")
4. Klikk 🔍 Search
5. Se gule highlights på matches
6. Vent på nye entries (auto-refresh hver 3 sek)

---

## 🚀 Quick Commands

### Åpne Port 8080
```bash
# Alt-i-ett kommando
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "sudo bash ~/quantum_trader/scripts/open_dashboard_port.sh"
```

### Sjekk Status
```bash
# Container status
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker ps | grep dashboard"

# Port listening
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "ss -tuln | grep 8080"

# Test lokal tilgang
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "curl -s http://localhost:8080/ | head -10"
```

### Restart Dashboard (hvis nødvendig)
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254 "docker restart quantum_dashboard"
```

---

## 📚 Relatert Dokumentasjon

- **Dashboard Features:** AI_SMART_LOG_SEARCH_DEPLOYED.md
- **WebSocket Streaming:** AI_WEBSOCKET_AUDIT_STREAMING_DEPLOYED.md
- **Testnet Setup:** AI_TESTNET_QUICK_REF.md

---

## ✅ Oppsummering

**Problem:** Dashboard ikke tilgjengelig (ERR_CONNECTION_TIMED_OUT)

**Root Cause:** Firewall blokkerer port 8080

**Fix:** 
```bash
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254
sudo bash ~/quantum_trader/scripts/open_dashboard_port.sh
```

**ETA:** < 1 minutt

**Verifisering:** Åpne http://46.224.116.254:8080 i nettleser

---

**Dato:** December 17, 2025  
**Status:** ✅ Diagnostikk komplett, fix klar  
**Krever:** Sudo-tilgang for å kjøre firewall script
