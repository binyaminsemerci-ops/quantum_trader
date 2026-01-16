# 🚀 Dashboard V3 Docker Startup Guide

## Starte Dashboard V3 med Docker

### 1. **Start Backend + Frontend**
```bash
# Start både backend og frontend
systemctl --profile dev up -d

# Eller start bare det du trenger:
systemctl up -d backend    # Backend først
systemctl up -d frontend   # Deretter frontend
```

### 2. **Verifiser at alt kjører**
```bash
# Sjekk status
systemctl ps

# Skal vise:
# quantum_backend       Running   0.0.0.0:8000->8000/tcp
# quantum_frontend_v3   Running   0.0.0.0:3000->3000/tcp
```

### 3. **Se logger**
```bash
# Backend logs
journalctl -u quantum_backend.service -f

# Frontend logs  
journalctl -u quantum_frontend_v.service3 -f

# Begge samtidig
systemctl logs -f backend frontend
```

### 4. **Åpne Dashboard**
- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **WebSocket:** ws://localhost:8000/ws/dashboard

---

## 🔧 Troubleshooting

### Frontend kan ikke koble til backend?

**Problem:** Frontend viser "Cannot connect to backend"

**Løsning 1 - Sjekk at backend kjører:**
```bash
curl http://localhost:8000/health
# Skal returnere: {"status":"healthy"}
```

**Løsning 2 - Restart services:**
```bash
systemctl restart backend frontend
```

**Løsning 3 - Rebuild hvis du har endret kode:**
```bash
systemctl down
systemctl up --build -d backend frontend
```

### WebSocket kobler ikke?

**Sjekk at backend WebSocket endpoint fungerer:**
```bash
# Install wscat hvis du ikke har det
npm install -g wscat

# Test WebSocket
wscat -c ws://localhost:8000/ws/dashboard
```

Du skal få: `Connected (press CTRL+C to quit)`

### Port 3000 eller 8000 er opptatt?

**Finn prosess som bruker porten:**
```powershell
# Port 8000
netstat -ano | findstr :8000

# Port 3000  
netstat -ano | findstr :3000

# Drep prosess (erstatt <PID> med tall fra output over)
taskkill /PID <PID> /F
```

### Database errors?

**Reset database:**
```bash
# Stopp alt
systemctl down

# Slett volumes (ADVARSEL: sletter all data)
docker volume rm quantum_trader_redis_data

# Start på nytt
systemctl --profile dev up -d
```

---

## 📦 Stoppe alt

```bash
# Stopp services (data bevares)
systemctl stop

# Stopp og fjern containers (data bevares)
systemctl down

# Stopp og fjern containers + volumes (SLETTER DATA)
systemctl down -v
```

---

## 🔄 Oppdatere Dashboard V3

Hvis du har gjort endringer i koden:

### Backend endringer:
```bash
systemctl restart backend
# Eller rebuild hvis du har nye avhengigheter:
systemctl up --build -d backend
```

### Frontend endringer:
```bash
# Next.js har hot-reload, så endringer vises automatisk
# Men hvis det ikke fungerer:
systemctl restart frontend

# Eller rebuild:
systemctl up --build -d frontend
```

---

## 🎯 Quick Commands

```bash
# Start alt
systemctl --profile dev up -d

# Se status
systemctl ps

# Se logs
systemctl logs -f

# Restart alt
systemctl restart

# Stopp alt
systemctl down

# Full rebuild
systemctl down && systemctl --profile dev up --build -d
```

---

## ✅ Verifiser at alt fungerer

1. **Backend health check:**
   ```bash
   curl http://localhost:8000/health
   # → {"status":"healthy"}
   ```

2. **Frontend åpner:**
   - Gå til http://localhost:3000
   - Skal se Dashboard med 5 tabs

3. **WebSocket kobler:**
   - Se på nederste status bar i dashboard
   - Skal vise grønn prikk og "Connected"

4. **Data lastes:**
   - Tabs skal vise ekte data (ikke bare loading spinners)
   - Timestamp nederst skal oppdateres hvert 5. sekund

---

## 📊 Dashboard V3 Features

### 5 Tabs:
1. **Overview** - GO-LIVE status, PnL, risk state, ESS
2. **Trading** - Live posisjoner, ordrer, signaler
3. **Risk** - RiskGate decisions, VaR/ES, drawdown
4. **System** - Services health, exchange status
5. **Classic** - Legacy dashboard view

### Real-time updates:
- WebSocket for sanntidsdata
- Polling fallback hvis WebSocket feiler
- Auto-refresh hver 3-15 sekund (avhengig av tab)

---

## 🆘 Hjelp

**Hvis noe ikke fungerer:**

1. Sjekk logger: `systemctl logs -f backend frontend`
2. Restart: `systemctl restart`
3. Full reset: `systemctl down && systemctl --profile dev up -d`
4. Se dokumentasjon: `DASHBOARD_V3_README.md`
5. Kjør validering: `python tests/validate_dashboard_v3_frontend.py`

---

**Laget:** 5. desember 2025  
**EPIC:** DASHBOARD-V3-001  
**Status:** ✅ Production Ready

