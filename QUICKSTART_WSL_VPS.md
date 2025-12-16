# 🎯 QUANTUM TRADER - QUICK START (WSL → VPS)

**Opprettet:** 2025-12-16  
**Status:** ✅ Klar for produksjon  
**Platform:** WSL2 + Podman → VPS Ubuntu/Debian

---

## 📋 WSL SETUP (Lokalt først)

### 1️⃣ Forberedelser i WSL
```bash
# Gå til prosjektet
cd ~/quantum_trader

# Sjekk at alle filer er på plass
ls -la docker-compose.wsl.yml scripts/start-wsl-podman.sh

# Gjør skriptene kjørbare
chmod +x scripts/*.sh
```

### 2️⃣ Start Services i WSL
```bash
# Metode A: Bruk oppstartsskript (ANBEFALT)
./scripts/start-wsl-podman.sh

# ELLER Metode B: Manuelt
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
```

### 3️⃣ Verifiser at det fungerer
```bash
# Kjør verifikasjonsskript
./scripts/verify-wsl-podman.sh

# Sjekk at containere kjører
podman ps

# Test health endpoint
curl http://localhost:8001/health
```

---

## 🌐 VPS DEPLOYMENT

### STEG 1: Initial VPS Setup

#### 1.1 Logg inn på VPS
```bash
# SSH til VPS (erstatt med din IP)
ssh root@YOUR_VPS_IP
```

#### 1.2 Kjør setup-script
```bash
# Kopier setup-script til VPS
# Fra din lokale maskin (WSL):
scp scripts/setup-vps.sh root@YOUR_VPS_IP:~/

# På VPS:
chmod +x ~/setup-vps.sh
./setup-vps.sh
```

**Dette installerer:**
- ✅ Podman
- ✅ Python 3 + pip
- ✅ podman-compose
- ✅ Git, curl, jq
- ✅ UFW Firewall (åpner porter 22, 8000, 8001)

---

### STEG 2: Clone Repository

```bash
# På VPS:
cd ~
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git
cd quantum_trader
```

---

### STEG 3: Konfigurer Environment

#### 3.1 Kopier .env fil fra lokal WSL
```bash
# Fra din WSL:
scp ~/quantum_trader/.env root@YOUR_VPS_IP:~/quantum_trader/.env
```

#### 3.2 ELLER opprett .env manuelt på VPS
```bash
# På VPS:
nano ~/quantum_trader/.env

# Lim inn dine API nøkler og konfigurasjoner
# (Se .env.example for referanse)
```

**Kritiske verdier i .env:**
```env
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
BINANCE_TESTNET=true
REDIS_URL=redis://redis:6379
AI_MODEL=hybrid
```

---

### STEG 4: Start Services på VPS

```bash
# På VPS:
cd ~/quantum_trader

# Gjør skript kjørbart
chmod +x scripts/start-wsl-podman.sh
chmod +x scripts/verify-wsl-podman.sh

# Start Redis + AI Engine
./scripts/start-wsl-podman.sh
```

---

### STEG 5: Verifiser Deployment

```bash
# På VPS:
cd ~/quantum_trader

# Kjør verifikasjonsskript
./scripts/verify-wsl-podman.sh

# Sjekk containere
podman ps

# Test health endpoint
curl http://localhost:8001/health

# Se logs
podman logs quantum_ai_engine
```

---

## 📊 DEPLOYMENT CHECKLIST

### ✅ Pre-Deployment
- [ ] Testet i WSL
- [ ] Verifisert at imports fungerer
- [ ] Ingen /mnt/c paths
- [ ] .env fil konfigurert
- [ ] API credentials klare

### ✅ VPS Setup
- [ ] VPS opprettet og tilgjengelig via SSH
- [ ] DNS konfigurert (valgfritt)
- [ ] Firewall konfigurert (UFW)
- [ ] Podman installert
- [ ] podman-compose installert

### ✅ Deployment
- [ ] Repository klonet til ~/quantum_trader
- [ ] .env fil kopiert/opprettet
- [ ] Services startet med podman-compose
- [ ] Health checks passing
- [ ] Logs ser bra ut

### ✅ Monitoring
- [ ] Health endpoints tilgjengelige
- [ ] Redis responderer
- [ ] AI Engine returnerer predictions
- [ ] Ingen ImportErrors i logs

---

## 🔧 TROUBLESHOOTING

### Problem: Connection refused på VPS
```bash
# Sjekk at firewall tillater trafikk
sudo ufw status

# Åpne port hvis nødvendig
sudo ufw allow 8001/tcp
```

### Problem: Container starter ikke
```bash
# Se detaljerte logs
podman logs quantum_ai_engine

# Rebuild image
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml build ai-engine

# Restart
podman-compose -f docker-compose.wsl.yml up -d ai-engine
```

### Problem: Redis connection failed
```bash
# Sjekk at Redis kjører
podman ps | grep redis

# Test Redis
podman exec quantum_redis redis-cli ping

# Restart Redis
podman-compose -f docker-compose.wsl.yml restart redis
```

### Problem: Import errors
```bash
# Verifiser PYTHONPATH
podman exec quantum_ai_engine env | grep PYTHONPATH
# Skal være: PYTHONPATH=/app

# Sjekk sys.path
podman exec quantum_ai_engine python3 -c "import sys; print(sys.path)"
# Skal IKKE inneholde /mnt/c
```

---

## 🚀 PRODUCTION TIPS

### 1. Automatisk restart ved boot
```bash
# På VPS: Legg til i crontab
crontab -e

# Legg til:
@reboot cd /root/quantum_trader && podman-compose -f docker-compose.wsl.yml up -d
```

### 2. Logging til fil
```bash
# Redirect logs til fil
podman logs -f quantum_ai_engine > ~/logs/ai-engine.log 2>&1 &
```

### 3. Monitoring script
```bash
# Kjør verifisering hver 5. minutt
crontab -e

# Legg til:
*/5 * * * * cd /root/quantum_trader && ./scripts/verify-wsl-podman.sh >> ~/logs/health-check.log 2>&1
```

### 4. Backup strategy
```bash
# Backup Redis data
podman exec quantum_redis redis-cli SAVE
cp ~/quantum_trader/redis_data/dump.rdb ~/backups/redis-$(date +%Y%m%d).rdb

# Backup logs
tar -czf ~/backups/logs-$(date +%Y%m%d).tar.gz ~/quantum_trader/logs/
```

---

## 📞 KOMMANDOER (COPY-PASTE)

### Start services
```bash
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
```

### Verifiser
```bash
./scripts/verify-wsl-podman.sh
curl http://localhost:8001/health
```

### Se logs
```bash
podman logs -f quantum_ai_engine
```

### Restart en service
```bash
podman-compose -f docker-compose.wsl.yml restart ai-engine
```

### Stopp alt
```bash
podman-compose -f docker-compose.wsl.yml down
```

### Rebuild og start
```bash
podman-compose -f docker-compose.wsl.yml up -d --build
```

---

## 🎯 SUCCESS METRICS

Du vet at deployment er vellykket når:

1. ✅ `podman ps` viser begge containere som "Up"
2. ✅ `curl http://localhost:8001/health` returnerer HTTP 200
3. ✅ Ingen ImportError eller ModuleNotFoundError i logs
4. ✅ Redis responderer på PING
5. ✅ Python sys.path inneholder kun `/app` paths (ikke `/mnt/c`)

---

## 🔐 SECURITY NOTES

### Før production:
1. ✅ Endre default credentials i .env
2. ✅ Konfigurer UFW firewall
3. ✅ Sett opp SSH nøkkel (ikke passord)
4. ✅ Enable fail2ban for SSH
5. ✅ Regelmessige system oppdateringer

---

**Lykke til med deployment! 🚀**

For spørsmål eller problemer, sjekk logs først:
```bash
podman logs quantum_ai_engine
```
