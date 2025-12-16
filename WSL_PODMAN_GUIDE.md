# 🚀 Quantum Trader - WSL + Podman Guide

**Opprettet:** 2025-12-16  
**Platform:** WSL2 (Ubuntu) + Podman + podman-compose  
**Kompatibel med:** VPS deployment senere

---

## 📋 Forutsetninger

### 1️⃣ WSL Setup
```bash
# Sjekk at du er i WSL
grep -i microsoft /proc/version

# Gå til quantum_trader katalogen
cd ~/quantum_trader
```

### 2️⃣ Installer Podman og podman-compose
```bash
# Installer podman (hvis ikke allerede installert)
sudo apt update
sudo apt install -y podman

# Installer podman-compose
pip3 install podman-compose

# Verifiser installasjoner
podman --version
podman-compose --version
```

### 3️⃣ Verifiser Python venv
```bash
# Ditt venv skal allerede fungere
source ~/quantum_trader/venv/bin/activate
python --version
```

---

## 🎯 Oppstart (Quick Start)

### Metode 1: Bruk oppstartsskript
```bash
cd ~/quantum_trader

# Gjør skript kjørbart
chmod +x scripts/start-wsl-podman.sh
chmod +x scripts/verify-wsl-podman.sh

# Start services
./scripts/start-wsl-podman.sh
```

### Metode 2: Manuelle kommandoer
```bash
cd ~/quantum_trader

# Stopp eventuelle eksisterende containere
podman-compose -f docker-compose.wsl.yml down

# Start Redis + AI-Engine
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine

# Sjekk status
podman ps
```

---

## ✅ Verifikasjon

### 1️⃣ Kjør verifikasjonsskript
```bash
cd ~/quantum_trader
./scripts/verify-wsl-podman.sh
```

### 2️⃣ Manuelle sjekker
```bash
# Sjekk at containere kjører
podman ps

# Sjekk Redis
podman exec quantum_redis redis-cli ping
# Forventet output: PONG

# Test AI Engine health endpoint
curl http://localhost:8001/health
# Forventet: JSON med "status": "healthy"

# Se AI Engine logs
podman logs quantum_ai_engine

# Se siste 50 linjer med live oppdatering
podman logs -f --tail 50 quantum_ai_engine
```

### 3️⃣ Test ServiceHealth
```bash
# Sjekk at imports fungerer uten /mnt/c collision
podman exec quantum_ai_engine python3 -c "
from microservices.ai_engine.service_health import ServiceHealth
print('✅ ServiceHealth import successful')
"
```

### 4️⃣ Verifiser PYTHONPATH
```bash
# Sjekk at PYTHONPATH = /app (IKKE /mnt/c)
podman exec quantum_ai_engine env | grep PYTHONPATH

# Sjekk Python sys.path
podman exec quantum_ai_engine python3 -c "
import sys
for p in sys.path:
    print(p)
" | grep -v "/mnt/c" || echo "✅ No /mnt/c paths found"
```

---

## 🛠️ Troubleshooting

### Problem: Container starter ikke
```bash
# Se detaljerte logs
podman logs quantum_ai_engine

# Rebuild image
podman-compose -f docker-compose.wsl.yml build ai-engine

# Start på nytt
podman-compose -f docker-compose.wsl.yml up -d ai-engine
```

### Problem: Import errors
```bash
# Sjekk at ingen /mnt/c paths er i Python
podman exec quantum_ai_engine python3 -c "import sys; print(sys.path)"

# Verifiser at PYTHONPATH=/app
podman exec quantum_ai_engine env | grep PYTHONPATH
```

### Problem: Redis connection failed
```bash
# Sjekk at Redis kjører
podman ps | grep redis

# Test Redis direkte
podman exec quantum_redis redis-cli ping

# Restart Redis
podman-compose -f docker-compose.wsl.yml restart redis
```

### Problem: Health endpoint 404/500
```bash
# Se logs for feilmeldinger
podman logs --tail 100 quantum_ai_engine

# Kjør health check manuelt
podman exec quantum_ai_engine python3 -c "
import requests
response = requests.get('http://localhost:8001/health')
print(response.status_code, response.text)
"
```

---

## 🔧 Nyttige Kommandoer

### Container Management
```bash
# List alle containere
podman ps -a

# Stopp alle services
podman-compose -f docker-compose.wsl.yml down

# Start en spesifikk service
podman-compose -f docker-compose.wsl.yml up -d redis

# Restart en service
podman-compose -f docker-compose.wsl.yml restart ai-engine

# Rebuild og start
podman-compose -f docker-compose.wsl.yml up -d --build ai-engine
```

### Logs og Debugging
```bash
# Se logs (alle services)
podman-compose -f docker-compose.wsl.yml logs

# Se logs (en service)
podman logs quantum_ai_engine

# Follow logs (live)
podman logs -f quantum_ai_engine

# Siste 50 linjer
podman logs --tail 50 quantum_ai_engine
```

### Enter Container Shell
```bash
# Åpne bash i AI Engine container
podman exec -it quantum_ai_engine bash

# Kjør Python kommandoer
podman exec quantum_ai_engine python3 -c "print('Hello from container')"
```

### Cleanup
```bash
# Stopp og fjern containere
podman-compose -f docker-compose.wsl.yml down

# Fjern volumes (⚠️ sletter Redis data)
podman volume rm quantum_trader_redis_data

# Fjern images
podman rmi quantum_ai_engine:latest

# Full cleanup (⚠️ fjerner alt)
podman system prune -a --volumes
```

---

## 🌐 VPS Deployment (Senere)

Dette oppsettet er **100% kompatibelt** med VPS deployment fordi:

### 1. **Samme Structure**
```
~/quantum_trader/       # Samme på både WSL og VPS
├── docker-compose.wsl.yml
├── microservices/
│   └── ai_engine/
├── backend/
├── models/
└── data/
```

### 2. **Ingen Hardcoded Paths**
- ✅ Bruker relative paths (`./microservices`, `./backend`)
- ✅ PYTHONPATH=/app (ikke `/mnt/c`)
- ❌ Ingen Windows-paths i container

### 3. **Samme Kommandoer**
På VPS kjører du **eksakt samme kommandoer**:
```bash
cd ~/quantum_trader
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine
```

### 4. **Environment Parity**
- Samme `.env` fil
- Samme Docker images
- Samme network konfiguration
- Samme volume struktur

### VPS Migration Checklist
Når du skal deploye på VPS:
```bash
# 1. Clone repo til VPS
git clone https://github.com/binyaminsemerci-ops/quantum_trader.git ~/quantum_trader

# 2. Copy .env fil
scp .env user@vps:~/quantum_trader/.env

# 3. Installer podman
ssh user@vps "sudo apt update && sudo apt install -y podman"

# 4. Installer podman-compose
ssh user@vps "pip3 install podman-compose"

# 5. Start services
ssh user@vps "cd ~/quantum_trader && podman-compose -f docker-compose.wsl.yml up -d"

# 6. Verifiser
ssh user@vps "curl http://localhost:8001/health"
```

---

## 📊 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│  WSL2 Ubuntu (Linux Kernel)                          │
│                                                       │
│  ┌─────────────────────────────────────────────┐    │
│  │  Podman (rootless container runtime)        │    │
│  │                                              │    │
│  │  ┌──────────────┐    ┌──────────────────┐  │    │
│  │  │   Redis      │    │   AI Engine      │  │    │
│  │  │   :6379      │◄───┤   :8001          │  │    │
│  │  │              │    │   - XGBoost      │  │    │
│  │  │              │    │   - LightGBM     │  │    │
│  │  │              │    │   - N-HiTS       │  │    │
│  │  └──────────────┘    │   - PatchTST     │  │    │
│  │                      └──────────────────┘  │    │
│  │                                              │    │
│  │  Volume: ~/quantum_trader mounted as /app  │    │
│  │  Network: quantum_trader (bridge)          │    │
│  └─────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

---

## ❗ Viktige Regler

### ✅ DO (Gjør)
- Bruk `~/quantum_trader` i WSL
- Bruk `podman-compose -f docker-compose.wsl.yml`
- Bruk relative paths i compose fil
- Sett `PYTHONPATH=/app` i container
- Test health endpoints etter start

### ❌ DON'T (Ikke gjør)
- ❌ Ikke bruk `/mnt/c/quantum_trader`
- ❌ Ikke bruk Docker Desktop
- ❌ Ikke hardcode Windows paths
- ❌ Ikke anta GPU tilgjengelig
- ❌ Ikke kjør uten å verifisere imports

---

## 🎯 Success Criteria

Du vet at det fungerer når:
1. ✅ `podman ps` viser `quantum_redis` og `quantum_ai_engine` som "Up"
2. ✅ `curl http://localhost:8001/health` returnerer `{"status":"healthy"}`
3. ✅ Ingen `/mnt/c` i Python sys.path
4. ✅ `ServiceHealth.create()` fungerer uten ImportError
5. ✅ Redis PING returnerer PONG

---

## 📞 Support

Hvis du trenger hjelp:
1. Kjør verifikasjonsskriptet: `./scripts/verify-wsl-podman.sh`
2. Sjekk logs: `podman logs quantum_ai_engine`
3. Verifiser at venv fungerer: `source venv/bin/activate && python --version`

---

**Lykke til! 🚀**
