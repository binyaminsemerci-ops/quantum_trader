# 🚀 Quantum Trader - WSL Podman Setup

## 📋 1. ANALYSE

### Eksisterende Setup
- ✅ `docker-compose.yml` - Fullstendig setup med alle services
- ✅ Redis service definert (port 6379)
- ✅ AI-Engine service definert (port 8001)
- ❌ Problem: Bruker `/mnt/c` paths → import collisions

### WSL Løsning
- ✅ Ny fil: `docker-compose.wsl.yml`
- ✅ Bruker kun `~/quantum_trader` paths
- ✅ PYTHONPATH=/app (ingen `/mnt/c`)
- ✅ Redis + AI-Engine som containere
- ✅ Backend kjører i host venv (raskere dev)

---

## 🛠️ 2. SETUP

### Kopier Prosjektet til WSL

```bash
# Fra WSL terminal
cd ~
cp -r /mnt/c/quantum_trader ~/quantum_trader
cd ~/quantum_trader
```

**VIKTIG**: Alt arbeid skjer nå i `~/quantum_trader` (IKKE `/mnt/c/quantum_trader`)

### Installer Podman

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y podman

# Installer podman-compose
pip3 install podman-compose

# Verifiser
podman --version
podman-compose --version
```

---

## 🚀 3. KJØR QUANTUM TRADER

### Metode 1: Automatisk (Anbefalt)

```bash
cd ~/quantum_trader

# Gi execute permissions
chmod +x start-wsl.sh

# Kjør alt
./start-wsl.sh
```

### Metode 2: Manuelt (Steg for steg)

```bash
cd ~/quantum_trader

# 1. Bygg AI Engine
podman-compose -f docker-compose.wsl.yml build ai-engine

# 2. Start Redis + AI-Engine
podman-compose -f docker-compose.wsl.yml up -d redis ai-engine

# 3. Se status
podman-compose -f docker-compose.wsl.yml ps

# 4. Se logs
podman-compose -f docker-compose.wsl.yml logs -f ai-engine
```

---

## 🔍 4. VERIFISER

### Sjekk Containere

```bash
# Vis kjørende containere
podman ps

# Forventet output:
# CONTAINER ID  IMAGE                        STATUS    PORTS                   NAMES
# xxx           redis:7-alpine               Up        0.0.0.0:6379->6379/tcp  quantum_redis
# yyy           localhost/quantum_ai_engine  Up        0.0.0.0:8001->8001/tcp  quantum_ai_engine
```

### Sjekk Logs

```bash
# AI Engine logs
podman-compose -f docker-compose.wsl.yml logs ai-engine

# Redis logs
podman-compose -f docker-compose.wsl.yml logs redis

# Live logs (følg i sanntid)
podman-compose -f docker-compose.wsl.yml logs -f ai-engine
```

### Test Health Endpoints

```bash
# Test Redis
podman exec quantum_redis redis-cli ping
# Forventet: PONG

# Test AI Engine
curl http://localhost:8001/health
# Forventet: {"status":"ok",...}

# Eller med httpie
http localhost:8001/health
```

### Test ServiceHealth.create()

```bash
# Aktiver venv
source .venv/bin/activate

# Test ServiceHealth fra backend
python -c "
from backend.services.monitoring_health_service.service_health import ServiceHealth
health = ServiceHealth.create('test_service', 'redis://localhost:6379')
print('✅ ServiceHealth.create() fungerer!')
print(f'   Service: {health.service_name}')
print(f'   Status: {health.status}')
"

# Forventet output:
# ✅ ServiceHealth.create() fungerer!
#    Service: test_service
#    Status: healthy
```

### Test Backend med Redis

```bash
# Start backend i venv (host, ikke container)
source .venv/bin/activate
cd ~/quantum_trader

# Kjør backend
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Test i annen terminal
curl http://localhost:8000/health
```

---

## 💡 5. NYTTIGE KOMMANDOER

### Container Management

```bash
# Stopp alt
podman-compose -f docker-compose.wsl.yml down

# Stopp én service
podman-compose -f docker-compose.wsl.yml stop ai-engine

# Restart service
podman-compose -f docker-compose.wsl.yml restart ai-engine

# Rebuild etter kodeendring
podman-compose -f docker-compose.wsl.yml build ai-engine
podman-compose -f docker-compose.wsl.yml up -d ai-engine
```

### Debugging

```bash
# Gå inn i container
podman exec -it quantum_ai_engine bash

# Sjekk Python paths i container
podman exec quantum_ai_engine python -c "import sys; print('\n'.join(sys.path))"

# Sjekk at ingen /mnt/c paths
podman exec quantum_ai_engine python -c "
import sys
mnt_paths = [p for p in sys.path if '/mnt/c' in p]
if mnt_paths:
    print('❌ FEIL: /mnt/c paths funnet!')
    for p in mnt_paths: print(f'   {p}')
else:
    print('✅ OK: Ingen /mnt/c paths')
"

# Test imports i container
podman exec quantum_ai_engine python -c "
from backend.config.exit_mode import get_exit_mode
print(f'✅ Import fungerer: EXIT_MODE={get_exit_mode()}')
"
```

### Monitoring

```bash
# Resource usage
podman stats quantum_ai_engine

# Disk usage
podman system df

# Network info
podman network inspect quantum_trader
```

---

## 📊 6. HVORFOR DETTE FUNGERER

### I WSL

✅ **Ingen /mnt/c paths**
- Alt kjører fra `~/quantum_trader`
- Python finner moduler uten Windows filesystem overhead
- Ingen import collisions

✅ **PYTHONPATH=/app**
- Container bruker `/app` som root
- Volumes mounter relativt til `/app`
- Clean import paths: `from backend.config import ...`

✅ **Podman vs Docker**
- Rootless by default (sikrere)
- Docker API compatible
- `podman-compose` fungerer som `docker-compose`

✅ **Redis som Container**
- Isolert fra host
- Ingen port conflicts
- Enkel å restarte/resette

### På VPS (Senere)

Dette oppsettet er **identisk** til VPS deployment fordi:

✅ **Same compose file**
```bash
# WSL
podman-compose -f docker-compose.wsl.yml up -d

# VPS
docker-compose -f docker-compose.wsl.yml up -d
# Eller
podman-compose -f docker-compose.wsl.yml up -d
```

✅ **Same paths**
- Begge bruker relative paths fra prosjektrot
- Ingen hardkodede `/mnt/c` eller `C:\` paths

✅ **Same environment**
- `.env` fil fungerer likt
- Environment variables identiske
- Network setup identisk

✅ **Same dependencies**
- Dockerfile bygger likt i WSL og VPS
- Python dependencies fra pip (ikke OS-specific)

---

## 🎯 PRODUCTION WORKFLOW

### Development (WSL)

```bash
# 1. Start containere
cd ~/quantum_trader
./start-wsl.sh

# 2. Aktiver venv for backend utvikling
source .venv/bin/activate

# 3. Kjør backend lokalt (hot reload)
python -m uvicorn backend.main:app --reload

# 4. Edit kode i VS Code på Windows
#    Files sync automatisk til WSL

# 5. Test endringer
curl http://localhost:8000/health
```

### Deploy til VPS

```bash
# 1. SSH til VPS
ssh user@vps-ip

# 2. Clone/pull repo
git clone https://github.com/user/quantum_trader.git
cd quantum_trader

# 3. Kjør SAMME compose file
docker-compose -f docker-compose.wsl.yml up -d

# 4. Verifiser
docker ps
curl http://localhost:8001/health
```

**Ingen endringer nødvendig!** 🎉

---

## 🐛 TROUBLESHOOTING

### Problem: "Cannot connect to Podman"

```bash
# Start podman socket
systemctl --user start podman.socket

# Eller restart podman
sudo systemctl restart podman
```

### Problem: "Port already in use"

```bash
# Finn prosess på port
sudo lsof -i :8001

# Kill prosess
sudo kill -9 <PID>

# Eller stopp containere
podman-compose -f docker-compose.wsl.yml down
```

### Problem: "Import errors"

```bash
# Sjekk PYTHONPATH i container
podman exec quantum_ai_engine env | grep PYTHONPATH

# Skal være: PYTHONPATH=/app

# Sjekk at volumes er mounted
podman inspect quantum_ai_engine | grep -A 10 Mounts
```

### Problem: "Redis connection failed"

```bash
# Test Redis direkte
podman exec quantum_redis redis-cli ping

# Sjekk Redis logs
podman logs quantum_redis

# Test fra backend
python -c "import redis; r=redis.Redis(host='localhost', port=6379); print(r.ping())"
```

---

## 📝 OPPSUMMERING

### WSL Setup
```bash
cd ~/quantum_trader
chmod +x start-wsl.sh
./start-wsl.sh
```

### Verifiser
```bash
podman ps                                    # Sjekk containere
curl http://localhost:8001/health            # Test AI Engine
podman exec quantum_redis redis-cli ping     # Test Redis
```

### Utvikle
```bash
source .venv/bin/activate
python -m uvicorn backend.main:app --reload
```

### Deploy til VPS (Senere)
```bash
# SAMME kommandoer, samme compose file! 🚀
docker-compose -f docker-compose.wsl.yml up -d
```

---

**✅ Alt klart! Kjør `./start-wsl.sh` for å starte!**
