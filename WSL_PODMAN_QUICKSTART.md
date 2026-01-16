# 🚀 Quantum Trader - WSL + Podman-Compose Quick Start

Kjør Quantum Trader i WSL med podman-compose for containerisert deployment.

---

## 📋 Forutsetninger

1. **WSL2 installert** på Windows
2. **Ubuntu** (anbefalt) eller annen Linux-distribusjon i WSL
3. **Quantum Trader** klonet til `C:\quantum_trader`

---

## 🛠️ Installasjon

### Trinn 1: Åpne WSL Terminal

```powershell
# Fra PowerShell/Windows Terminal
wsl
```

### Trinn 2: Installer Podman og Podman-Compose

```bash
# Oppdater pakkelisten
sudo apt-get update

# Installer podman og podman-compose
sudo apt-get install -y podman podman-compose

# Verifiser installasjonen
podman --version
podman-compose --version
```

### Trinn 3: Naviger til Prosjektet

```bash
# Naviger til Windows-mappen fra WSL
cd /mnt/c/quantum_trader

# Verifiser at du er i riktig mappe
ls -la
# Du skal se: systemctl.yml, backend/, ai_engine/, etc.
```

---

## 🚀 Start Quantum Trader

### Metode 1: Dev Profil (Anbefalt for Testing)

```bash
# Bygg containere
podman-compose build

# Start backend i dev-modus (Binance TESTNET)
podman-compose --profile dev up -d

# Se logger
podman-compose logs -f backend
```

### Metode 2: Live Profil (Produksjon)

```bash
# Start backend i live-modus (Binance MAINNET)
podman-compose --profile live up -d

# Se logger
podman-compose logs -f backend-live
```

### Metode 3: Med Strategy Generator

```bash
# Start backend + strategy generator
podman-compose --profile dev --profile strategy-gen up -d

# Se alle logger
podman-compose logs -f
```

---

## 📊 Nyttige Kommandoer

### Se Status

```bash
# Vis kjørende containere
podman-compose ps

# Se container detaljer
podman ps -a
```

### Se Logs

```bash
# Alle containere
podman-compose logs -f

# Kun backend
podman-compose logs -f backend

# Siste 100 linjer
podman-compose logs --tail=100 backend
```

### Stopp Containere

```bash
# Stopp alle containere
podman-compose down

# Stopp og fjern volumes
podman-compose down -v

# Stopp kun en container
podman-compose stop backend
```

### Restart Containere

```bash
# Restart alt
podman-compose restart

# Restart kun backend
podman-compose restart backend
```

### Rebuild Containere

```bash
# Rebuild etter kodeendringer
podman-compose build --no-cache

# Rebuild og restart
podman-compose up -d --build
```

---

## 🔍 Tilgang til Applikasjonen

Når containeren kjører, er backend tilgjengelig på:

- **API**: http://localhost:8000
- **Health Check**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs

Du kan teste fra **Windows** (ikke i WSL):

```powershell
# Fra PowerShell
Invoke-WebRequest http://localhost:8000/health
```

Eller fra **WSL**:

```bash
curl http://localhost:8000/health
```

---

## 🐛 Debugging

### Se Container Logs Live

```bash
# Follow logs i sanntid
podman-compose logs -f backend | grep -E "ERROR|WARNING|EXIT_BRAIN"
```

### Gå Inn i Container

```bash
# Åpne bash shell i backend container
podman exec -it quantum_backend bash

# Eller bruk podman-compose
podman-compose exec backend bash

# Sjekk filer, kjør Python-kommandoer, etc.
```

### Kjør Python-kommandoer i Container

```bash
# Sjekk EXIT_MODE konfigurasjon
podman exec quantum_backend python -c "from backend.config.exit_mode import get_exit_mode, is_challenge_100_profile; print(f'EXIT_MODE: {get_exit_mode()}'); print(f'CHALLENGE_100: {is_challenge_100_profile()}')"

# Sjekk database tabeller
podman exec quantum_backend python -c "from backend.database import engine; from sqlalchemy import inspect; print(inspect(engine).get_table_names())"
```

### Sjekk Container Resources

```bash
# CPU og minne bruk
podman stats quantum_backend

# Disk bruk
podman system df
```

---

## 🔧 Konfigurasjon

### Environment Variables

Hovedkonfigurasjonen er i `.env` filen i prosjektroten:

```bash
# Fra WSL
cd /mnt/c/quantum_trader
nano .env

# Viktige variabler:
# EXIT_MODE=EXIT_BRAIN_V3
# EXIT_EXECUTOR_MODE=LIVE
# EXIT_BRAIN_PROFILE=CHALLENGE_100
# BINANCE_TESTNET=true
```

### Docker Compose Profiler

- **`dev`**: Testnet trading med utviklingsinnstillinger
- **`live`**: Mainnet trading med produksjonsinnstillinger
- **`strategy-gen`**: AI strategy generator (kontinuerlig evolusjon)

---

## 📦 Volumes og Data Persistence

Podman-compose monterer følgende mapper fra Windows til containeren:

```yaml
volumes:
  - ./backend:/app/backend       # Backend kode (live reload)
  - ./ai_engine:/app/ai_engine   # AI modeller
  - ./models:/app/models          # Trente modeller
  - ./database:/app/database      # SQLite database
```

Data lagres på Windows-filsystemet, så den overlever container-restarts.

---

## 🔄 Workflow for Utvikling

### 1. Gjør Kodeendringer (Windows)

Rediger filer i VS Code på Windows som normalt:

```
C:\quantum_trader\backend\...
```

### 2. Rebuild og Restart (WSL)

```bash
# Gå til WSL
cd /mnt/c/quantum_trader

# Rebuild container
podman-compose build backend

# Restart med ny kode
podman-compose up -d backend

# Se nye logs
podman-compose logs -f backend
```

### 3. Test Endringer

```bash
# Kjør verification
podman exec quantum_backend python verify_challenge_100_hotfix.py

# Eller fra Windows
wsl -e podman exec quantum_backend python verify_challenge_100_hotfix.py
```

---

## 🚨 Troubleshooting

### Problem: "Cannot connect to Podman socket"

```bash
# Start podman socket
systemctl --user enable --now podman.socket

# Eller restart podman service
sudo systemctl restart podman
```

### Problem: "Permission denied"

```bash
# Legg bruker til podman gruppe
sudo usermod -aG podman $USER

# Log ut og inn igjen for at gruppe-endringen skal tre i kraft
```

### Problem: "Port 8000 already in use"

```bash
# Finn prosess som bruker port 8000
sudo lsof -i :8000

# Eller stop eksisterende backend
podman stop quantum_backend

# Eller fra Windows PowerShell:
Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
```

### Problem: "Build failed"

```bash
# Rens podman cache
podman system prune -a -f

# Rebuild fra scratch
podman-compose build --no-cache
```

---

## 🎯 Produksjonskjøring

For produksjon, bruk `live` profilen med `.env.live`:

```bash
# Stopp dev-containere
podman-compose --profile dev down

# Start live-containere
podman-compose --profile live up -d

# Monitor produksjon
podman-compose logs -f backend-live | grep -E "CHALLENGE_100|EXIT_BRAIN|HARD_SL"
```

---

## 📚 Mer Informasjon

- **CHALLENGE_100 Hotfix**: Se `CHALLENGE_100_HOTFIX_COMPLETE.md`
- **Exit Brain v3**: Se `AI_EXIT_BRAIN_V3_GUIDE.md`
- **API Dokumentasjon**: http://localhost:8000/docs (når backend kjører)

---

**Status**: ✅ Ready for WSL + Podman-Compose Deployment

_Laget: 2025-12-14_

