# 🚀 QUANTUM TRADER - WSL + PODMAN QUICK REFERENCE

## 📋 1. ANALYSE - Hva kjører vi?

### Services i containers:
- **Redis** (port 6379) - EventBus & Cache
- **AI-Engine** (port 8001) - ML inference microservice

### Services i WSL venv:
- **Backend** (port 8000) - Main orchestrator (kjøres manuelt)

---

## ⚡ 2. EKSAKTE KOMMANDOER (WSL)

### Start services

```bash
# 1. Naviger til WSL directory (IKKE /mnt/c)
cd ~/quantum_trader

# 2. Verifiser path
pwd
# Output: /home/<user>/quantum_trader

# 3. Start Redis + AI-Engine
podman-compose -f systemctl.wsl.yml up -d redis ai-engine

# 4. Sjekk status
podman-compose -f systemctl.wsl.yml ps
```

---

## ✅ 3. VERIFY

### A) Container status
```bash
podman ps
# Skal vise: quantum_redis, quantum_ai_engine
```

### B) Redis
```bash
podman logs quantum_redis | tail -20
podman exec quantum_redis redis-cli ping
# Output: PONG
```

### C) AI-Engine
```bash
podman logs -f quantum_ai_engine
# CTRL+C for å stoppe

curl http://localhost:8001/health
# Output: {"status":"healthy", ...}
```

### D) Import check (ingen /mnt/c)
```bash
podman exec quantum_ai_engine python -c "import sys; print('\\n'.join(sys.path))"
# Skal IKKE inneholde /mnt/c
```

### E) ServiceHealth.create() test
```bash
source .venv/bin/activate
python -c "from backend.infra.service_health import ServiceHealth; h = ServiceHealth.create(service_name='test'); print(f'✅ OK: {h.service_name}')"
```

---

## 🛑 4. STOP / RESTART

### Stop services
```bash
cd ~/quantum_trader
podman-compose -f systemctl.wsl.yml down
```

### Restart AI-Engine
```bash
podman-compose -f systemctl.wsl.yml restart ai-engine
```

### Full cleanup (med volumes)
```bash
podman-compose -f systemctl.wsl.yml down -v
```

---

## 📊 5. MONITORING

### Logs (live follow)
```bash
podman logs -f quantum_ai_engine
podman logs -f quantum_redis
```

### Resource usage
```bash
podman stats
```

### Redis inspect
```bash
podman exec quantum_redis redis-cli INFO
```

---

## 🔍 6. DEBUGGING

### Container starter ikke?
```bash
# Se full build log
podman logs quantum_ai_engine

# Inspiser config
podman inspect quantum_ai_engine | less

# Gå inn i container
podman exec -it quantum_ai_engine bash
```

### PYTHONPATH sjekk
```bash
podman exec quantum_ai_engine printenv PYTHONPATH
# Output: /app
```

### Import test
```bash
podman exec quantum_ai_engine python -c "from microservices.ai_engine.main import app; print('✅ Import OK')"
```

---

## 🎯 7. FORKLARING

### Hvorfor fungerer dette i WSL?

1. **Native Linux paths**: `~/quantum_trader` er ekte Linux, ikke NTFS
2. **Ingen /mnt/c**: Unngår import-collisions
3. **PYTHONPATH=/app**: Eksplisitt, ikke implisitt
4. **Podman rootless**: Ingen Docker Desktop overhead

### Hvorfor fungerer det på VPS?

**Identisk oppsett:**
- VPS er Linux native (som WSL)
- Samme `systemctl.wsl.yml`
- Samme PYTHONPATH-logikk
- Samme mount structure

**Eneste forskjell:**
- VPS bruker kanskje `systemctl` i stedet for `podman-compose` (syntax 100% lik)
- VPS kan bruke systemd for auto-start

---

## ⚠️ VIKTIGE REGLER

### ❌ ALDRI:
- Bruk `/mnt/c/quantum_trader` i runtime
- Start fra Windows PowerShell
- Mount Windows paths i containers
- Hardkode absolute paths

### ✅ ALLTID:
- Kjør fra `~/quantum_trader` i WSL
- Bruk relative paths (`.`)
- Sett `PYTHONPATH=/app` eksplisitt
- Test `/health` endpoints etter start

---

## 🎓 8. NESTE STEG

### Når Redis + AI-Engine kjører:

1. **Start Backend** (i WSL venv):
   ```bash
   source .venv/bin/activate
   python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. **Test full integration**:
   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8001/health
   ```

3. **Kjør orchestrator**:
   ```bash
   python -m backend.orchestrator.main
   ```

---

## 📦 SUCCESS CRITERIA

System er ready når:

- [x] `podman ps` viser 2 containers
- [x] `curl localhost:6379` (Redis)
- [x] `curl localhost:8001/health` → 200 OK
- [x] AI-Engine logs: "Application startup complete"
- [x] Ingen `/mnt/c` i sys.path
- [x] ServiceHealth.create() fungerer

**🚀 Da er du klar for trading!**

