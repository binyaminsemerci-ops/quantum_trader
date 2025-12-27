# 🔄 SERVER RESTART - QUICK GUIDE

## 🎯 Hvorfor Restart?

Serveren må restartes for å aktivere:
- ✅ JWT Authentication (`backend/auth.py`)
- ✅ Redis Caching (`backend/cache.py`)
- ✅ Security Headers (`backend/https_config.py`)

Koden er integrert i `backend/main.py`, men serveren kjører fortsatt gammel kode fra minnet.

---

## 🚀 Metode 1: Automatisk Restart (Anbefalt)

```powershell
# Kjør restart script
.\scripts\restart_server.ps1

# Følg instruksjonene på skjermen
# Script vil:
# 1. Finne kjørende server (PID 17448)
# 2. Spørre om du vil force restart
# 3. Stoppe gammel server
# 4. Starte ny server med --reload
```

---

## 🔧 Metode 2: Manuell Restart

### Steg 1: Finn terminal med uvicorn

Serveren kjører sannsynligvis i en annen terminal/VS Code terminal.

### Steg 2: Stopp serveren

I den terminalen som kjører uvicorn:
```
Press: Ctrl + C
```

Vent til du ser:
```
Shutting down
```

### Steg 3: Start serveren på nytt

```powershell
uvicorn backend.main:app --reload
```

---

## 🔍 Metode 3: Force Kill og Restart (Siste utvei)

```powershell
# Stopp serveren (force)
Stop-Process -Id 17448 -Force

# Vent litt
Start-Sleep -Seconds 3

# Start på nytt
uvicorn backend.main:app --reload
```

---

## ✅ Verifiser at det virker

### Sjekk Logger

Når serveren starter, se etter disse meldingene:

```
✅ [SEARCH] Initializing Authentication System...
✅ [OK] Authentication system initialized (JWT + Redis)

✅ [SEARCH] Initializing Caching Layer...
✅ [OK] Caching layer initialized (Redis + pooling for P99 optimization)
```

### Test API Dokumentasjon

1. Åpne: http://localhost:8000/api/docs
2. Se etter nye endepunkter:
   - POST `/api/auth/login`
   - POST `/api/auth/refresh`
   - POST `/api/auth/logout`

### Test Login

I Swagger UI (http://localhost:8000/api/docs):

1. Finn `POST /api/auth/login`
2. Klikk "Try it out"
3. Skriv inn:
   ```json
   {
     "username": "admin",
     "password": "admin123"
   }
   ```
4. Klikk "Execute"
5. Du skal få tilbake en `access_token`

### Test Beskyttede Endepunkter

1. Kopier `access_token` fra login response
2. Klikk "Authorize" knappen (lås-ikon øverst i Swagger)
3. Skriv: `Bearer <din-token-her>`
4. Klikk "Authorize"
5. Prøv `/api/dashboard/trading` - skal virke nå!

---

## 🧪 Kjør Tester

Etter restart, kjør disse for å verifisere forbedringer:

```powershell
# Integrasjonstester (auth + cache + security headers)
python scripts/test_integration.py

# Sikkerhetsaudit (skal gå fra 62.5% til ~87.5%)
python scripts/test_security.py

# Performance (P99 skal forbedres dramatisk)
python scripts/test_performance.py
```

---

## 📊 Forventede Forbedringer

### Security Audit

| Test | Før Restart | Etter Restart |
|------|-------------|---------------|
| HTTPS Usage | ❌ FAIL | ❌ FAIL (trenger SSL cert) |
| **Authentication** | **❌ FAIL** | **✅ PASS** ✅ |
| **Rate Limiting** | **❌ FAIL** | **✅ PASS** ✅ |
| SQL Injection | ✅ PASS | ✅ PASS |
| XSS Protection | ✅ PASS | ✅ PASS |
| **Score** | **62.5%** | **~87.5%** ✅ |

### Performance

| Endpoint | P99 Før | P99 Etter (Forventet) |
|----------|---------|------------------------|
| Dashboard Trading | 16.3s ❌ | <1s ✅ |
| AI Signals | 1.1s | <500ms ✅ |
| Cache Speedup | 0.86x | 10-35x ✅ |

---

## 🚨 Feilsøking

### Problem: "Address already in use"

```powershell
# Port 8000 er fortsatt i bruk
# Finn prosessen:
netstat -ano | findstr :8000

# Stopp den:
Stop-Process -Id <PID> -Force
```

### Problem: "Redis connection failed"

```powershell
# Sjekk at Redis kjører:
docker ps | Select-String redis

# Start Redis hvis den ikke kjører:
docker start quantum_redis
```

### Problem: "Module not found: backend.auth"

```powershell
# Sjekk at filene eksisterer:
Test-Path backend/auth.py
Test-Path backend/cache.py
Test-Path backend/https_config.py

# Hvis alle er True, restart skal virke
```

### Problem: Ingen nye endepunkter i /api/docs

```
Dette betyr at serveren ikke lastet ny kode.
Prøv:
1. Force kill prosessen
2. Clear Python cache: Remove-Item -Recurse __pycache__, backend/__pycache__
3. Start på nytt
```

---

## 📝 Quick Checklist

Før restart:
- [x] Kode integrert i main.py
- [x] Dependencies installert
- [x] Redis kjører
- [x] Environment konfigurert

Etter restart:
- [ ] Server starter uten errors
- [ ] Auth endpoints vises i /api/docs
- [ ] Logger viser "Auth system initialized"
- [ ] Logger viser "Caching layer initialized"
- [ ] Login virker (admin/admin123)
- [ ] Beskyttede endepunkter krever token
- [ ] X-Cache headers vises i responses

Validering:
- [ ] `test_integration.py` - PASS
- [ ] `test_security.py` - 87.5%+ PASS
- [ ] `test_performance.py` - P99 < 1s

---

## 🎉 Suksess!

Når alt er verifisert:

```powershell
# Generer final rapport
python scripts/generate_final_report.py

# Dokumentasjonen er allerede klar:
# - IMPLEMENTATION_COMPLETE.md
# - PRODUCTION_READINESS_IMPLEMENTATION.md
# - FINAL_QA_REPORT_UPDATED.md
```

---

**Status:** Klar for restart! 🚀

**Anbefalt:** Kjør `.\scripts\restart_server.ps1` for automatisk restart.
