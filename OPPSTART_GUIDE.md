# QUANTUM TRADER - Oppstartsinstruksjoner

## 🚀 Rask Start (Alt-i-ett)

```powershell
.\startup_all.ps1
```

Dette starter:
1. ✅ Docker containers (hvis tilgjengelig)
2. ✅ Backend API (port 8000)
3. ✅ Frontend Dashboard (port 3000)
4. ✅ Continuous Learning Manager (CML)
5. ✅ AI Model Training (30-40 min)

---

## 📋 Steg-for-steg Manuell Oppstart

### 1. Docker (Valgfritt)
```powershell
systemctl --profile dev up -d --build
```

### 2. Backend Server
```powershell
# Alternativ A: Med task (anbefalt)
# Bruk VS Code task: "Start Backend (Terminal)"

# Alternativ B: Manuelt
.\scripts\start-backend.ps1
```

Backend kjører på: `http://localhost:8000`  
Health check: `http://localhost:8000/health`

### 3. Frontend Dashboard
```powershell
cd frontend
npm run dev
```

Frontend kjører på: `http://localhost:3000`

### 4. Aktiver CML (Continuous Learning)
```powershell
python activate_retraining_system.py
```

### 5. Tren AI Modeller
```powershell
# Alle modeller (anbefalt, 30-40 min)
python scripts/train_all_models.py

# Kun en modell (raskere)
python scripts/train_xgboost_quick.py      # ~2-3 min
python scripts/train_lightgbm.py           # ~2-3 min
python scripts/train_nhits.py              # ~10-15 min
python scripts/train_patchtst.py           # ~15-20 min
```

---

## ⚙️ Startup Script Opsjoner

### Full oppstart (standard)
```powershell
.\startup_all.ps1
```

### Hopp over Docker
```powershell
.\startup_all.ps1 -SkipDocker
```

### Hopp over AI training
```powershell
.\startup_all.ps1 -SkipTraining
```

### Kun trening (ikke backend/frontend)
```powershell
.\startup_all.ps1 -TrainingOnly
```

---

## 🔍 Verifisering

### Sjekk Backend
```powershell
Invoke-WebRequest http://localhost:8000/health
```

### Sjekk Frontend
```powershell
Invoke-WebRequest http://localhost:3000
```

### Sjekk Docker Containers
```powershell
systemctl list-units
```

### Sjekk CML Status
```powershell
python check_learning_status.py
```

---

## 📊 Systemkomponenter

| Komponent | Port | Status |
|-----------|------|--------|
| Backend API | 8000 | ✅ Kjører |
| Frontend Dashboard | 3000 | ✅ Kjører |
| Continuous Learning | - | ✅ Aktivert |
| AI Training | - | 🔄 I gang |

---

## 🤖 AI Modeller

Systemet bruker 4-modell ensemble:

1. **XGBoost** (25% vekt)
   - Raskest: 2-3 minutter
   - Tre-basert gradient boosting

2. **LightGBM** (25% vekt)
   - Rask: 2-3 minutter
   - Optimalisert gradient boosting

3. **N-HiTS** (30% vekt)
   - Medium: 10-15 minutter
   - Multi-rate temporal neural network

4. **PatchTST** (20% vekt)
   - Sakte: 15-20 minutter
   - Transformer-basert tidsserie

**Total treingstid:** ~30-40 minutter

---

## 🔄 Continuous Learning Cycle

```
1. 📊 AI predicts → Trade execution
2. 💰 Position closes → Outcome recorded
3. 💾 Training data collected
4. 🔄 Retraining triggered (daglig/performance)
5. 🧠 New model trained → Deployment evaluation
6. ✅ Deploy if better → Better predictions!
7. 🔁 Loop continues forever...
```

### CML Triggere:
- ⏰ **TIME-DRIVEN:** Daglig retraining på schedule
- 📉 **PERFORMANCE-DRIVEN:** Hvis win rate < 50%
- 🌊 **REGIME-DRIVEN:** Ved endring i market regime
- 📊 **DRIFT-DETECTED:** Ved model drift detection

---

## 🛠️ Feilsøking

### Backend starter ikke
```powershell
# Sjekk om port 8000 er i bruk
netstat -ano | findstr :8000

# Sjekk logs
Get-Content backend_startup.log -Tail 50 -Wait
```

### Frontend starter ikke
```powershell
# Reinstaller dependencies
cd frontend
npm install
npm run dev
```

### Training feiler
```powershell
# Sjekk om training data finnes
Test-Path data\binance_training_data_full.csv

# Fetch ny data
python scripts/fetch_all_data.py
```

### Docker problemer
```powershell
# Restart Docker
systemctl down
systemctl --profile dev up -d --build
```

---

## 📞 Support

- **Logs:** `backend_startup.log`, `training_output.log`
- **Config:** `.env` (Binance testnet credentials)
- **Data:** `data/` directory
- **Models:** `models/` directory

---

## ⚡ Quick Reference

### Start Alt
```powershell
.\startup_all.ps1
```

### Stopp Alt
```powershell
# Stopp Docker
systemctl down

# Stopp backend (Ctrl+C i terminal eller lukk vindu)
# Stopp frontend (Ctrl+C i terminal eller lukk vindu)
```

### Status Check
```powershell
# Backend
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Docker
systemctl list-units
```

---

**Sist oppdatert:** 2025-12-11  
**Versjon:** 1.0

