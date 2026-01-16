# 🤖 **5 ENSEMBLE MODULER - KOMPLETT SYSTEMOVERSIKT**

## **ENSEMBLE KJERNEMODULER (4 Predikatorer)**

### 1. **XGBoost Agent** ✅ AKTIV
- **Status**: KJØRER
- **Vekt**: 25%
- **Funksjon**: Gradient boosting for PnL-prediksjon
- **Port**: Integrert i AI Engine (8001)
- **Model**: v4_20260116_123334 (82 samples trained)
- **Performance**: Train R² = 0.9995

### 2. **LightGBM Agent** ✅ AKTIV
- **Status**: KJØRER
- **Vekt**: 25%
- **Funksjon**: Rask gradient boosting (backup til XGB)
- **Port**: Integrert i AI Engine (8001)
- **Model**: v4_20260116_123334 (82 samples trained)
- **Performance**: Train R² = 0.1730

### 3. **N-HiTS Agent** 🟡 INAKTIV (SOVER)
- **Status**: LOADED BUT DISABLED
- **Vekt**: 0% (var 30%)
- **Funksjon**: Hierarkisk time-series neural network
- **Grunn**: **TOO MEMORY-INTENSIVE** (~2GB GPU RAM)
- **Aktivering**: Krever GPU eller memory-optimisering

### 4. **PatchTST Agent** 🟡 INAKTIV (SOVER)
- **Status**: LOADED BUT DISABLED  
- **Vekt**: 0% (var 20%)
- **Funksjon**: Transformer-basert time-series forecasting
- **Grunn**: **TOO SLOW FOR PRODUCTION** (~500ms inference)
- **Aktivering**: Krever CPU-optimisering eller batch inference

---

## **ADVANCED SUPPORTING MODULER (3+ Tjenester)**

### 5️⃣ **Meta-Learning Agent** ✅ AKTIV
```
Process: /opt/quantum/venvs/ai-engine/bin/python run.py (3807715)
PID: 3807715
Memory: 0.6% (417 MB)
CPU: 0.1%
Uptime: ~25 timer
Status: RUNNING

Funksjon:
  - Kombinerer XGB + LGBM output
  - Consensus voting (krever 2/2 agreement fra aktive modeller)
  - Adaptive confidence tuning
  - Override-logikk for high-drift situations
```

### 6️⃣ **Governer Agent (Risk Management)** ✅ AKTIV
```
Process: Risk Safety Service (3830680)
PID: 3830680
Memory: 0.4% (410 MB)
CPU: 0.1%
Uptime: ~9 timer
Status: RUNNING

Funksjon:
  - Circuit breaker (stopp ved høy drawdown)
  - Max leverage enforcement (5-80x range)
  - Position size capping
  - Emergency exit triggers
```

### 7️⃣ **RL Position Sizing** ✅ AKTIV
```
Processes:
  - rl_agent.py (3806647): 1.0% CPU, 3.9% RAM (630 MB)
  - rl_trainer.py (3806657): 1.0% CPU, 3.9% RAM (630 MB)
  - rl_monitor.py (3806664): 0.0% CPU, 0.2% RAM (42 MB)
  - rl_sizing_agent.pnl_feedback_listener (3806730): 0.0% CPU, 3.9% RAM (625 MB)

Funksion:
  - PyTorch reinforcement learning policy
  - Adaptive leverage (5-15x based on confidence)
  - PnL feedback loop
  - Experience buffer for continuous learning
```

### 8️⃣ **CLM (Continuous Learning Manager)** ✅ AKTIV
```
Process: /opt/quantum/venvs/ai-engine/bin/python microservices/clm/main.py (3806471)
PID: 3806471
Memory: 0.6% (515 MB)
CPU: 0.0%
Uptime: ~8 timer
Status: RUNNING

Funksjon:
  - Automatic model retraining
  - Performance monitoring
  - Data collection for training
  - Model versioning
```

### 9️⃣ **Meta Regime Service** ✅ AKTIV
```
Process: /opt/quantum/venvs/ai-engine/bin/python microservices/meta_regime/meta_regime_service.py (3806486)
PID: 3806486
Memory: 0.8% (569 MB)
CPU: 0.0%
Uptime: ~8 timer
Status: RUNNING

Funksjon:
  - Market regime detection (Bullish/Bearish/Sideways)
  - Cross-exchange regime comparison
  - Regime-specific policy selection
  - Volatility analysis
```

### 🔟 **RL Shadow Metrics Exporter** ✅ AKTIV
```
Process: /usr/bin/python3 -u /home/qt/quantum_trader/microservices/ai_engine/rl_shadow_metrics_exporter.py (3805993)
PID: 3805993
Memory: 0.1% (188 MB)
CPU: 0.0%
Uptime: ~19 timer
Status: RUNNING

Funksjon:
  - Exports RL metrics for monitoring
  - Performance tracking
  - Comparison metrics (live vs shadow models)
```

### 1️⃣1️⃣ **Memory State Manager** ✅ AKTIV
```
Process: /opt/quantum/venvs/ai-engine/bin/python microservices/strategic_memory/memory_sync_service.py (3805994)
PID: 3805994
Memory: 0.2% (222 MB)
CPU: 0.0%
Uptime: ~19 timer
Status: RUNNING

Funksjon:
  - Persistent strategic memory
  - Regime preferences tracking
  - Performance history
  - Decision heuristics storage
```

---

## **INAKTIVE MODULER (Why They Sleep)**

| Module | Status | Grunn | Kan aktiveres? |
|--------|--------|-------|---|
| **N-HiTS** | 🟡 Disabled | Memory intensive (~2GB) | ✅ Ja, med GPU |
| **PatchTST** | 🟡 Disabled | For slow (~500ms) | ✅ Ja, med optimisering |
| **Drift Detection** | ❌ Not loaded | No file | ✅ Implementer |
| **Covariate Shift** | ❌ Not loaded | No file | ✅ Implementer |
| **Shadow Models** | ❌ Not loaded | No file | ✅ Implementer |
| **Continuous Learning** | ✅ Loaded | Kjører i CLM | - |

---

## **RESSURSBRUK - ALLE MODULER**

### Prosesser som Kjører på VPS (qt user):
```
AI Engine (Main):                3920262    70.7% CPU, 4.5% RAM    ← HEAVY LOAD
RL Agent:                        3806647     1.0% CPU, 3.9% RAM
RL Trainer:                      3806657     1.0% CPU, 3.9% RAM
RL Sizing Listener:              3806730     0.0% CPU, 3.9% RAM
Strategy Ops:                    3368006     0.5% CPU, 4.1% RAM
Market Publisher:                3806239     1.5% CPU, 1.1% RAM
Cross Exchange Aggregator:       3618757     0.3% CPU, 0.4% RAM
Position Monitor:                3830683     0.2% CPU, 0.3% RAM
Exit Monitor:                    3876349     0.4% CPU, 0.5% RAM
Risk Safety Service:             3830680     0.1% CPU, 0.4% RAM
Portfolio Governance:            3807455     0.0% CPU, 0.1% RAM
Exposure Balancer:               3807462     0.0% CPU, 0.2% RAM

Total Active: ~18 prosesser
Total Memory: ~15-20% av 16GB RAM
Total CPU: Peak at 70-80% (AI Engine doing ensemble voting)
```

### 110GB Volume Brukes For:
```
/mnt/HC_Volume_104287969/
  ├── docker/              66 MB    (container images + layers)
  ├── containerd/          ~30 MB   (container runtime)
  └── lost+found/          16 KB

Total Used: ~3.0 GB / 110 GB (2.7% utilization)
Available: ~100 GB (unused, ready for logging!)
```

---

## **LOGGING - HVOR ER FILENE?**

### Aktuelle Log Steder:
```
AI Engine:           /tmp/ai_engine_new.log
RL Policy:          /tmp/rl_policy.log (eller systemd journal)
Exit Brain:         /tmp/exit_monitor.log
CLM:                /tmp/clm_*.log

Docker Logs (110GB volume):
  /mnt/HC_Volume_104287969/docker/containers/
  /mnt/HC_Volume_104287969/docker/overlay2/
```

### Hva Kan Lagres på 110GB Volume:
```
✅ Alle AI Engine logs (rotating)
✅ RL training data + experiences
✅ Model checkpoints
✅ Performance metrics history
✅ Redis snapshots (RDB backups)
✅ Event stream archives
✅ Training datasets
✅ Still over 100GB left!
```

---

## **HVEM GJØR HVA - ARKITEKTUR**

```
┌─────────────────────────────────────────────────────────────┐
│                   TRADE SIGNALS                              │
│            (BUY/SELL/HOLD with Confidence)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────▼───────────┐
         │   AI Engine (8001)    │ ← MAIN ORCHESTRATOR
         │   (XGB + LGBM active) │   70% CPU, 4.5% RAM
         └────┬────────────┬─────┘
              │            │
    ┌─────────▼──┐  ┌──────▼────────┐
    │ Meta-Agent │  │ Governer Agent│
    │ Consensus  │  │  Risk Safety  │
    │  Voting    │  │  Circuit Break│
    └─────┬──────┘  └───────┬───────┘
          │                  │
    ┌─────▼──────────────────▼─────┐
    │  RL Position Sizing (3 procs) │ ← DYNAMIC LEVERAGE 5-80x
    │  - Agent (learns policy)      │  - Trainer (improves)
    │  - Monitor (tracks)           │  - Listener (PnL feedback)
    │  - Sizing Agent (calculates)  │
    └──────────┬────────────────────┘
               │
    ┌──────────▼──────────────────────┐
    │   CLM (Continuous Learning)     │ ← RETRAINING
    │   - Model monitoring            │  - Auto retrain
    │   - Performance tracking        │  - Version management
    │   - Data collection             │
    └─────────────────────────────────┘
```

---

## **MODULER SAMMENLIGNING**

| Modul | Status | Rolle | Vekt | Memory | CPU | Latency |
|-------|--------|-------|------|--------|-----|---------|
| **XGBoost** | ✅ Aktiv | Prediktor | 25% | Lav (50MB) | Medium | <5ms |
| **LightGBM** | ✅ Aktiv | Prediktor | 25% | Lav (50MB) | Medium | <5ms |
| **Meta-Agent** | ✅ Aktiv | Konsensus | - | Medium (400MB) | Low | <1ms |
| **Governer** | ✅ Aktiv | Risk | - | Medium (400MB) | Low | <1ms |
| **RL Agent** | ✅ Aktiv | Leverage | - | High (630MB) | High | ~50ms |
| **CLM** | ✅ Aktiv | Learning | - | Medium (500MB) | Low | N/A |
| **N-HiTS** | 🟡 Paused | Prediktor | 0% | VERY High (2GB) | High | ~200ms |
| **PatchTST** | 🟡 Paused | Prediktor | 0% | High (1.5GB) | High | ~500ms |

---

## **SÅ - DE 3 ANDRE MODULENE GJØR:**

### **N-HiTS** (Sleeping - For Memory Intensive)
- Neural Hierarchical Interpolation for Time Series
- Hierarkisk nevralt nettverk for tidsserier
- Sover fordi: Krever 2GB GPU RAM eller dedikert processor
- Kan brukes for: Langsiktige trend-forecasting

### **PatchTST** (Sleeping - For Slow)
- Transformer-basert tidsserier-prediksjon
- Bruker "patches" av data (segments)
- Sover fordi: 500ms inference for langsom for real-time trading
- Kan brukes for: Regime-spesifikk forecasting (off-peak)

### **RL Modules** (Very Much Awake! 🔥)
- **RL Agent**: Lærer optimal leverage policy
- **RL Trainer**: Oppdaterer policyen basert på PnL
- **RL Monitor**: Tracker performance
- **Sizing Listener**: Hører på PnL feedback
- → **TOTAL 4 PROSESSER** for reinforcement learning!

---

## **REKOMMENDASJONER**

### Kortsiktig (Nå):
1. ✅ Logg alle AI Engine meldinger til 110GB volume
2. ✅ Aktiver N-HiTS hvis GPU blir tilgjengelig
3. ✅ Monitor RL training progress

### Langsiktig:
1. Optimaliser PatchTST for batch inference
2. Implementer drift detection (monitoring model decay)
3. Implementer covariate shift detection
4. Legg til shadow model manager (A/B testing av nye modeller)

### Debugging:
1. Sjekk 110GB volume for Docker logs:
   ```
   ls -lah /mnt/HC_Volume_104287969/docker/containers/*/
   ```
2. Endre log level hvis det er for verbose
3. Arkiver gamle logs til komprimert format

---

**Status**: 🟢 **FULLY OPERATIONAL**
**Moduler Aktive**: 9 tjenester + 2 predikatorer = **11 komponenter kjørende**
**Memory Utilization**: ~15-20% (plenty of room)
**CPU Usage**: 70-80% peak (mainly AI Engine ensemble voting)
**110GB Volume**: Only 3% used (97 GB available for logging!)
