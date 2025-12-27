# Phase 4S+ Deployment Guide

## 📋 Available Deployment Scripts

### 🖥️ **VPS Remote Deployment** (Recommended for Production)
**Scripts:** 
- `scripts/deploy_phase4s.ps1` (PowerShell - 15 steps)
- `scripts/deploy_phase4s.sh` (Bash - 14 steps)

**Use when:** Deploying from your Windows machine to VPS via SSH

**Features:**
- ✅ Full remote SSH deployment
- ✅ Comprehensive 15-step validation
- ✅ Test data injection
- ✅ 60-second processing wait
- ✅ AI Engine health verification
- ✅ Governance integration checks

**Run:**
```powershell
# Windows PowerShell
.\scripts\deploy_phase4s.ps1
```

```bash
# Local bash (via WSL)
./scripts/deploy_phase4s.sh
```

---

### 🐧 **VPS Local Deployment** (Run directly on VPS)
**Script:** `scripts/deploy_phase4s_vps_local.sh`

**Use when:** Already SSH'd into VPS and want to deploy locally

**Features:**
- ✅ Runs directly on VPS (no SSH overhead)
- ✅ 12-step simplified deployment
- ✅ Test data injection
- ✅ Full validation cycle
- ✅ jq support for JSON parsing

**Setup:**
```bash
# On VPS
cd /home/qt/quantum_trader

# Upload script
scp -i ~/.ssh/hetzner_fresh scripts/deploy_phase4s_vps_local.sh qt@46.224.116.254:/home/qt/quantum_trader/

# Make executable
chmod +x deploy_phase4s_vps_local.sh

# Run
./deploy_phase4s_vps_local.sh
```

---

### 💻 **Local Docker Deployment** (Development only)
**Script:** `scripts/deploy_phase4s_local.ps1`

**Use when:** Testing locally with Docker Desktop on Windows

**Features:**
- ✅ Local Docker environment
- ✅ 12-step validation
- ✅ No SSH required
- ✅ Immediate feedback

**Run:**
```powershell
.\scripts\deploy_phase4s_local.ps1
```

---

## 🔧 Script Comparison

| Feature | Remote (PS1) | Remote (SH) | VPS Local | Local Docker |
|---------|--------------|-------------|-----------|--------------|
| **Steps** | 15 | 14 | 12 | 12 |
| **SSH Required** | ✅ | ✅ | ❌ | ❌ |
| **Git Pull** | ✅ | ✅ | ✅ | ✅ |
| **Test Data** | ✅ | ✅ | ✅ | ✅ |
| **60s Wait** | ✅ | ✅ | ✅ | ✅ |
| **AI Health** | ✅ | ✅ | ✅ | ✅ |
| **Feedback Check** | ✅ | ✅ | ✅ | ✅ |
| **Governance** | ✅ | ✅ | ✅ | ✅ |
| **Watch Commands** | ✅ | ✅ | ✅ | ✅ |

---

## 📊 Monitoring Tools

### 🔁 **Continuous Feedback Monitor**
```powershell
.\scripts\watch_feedback_loop.ps1
```
- Real-time feedback display (15s refresh)
- Policy recommendations
- Regime performance metrics
- Change alerts

### 🔍 **Integration Verification**
```powershell
.\scripts\verify_phase4s_integration.ps1
```
- 8 comprehensive tests
- Container health
- Redis connectivity
- AI Engine integration
- Governance linkage

---

## 🎯 Recommended Workflow

### For Production VPS:
```powershell
# 1. Deploy with full validation
.\scripts\deploy_phase4s.ps1

# 2. Verify all integrations
.\scripts\verify_phase4s_integration.ps1

# 3. Monitor live feedback
.\scripts\watch_feedback_loop.ps1
```

### For VPS Direct Access:
```bash
# SSH to VPS
ssh -i ~/.ssh/hetzner_fresh qt@46.224.116.254

# Deploy locally
./deploy_phase4s_vps_local.sh

# Monitor
watch -n 15 "docker exec quantum_redis redis-cli GET quantum:feedback:strategic_memory"
```

### For Local Development:
```powershell
# Deploy to local Docker
.\scripts\deploy_phase4s_local.ps1

# Check status
docker ps | Select-String strategic_memory
docker logs -f quantum_strategic_memory
```

---

## 🧪 Test Data Injection

All scripts inject 3 synthetic regime observations:
```bash
XADD quantum:stream:meta.regime * regime BULL pnl 0.42
XADD quantum:stream:meta.regime * regime BEAR pnl -0.18
XADD quantum:stream:meta.regime * regime RANGE pnl 0.12
```

This triggers immediate analysis and feedback generation.

---

## 📈 Expected Results

After successful deployment:

### Redis Feedback Key:
```json
{
  "preferred_regime": "BULL",
  "updated_policy": "AGGRESSIVE",
  "confidence_boost": 0.5951,
  "leverage_hint": 1.74,
  "regime_performance": {
    "avg_pnl": 0.39,
    "win_rate": 1.0,
    "sample_count": 21
  },
  "timestamp": "2025-12-21T10:30:45Z"
}
```

### AI Engine Health:
```json
{
  "strategic_memory": {
    "status": "active",
    "preferred_regime": "BULL",
    "recommended_policy": "AGGRESSIVE",
    "confidence_boost": 0.5951,
    "leverage_hint": 1.74,
    "performance": {
      "avg_pnl": 0.39,
      "win_rate": 1.0
    }
  }
}
```

---

## 🔗 System Integration

### Phase 4S+ feeds into:

| Component | Data Used | Impact |
|-----------|-----------|--------|
| 🧩 **AI Engine** | confidence_boost | Adjusts strategy weights |
| 🧩 **Exit Brain v3.5** | recommended_policy | TP/SL aggressiveness |
| 🧩 **Exposure Balancer** | leverage_hint | Margin limit adjustment |
| 🧩 **Portfolio Governance** | updated_policy | Policy switching |
| 🧩 **RL Agent** | confidence_boost | Leverage multiplier |

---

## 📞 Troubleshooting

### Container not starting:
```bash
docker logs quantum_strategic_memory
docker ps -a | grep strategic_memory
```

### No feedback generated:
- Need 3+ regime observations
- Check stream length: `docker exec quantum_redis redis-cli XLEN quantum:stream:meta.regime`
- Wait for 60s processing cycle

### Redis connection issues:
```bash
docker exec quantum_redis redis-cli PING
docker ps | grep redis
```

### AI Engine not exposing metrics:
```bash
curl -s http://localhost:8001/health | jq '.metrics.strategic_memory'
docker logs quantum_ai_engine --tail 50
```

---

## 🎓 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 4S+ - STRATEGIC MEMORY SYNC                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📥 INPUT (9 Data Sources)                                  │
│    • quantum:governance:policy                              │
│    • quantum:governance:preferred_regime                    │
│    • quantum:stream:meta.regime                             │
│    • quantum:stream:portfolio.memory                        │
│    • quantum:stream:trade.results                           │
│    • quantum:exposure:current                               │
│    • quantum:leverage:active                                │
│    • quantum:exit:statistics                                │
│    • quantum:trades:history                                 │
│                                                             │
│  ⚙️ PROCESSING (60s Loop)                                   │
│    MemoryLoader → PatternAnalyzer → ReinforcementFeedback  │
│                                                             │
│  📤 OUTPUT                                                   │
│    • quantum:feedback:strategic_memory (Redis)              │
│    • quantum:events:strategic_feedback (Event Bus)          │
│    • AI Engine /health metrics                              │
│                                                             │
│  🔁 FEEDBACK LOOP                                            │
│    Portfolio Governance ← recommended_policy                │
│    RL Agent ← confidence_boost                              │
│    Exit Brain ← policy aggressiveness                       │
│    Exposure Balancer ← leverage_hint                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated:** December 21, 2025  
**Version:** Phase 4S+ Enhanced  
**Status:** ✅ Production Ready
