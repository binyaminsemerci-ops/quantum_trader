# Phase 4R+ Deployment & Monitoring Scripts

## 📋 Overview

This directory contains deployment, validation, and monitoring scripts for the **Phase 4R Meta-Regime Correlator**.

## 🚀 Scripts

### 1. deploy_phase4r.sh (Linux/WSL)
**Purpose:** Deploy Meta-Regime Correlator to VPS with full validation

**Usage:**
```bash
# On VPS
cd /home/qt/quantum_trader/scripts
chmod +x deploy_phase4r.sh
./deploy_phase4r.sh
```

**What it does:**
- Pulls latest code from Git
- Builds meta-regime Docker image
- Starts the container
- Verifies deployment
- Injects test data
- Checks AI Engine health
- Displays comprehensive status report

**Output:**
```
🎯  PHASE 4R+ DEPLOYMENT COMPLETE
✅  Service Status:
   • Container: quantum_meta_regime
   • Status: Running
   • Redis Stream: quantum:stream:meta.regime (5 entries)
   • Preferred Regime: BULL
```

---

### 2. deploy_phase4r.ps1 (Windows PowerShell)
**Purpose:** Deploy Meta-Regime Correlator from Windows to VPS via SSH

**Usage:**
```powershell
cd C:\quantum_trader\scripts
.\deploy_phase4r.ps1
```

**Parameters:**
- `-VpsHost`: VPS hostname or IP (default: 46.224.116.254)
- `-VpsUser`: SSH username (default: qt)
- `-SshKey`: SSH private key path (default: ~/.ssh/hetzner_fresh)

**Example with custom VPS:**
```powershell
.\deploy_phase4r.ps1 -VpsHost "192.168.1.100" -VpsUser "trader" -SshKey "~/.ssh/my_key"
```

**What it does:**
- Creates deployment archive
- Uploads to VPS via SSH/SCP
- Extracts and builds on VPS
- Starts container
- Injects test regime data
- Validates AI Engine integration
- Shows deployment summary

---

### 3. monitor_meta_regime.ps1 (Windows PowerShell)
**Purpose:** Real-time monitoring of Meta-Regime Correlator activity

**Usage:**
```powershell
cd C:\quantum_trader\scripts
.\monitor_meta_regime.ps1
```

**Parameters:**
- `-VpsHost`: VPS hostname (default: 46.224.116.254)
- `-VpsUser`: SSH username (default: qt)
- `-SshKey`: SSH key path (default: ~/.ssh/hetzner_fresh)
- `-RefreshInterval`: Refresh rate in seconds (default: 20)

**Example with 10-second refresh:**
```powershell
.\monitor_meta_regime.ps1 -RefreshInterval 10
```

**Display Features:**
- Current preferred regime (BULL/BEAR/RANGE/VOLATILE)
- Current governance policy (AGGRESSIVE/BALANCED/CONSERVATIVE)
- Stream sample count
- AI Engine health status
- Best performing regime with PnL
- Regime performance breakdown table
- Recent activity logs

**Screenshot Example:**
```
═══════════════════════════════════════════════════════════════════════════
  🧠 META-REGIME CORRELATOR - REAL-TIME MONITOR
═══════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📊 CURRENT REGIME STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Preferred Regime: BULL
  Current Policy:   AGGRESSIVE
  Stream Samples:   145

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  📈 REGIME PERFORMANCE BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Regime      Samples   Avg PnL    Win Rate   Avg Vol
  ───────────────────────────────────────────────────
  BULL           68     0.0385      62.00%     0.013
  RANGE          42     0.0098      54.00%     0.007
  BEAR           22    -0.0165      38.00%     0.024
  VOLATILE       13    -0.0280      31.00%     0.045
```

**Controls:**
- Press `Ctrl+C` to exit

---

### 4. test_meta_regime.ps1 (Windows PowerShell)
**Purpose:** Test Meta-Regime Correlator with various market scenarios

**Usage:**
```powershell
cd C:\quantum_trader\scripts
.\test_meta_regime.ps1
```

**Test Scenarios:**
1. **Strong Bull Market** - High positive PnL, low volatility
2. **Bear Market with Losses** - Negative PnL, moderate volatility
3. **Volatile Mixed Performance** - High volatility, mixed results
4. **Range-Bound Small Profits** - Low volatility, small gains

**What it does:**
- Clears existing regime data
- Injects 4 different market scenarios
- Shows regime preference after each scenario
- Displays final statistics
- Validates AI Engine response

**Output:**
```
═══════════════════════════════════════════════════════════════════════════
  🧪 META-REGIME CORRELATOR - TEST SUITE
═══════════════════════════════════════════════════════════════════════════

📊  Testing Scenario: Strong Bull Market
    Injecting 6 observations...
    ✅  Injected successfully
    📈  Result:
        Preferred Regime: BULL
        Current Policy: AGGRESSIVE

[... more scenarios ...]

═══════════════════════════════════════════════════════════════════════════
  📊 FINAL STATISTICS
═══════════════════════════════════════════════════════════════════════════

  Total Observations: 21
  Preferred Regime:   BULL
  Current Policy:     AGGRESSIVE

  🎯  Best Performing Regime: BULL
      Average PnL: 0.0420
      Total Regimes Detected: 4
      Status: active

  ✅  TEST SUITE COMPLETED
```

---

## 📊 Manual Monitoring Commands

### Check Container Status
```bash
# On VPS
docker ps --filter 'name=quantum_meta_regime'
docker logs -f quantum_meta_regime
```

### Check Redis Data
```bash
# Stream length
docker exec redis redis-cli XLEN quantum:stream:meta.regime

# Preferred regime
docker exec redis redis-cli GET quantum:governance:preferred_regime

# Regime statistics
docker exec redis redis-cli GET quantum:governance:regime_stats | jq

# Current policy
docker exec redis redis-cli GET quantum:governance:policy
```

### Check AI Engine Health
```bash
# Full health
curl -s http://localhost:8001/health | jq '.metrics.meta_regime'

# Quick status
curl -s http://localhost:8001/health | jq -r '.metrics.meta_regime.status'
```

### Watch Preferred Regime (Auto-refresh)
```bash
# Linux/WSL
watch -n 20 'docker exec redis redis-cli GET quantum:governance:preferred_regime'

# Windows PowerShell
while ($true) { 
    docker exec redis redis-cli GET quantum:governance:preferred_regime
    Start-Sleep -Seconds 20 
}
```

---

## 🔧 Troubleshooting

### Container Not Starting
```bash
# Check logs
docker logs quantum_meta_regime

# Check Redis connectivity
docker exec quantum_meta_regime python -c "import redis; r=redis.from_url('redis://redis:6379/0'); r.ping(); print('OK')"

# Restart container
docker compose -f docker-compose.vps.yml restart meta-regime
```

### No Regime Data
```bash
# Check if cross-exchange feed is running
docker ps | grep cross_exchange

# Check market data streams
docker exec redis redis-cli XLEN quantum:market:BTCUSDT:prices

# Inject test data manually
docker exec redis redis-cli XADD quantum:stream:meta.regime '*' \
    regime BULL pnl 0.25 volatility 0.012 trend 0.002 confidence 0.85
```

### Policy Not Updating
```bash
# Check governance agent is running
docker ps | grep portfolio_governance

# Check governance policy key
docker exec redis redis-cli GET quantum:governance:policy

# Check meta-regime logs for policy updates
docker logs quantum_meta_regime | grep "Policy updated"
```

---

## 🎯 Expected Results

After successful deployment:

1. **Container Status:** Running & Healthy
2. **Preferred Regime:** Set within 5 minutes (or after test data injection)
3. **Stream Length:** Growing over time (1 entry per analysis interval)
4. **Policy Updates:** Automatic when regime changes
5. **AI Engine Health:** `status: "active"`

---

## 📝 Integration Checklist

- [ ] Meta-regime container running
- [ ] Redis stream populating
- [ ] Preferred regime being set
- [ ] Policy auto-updating on regime change
- [ ] AI Engine health showing meta-regime metrics
- [ ] Portfolio Governance receiving policy updates
- [ ] RL Sizing Agent getting regime context
- [ ] Exposure Balancer adjusting based on regime

---

## 🚀 Quick Start Guide

**First-Time Deployment:**
```powershell
# 1. Deploy to VPS
.\deploy_phase4r.ps1

# 2. Test with scenarios
.\test_meta_regime.ps1

# 3. Start monitoring
.\monitor_meta_regime.ps1
```

**Daily Operations:**
```powershell
# Start monitoring session
.\monitor_meta_regime.ps1

# Or use watch command
watch -n 20 'docker exec redis redis-cli GET quantum:governance:preferred_regime'
```

---

## 📚 Additional Resources

- **Full Documentation:** `AI_PHASE_4R_META_REGIME_DEPLOYED.md`
- **Component README:** `microservices/meta_regime/README.md`
- **Health Endpoint:** `http://vps-ip:8001/health`
- **Grafana Dashboard:** `http://vps-ip:3000` (if configured)

---

**Last Updated:** December 21, 2025  
**Version:** 1.0.0
