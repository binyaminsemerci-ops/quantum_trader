# 🚀 Quick Dashboard Access Guide

**Oppdatert**: 2026-01-16

---

## 📊 **GRAFANA DASHBOARDS**

### **Tilgang:**
🌐 **URL**: https://app.quantumfond.com/grafana

🔑 **Login**:
- **Username**: `admin`
- **Password**: `admin123`

### **Slik finner du dashboards:**

1. **Logg inn** på https://app.quantumfond.com/grafana
2. **Klikk** på ☰ (hamburger menu) **øverst til venstre**
3. **Velg** "Dashboards" fra menyen
4. **Klikk** på mappen **"Quantum Trader"**
5. **Se** alle 6 dashboards:

   - ✅ **P1-B: Log Aggregation** (4 panels)
   - ✅ **Quantum Trader - Execution & Trading** (10 panels)
   - ✅ **Quantum Trader - Infrastructure** (11 panels)
   - ✅ **Quantum Trader - Redis & Postgres** (12 panels)
   - ✅ **Quantum Trader - System Overview** (9 panels)
   - ✅ **RL Shadow System - Performance Monitoring** (8 panels) 🧠

### **Direktelinker:**

**RL Shadow Dashboard**:
```
https://app.quantumfond.com/grafana/d/rl-shadow-performance
```

**Execution & Trading**:
```
https://app.quantumfond.com/grafana/d/2a0c7019-5143-4bec-8334-68371c1953fa
```

**System Overview**:
```
https://app.quantumfond.com/grafana/d/1fa65b1b-56ce-4ce0-8f7b-a5b05e0d89a0
```

---

## 🧠 **RL INTELLIGENCE DASHBOARD**

### **Tilgang:**
🌐 **URL**: https://app.quantumfond.com/rl

🔓 **Ingen login nødvendig**

### **Features:**
- ✅ **10 live symboler** med real-time grafer
- ✅ **Performance heatmap** (reward per symbol)
- ✅ **Correlation matrix** (hvordan symboler beveger seg sammen)
- ✅ **Auto-refresh** hver 3. sekund

### **Symboler som vises:**
- ETHUSDT
- BNBUSDT
- DOTUSDT
- OPUSDT
- SOLUSDT
- XRPUSDT
- BTCUSDT
- INJUSDT
- ARBUSDT
- STXUSDT

### **Hvis du ikke ser grafer:**
1. **Hard refresh** nettleseren: `Ctrl + Shift + R` (Windows) eller `Cmd + Shift + R` (Mac)
2. **Tøm cache**: F12 → Network tab → "Disable cache" ✓
3. **Reload** siden

---

## 🏠 **MAIN DASHBOARD (React)**

### **Tilgang:**
🌐 **URL**: https://app.quantumfond.com

🔓 **Ingen login nødvendig**

### **Sider:**
- `/` - **Overview**: System status, PnL, positions
- `/ai` - **AI Engine**: Modell accuracy, predictions, latency
- `/rl` - **RL Intelligence**: RL shadow system (10 symboler)
- `/portfolio` - **Portfolio**: Positions, exposure, drawdown
- `/risk` - **Risk**: VaR, CVaR, volatility, regime
- `/system` - **System Health**: CPU, RAM, disk, containers
- `/grafana` - **Grafana Link**: Redirect til Grafana

---

## 🔧 **Troubleshooting**

### **Problem: "Grafana finner ikke dashboards"**
✅ **Løsning**:
- Dashboards er i **"Quantum Trader" folder**, ikke root
- Bruk hamburger menu ☰ → Dashboards → Quantum Trader

### **Problem: "RL dashboard viser ikke grafer"**
✅ **Løsning**:
- Hard refresh: `Ctrl + Shift + R`
- Vent 3-5 sekunder for første datahenting
- Sjekk at backend kjører: `/api/rl-dashboard/` skal returnere JSON

### **Problem: "Kan ikke logge inn på Grafana"**
✅ **Løsning**:
- Username: `admin` (lowercase)
- Password: `admin123`
- Hvis ikke: Reset med `grafana-cli admin reset-admin-password admin123`

---

## 📱 **Quick Links**

| Dashboard | URL | Auth |
|-----------|-----|------|
| Main Frontend | https://app.quantumfond.com | None |
| RL Intelligence | https://app.quantumfond.com/rl | None |
| Grafana | https://app.quantumfond.com/grafana | admin:admin123 |
| RL Shadow (Grafana) | https://app.quantumfond.com/grafana/d/rl-shadow-performance | admin:admin123 |
| Backend API | https://app.quantumfond.com/api/health | None |

---

## 🎯 **Best Practices**

1. **For trading analysis**: Start med **RL Intelligence** → se hvilke symboler som performer best
2. **For system health**: Bruk **System Health** → sjekk CPU/RAM/disk
3. **For detailed metrics**: Bruk **Grafana dashboards** → time-series analysis
4. **For quick overview**: Bruk **Main Dashboard** → all-in-one view

---

**Last Updated**: 2026-01-16 23:45 UTC  
**Status**: ✅ All dashboards operational

