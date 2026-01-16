# 🚀 QUANTUM TRADER - FULLSTENDIG TESTNET AKTIVERINGSPLAN

**Dato:** 16. januar 2026  
**Mål:** Aktivere alle komponenter for live trading på Binance Testnet  
**Status:** Klargjort etter Docker→Systemd migration

---

## 📋 FORUTSETNINGER (PRE-FLIGHT CHECKLIST)

### 1. Binance Testnet Credentials
```bash
# Verifiser at du har testnet API keys
echo $BINANCE_TESTNET_API_KEY
echo $BINANCE_TESTNET_SECRET_KEY

# Hvis ikke satt, generer nye på:
# https://testnet.binancefuture.com
```

### 2. VPS Tilgang
```bash
# Test SSH tilkobling
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'echo "✅ VPS Connected"'
```

### 3. Git Repository Oppdatert
```bash
# På VPS: Sørg for latest code
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  cd /home/qt/quantum_trader && 
  git fetch origin && 
  git reset --hard origin/main && 
  echo "✅ Code Updated"
'
```

---

## 🔧 STEG 1: KONFIGURER TESTNET MILJØVARIABLER

### 1.1 Opprett Testnet Environment File
```bash
# På VPS:
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cat > /etc/quantum/testnet.env << EOF
# ============================================
# BINANCE TESTNET CREDENTIALS
# ============================================
USE_BINANCE_TESTNET=true
BINANCE_TESTNET=true
BINANCE_TESTNET_API_KEY=<DIN_TESTNET_API_KEY>
BINANCE_TESTNET_SECRET_KEY=<DIN_TESTNET_SECRET_KEY>

# ============================================
# TRADING CONFIGURATION
# ============================================
TRADING_MODE=TESTNET
TRADING_ENABLED=true
MAX_LEVERAGE=20
POSITION_SIZE_USD=100
MAX_POSITIONS=5
RISK_PER_TRADE=0.02

# ============================================
# AI ENGINE SETTINGS
# ============================================
AI_ENGINE_ENABLED=true
CONFIDENCE_THRESHOLD=0.65
MIN_SIGNAL_STRENGTH=0.70

# ============================================
# RL AGENT CONFIGURATION
# ============================================
RL_ENABLED=true
RL_MODE=shadow_gated
RL_CONFIDENCE_THRESHOLD=0.75

# ============================================
# PORTFOLIO SETTINGS
# ============================================
INITIAL_CAPITAL=10000
MAX_DRAWDOWN=0.15
DAILY_LOSS_LIMIT=500

# ============================================
# REDIS CONFIGURATION
# ============================================
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# ============================================
# LOGGING
# ============================================
LOG_LEVEL=INFO
LOG_TO_FILE=true
LOG_PATH=/var/log/quantum

# ============================================
# MONITORING
# ============================================
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
METRICS_PORT=9091
EOF
'
```

### 1.2 Sett Riktige Permissions
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  chmod 600 /etc/quantum/testnet.env &&
  chown quantum:quantum /etc/quantum/testnet.env &&
  echo "✅ Testnet Environment File Created"
'
```

### 1.3 Oppdater Alle Systemd Services til å Bruke Testnet Config
```bash
# På VPS: Oppdater alle service-filer
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  cd /etc/systemd/system &&
  
  # Legg til EnvironmentFile i alle quantum-*.service filer
  for service in quantum-*.service; do
    if ! grep -q "EnvironmentFile=/etc/quantum/testnet.env" "$service"; then
      sed -i "/\[Service\]/a EnvironmentFile=/etc/quantum/testnet.env" "$service"
      echo "✅ Updated: $service"
    fi
  done &&
  
  systemctl daemon-reload &&
  echo "✅ All Services Updated with Testnet Config"
'
```

---

## 🎯 STEG 2: AKTIVERINGSSEKVENS (RIKTIG REKKEFØLGE)

### 2.1 FASE 1: INFRASTRUKTUR (Foundation Layer)
**Må startes først - Andre services avhenger av disse**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 1: INFRASTRUKTUR" &&
  echo "========================================" &&
  
  # Redis (Kritisk - alt avhenger av dette)
  systemctl restart redis-server.service &&
  sleep 2 &&
  redis-cli PING | grep -q PONG && echo "✅ Redis: ONLINE" || echo "❌ Redis: FAILED" &&
  
  # PostgreSQL (Dashboard database)
  systemctl restart postgresql &&
  sleep 2 &&
  systemctl is-active --quiet postgresql && echo "✅ PostgreSQL: ONLINE" || echo "❌ PostgreSQL: FAILED" &&
  
  # Prometheus + Node Exporter (Monitoring)
  systemctl restart prometheus &&
  systemctl restart prometheus-node-exporter &&
  sleep 2 &&
  systemctl is-active --quiet prometheus && echo "✅ Prometheus: ONLINE" || echo "❌ Prometheus: FAILED" &&
  
  # Loki (Log aggregation)
  systemctl restart loki &&
  sleep 2 &&
  systemctl is-active --quiet loki && echo "✅ Loki: ONLINE" || echo "❌ Loki: FAILED" &&
  
  echo "" &&
  echo "✅ FASE 1 COMPLETE: Infrastructure Layer Running"
'
```

### 2.2 FASE 2: DATA PROVIDERS (Market Data)
**Leverer markedsdata til AI komponenter**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 2: DATA PROVIDERS" &&
  echo "========================================" &&
  
  # Market Data Publisher
  systemctl restart quantum-market-publisher.service &&
  sleep 3 &&
  
  # Cross-Exchange Data Aggregator
  systemctl restart quantum-cross-exchange.service &&
  sleep 3 &&
  
  # Universe OS (Symbol Discovery)
  systemctl restart quantum-universe-os.service &&
  sleep 2 &&
  
  # Binance PnL Tracker
  systemctl restart quantum-binance-pnl-tracker.service &&
  sleep 2 &&
  
  # Verifiser alle data providers
  for svc in market-publisher cross-exchange universe-os binance-pnl-tracker; do
    systemctl is-active --quiet quantum-${svc}.service && 
      echo "✅ ${svc}: ONLINE" || echo "❌ ${svc}: FAILED"
  done &&
  
  echo "" &&
  echo "✅ FASE 2 COMPLETE: Market Data Flowing"
'
```

### 2.3 FASE 3: AI CORE (Decision Making)
**AI Engine + Memory + Learning Systemer**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 3: AI CORE" &&
  echo "========================================" &&
  
  # AI Engine (Hjernen)
  systemctl restart quantum-ai-engine.service &&
  sleep 5 &&
  journalctl -u quantum-ai-engine.service --since "30 seconds ago" | grep -i "startup\|ready" &&
  
  # Continuous Learning Module
  systemctl restart quantum-clm.service &&
  sleep 3 &&
  
  # Strategic Memory
  systemctl restart quantum-strategic-memory.service &&
  sleep 2 &&
  
  # Strategic Evolution
  systemctl restart quantum-strategic-evolution.service &&
  sleep 2 &&
  
  # Meta Regime Detection
  systemctl restart quantum-meta-regime.service &&
  sleep 2 &&
  
  # Model Federation (Ensemble)
  systemctl restart quantum-model-federation.service &&
  sleep 2 &&
  
  # Verifiser AI Core
  for svc in ai-engine clm strategic-memory strategic-evolution meta-regime model-federation; do
    systemctl is-active --quiet quantum-${svc}.service && 
      echo "✅ ${svc}: ONLINE" || echo "❌ ${svc}: FAILED"
  done &&
  
  echo "" &&
  echo "✅ FASE 3 COMPLETE: AI Brain Active"
'
```

### 2.4 FASE 4: RL SYSTEM (Reinforcement Learning)
**Adaptive learning og policy optimization**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 4: RL SYSTEM" &&
  echo "========================================" &&
  
  # RL Agent (Primary decision maker)
  systemctl restart quantum-rl-agent.service &&
  sleep 3 &&
  
  # RL Trainer (Learning loop)
  systemctl restart quantum-rl-trainer.service &&
  sleep 2 &&
  
  # RL Monitor (Performance tracking)
  systemctl restart quantum-rl-monitor.service &&
  sleep 2 &&
  
  # RL Feedback V2 (Reward calculation)
  systemctl restart quantum-rl-feedback-v2.service &&
  sleep 2 &&
  
  # RL Sizer (Position sizing)
  systemctl restart quantum-rl-sizer.service &&
  sleep 2 &&
  
  # RL Dashboard (Monitoring UI)
  systemctl restart quantum-rl-dashboard.service &&
  sleep 2 &&
  
  # Verifiser RL System
  for svc in rl-agent rl-trainer rl-monitor rl-feedback-v2 rl-sizer rl-dashboard; do
    systemctl is-active --quiet quantum-${svc}.service && 
      echo "✅ ${svc}: ONLINE" || echo "❌ ${svc}: FAILED"
  done &&
  
  echo "" &&
  echo "✅ FASE 4 COMPLETE: RL Learning Active"
'
```

### 2.5 FASE 5: RISK MANAGEMENT (Safety Layer)
**Risk controls + Portfolio management**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 5: RISK MANAGEMENT" &&
  echo "========================================" &&
  
  # CEO Brain (Top-level strategy)
  systemctl restart quantum-ceo-brain.service &&
  sleep 3 &&
  
  # Strategy Brain (Tactical decisions)
  systemctl restart quantum-strategy-brain.service &&
  sleep 3 &&
  
  # Risk Brain (Risk assessment)
  systemctl restart quantum-risk-brain.service &&
  sleep 3 &&
  
  # Risk Safety (Hard limits)
  systemctl restart quantum-risk-safety.service &&
  sleep 2 &&
  
  # Portfolio Intelligence
  systemctl restart quantum-portfolio-intelligence.service &&
  sleep 2 &&
  
  # Portfolio Governance
  systemctl restart quantum-portfolio-governance.service &&
  sleep 2 &&
  
  # Exposure Balancer
  systemctl restart quantum-exposure-balancer.service &&
  sleep 2 &&
  
  # Verifiser Risk Management
  for svc in ceo-brain strategy-brain risk-brain risk-safety portfolio-intelligence portfolio-governance exposure-balancer; do
    systemctl is-active --quiet quantum-${svc}.service && 
      echo "✅ ${svc}: ONLINE" || echo "❌ ${svc}: FAILED"
  done &&
  
  echo "" &&
  echo "✅ FASE 5 COMPLETE: Risk Controls Active"
'
```

### 2.6 FASE 6: EXECUTION ENGINE (Trade Execution)
**Order placement + Position monitoring**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 6: EXECUTION ENGINE" &&
  echo "========================================" &&
  
  # Auto Executor (Order placer)
  systemctl restart quantum-execution.service &&
  sleep 5 &&
  journalctl -u quantum-execution.service --since "30 seconds ago" | grep -i "testnet\|connected" &&
  
  # Position Monitor (Track open positions)
  systemctl restart quantum-position-monitor.service &&
  sleep 3 &&
  
  # Trade Intent Consumer (Signal processor)
  systemctl restart quantum-trade-intent-consumer.service &&
  sleep 2 &&
  
  # Verifiser Execution
  for svc in execution position-monitor trade-intent-consumer; do
    systemctl is-active --quiet quantum-${svc}.service && 
      echo "✅ ${svc}: ONLINE" || echo "❌ ${svc}: FAILED"
  done &&
  
  echo "" &&
  echo "✅ FASE 6 COMPLETE: Ready to Execute Trades"
'
```

### 2.7 FASE 7: DASHBOARDS (Monitoring & UI)
**Visualisering og kontrollpanel**

```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "FASE 7: DASHBOARDS" &&
  echo "========================================" &&
  
  # Grafana (Metrics visualization)
  systemctl restart grafana-server &&
  sleep 3 &&
  
  # Main Dashboard Backend
  systemctl restart quantum-dashboard-backend.service &&
  sleep 2 &&
  
  # Main Dashboard Frontend
  systemctl restart quantum-dashboard-frontend.service &&
  sleep 2 &&
  
  # Quantum Fond Frontend (Public UI)
  systemctl restart quantum-quantumfond-frontend.service &&
  sleep 2 &&
  
  # Nginx Proxy (Reverse proxy)
  systemctl restart nginx &&
  sleep 2 &&
  
  # Verifiser Dashboards
  systemctl is-active --quiet grafana-server && echo "✅ Grafana: http://46.224.116.254:3000" || echo "❌ Grafana: FAILED" &&
  systemctl is-active --quiet quantum-dashboard-backend.service && echo "✅ Dashboard Backend: ONLINE" || echo "❌ Dashboard: FAILED" &&
  systemctl is-active --quiet nginx && echo "✅ Nginx: ONLINE" || echo "❌ Nginx: FAILED" &&
  
  echo "" &&
  echo "✅ FASE 7 COMPLETE: Monitoring Active"
'
```

---

## 📊 STEG 3: VERIFISERING (System Health Check)

### 3.1 Komplett Status Check
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "SYSTEM STATUS - KOMPLETT RAPPORT" &&
  echo "========================================" &&
  
  # Count active services
  ACTIVE=$(systemctl list-units "quantum-*.service" --state=active --no-legend | wc -l) &&
  FAILED=$(systemctl list-units "quantum-*.service" --state=failed --no-legend | wc -l) &&
  
  echo "" &&
  echo "📊 SERVICE METRICS:" &&
  echo "  ✅ Active: $ACTIVE services" &&
  echo "  ❌ Failed: $FAILED services" &&
  echo "" &&
  
  # Infrastructure
  echo "🏗️  INFRASTRUCTURE:" &&
  redis-cli PING | grep -q PONG && echo "  ✅ Redis: PONG" || echo "  ❌ Redis: DOWN" &&
  systemctl is-active --quiet postgresql && echo "  ✅ PostgreSQL: ONLINE" || echo "  ❌ PostgreSQL: DOWN" &&
  systemctl is-active --quiet prometheus && echo "  ✅ Prometheus: ONLINE" || echo "  ❌ Prometheus: DOWN" &&
  
  # Check AI Engine signals
  echo "" &&
  echo "🤖 AI ENGINE STATUS:" &&
  journalctl -u quantum-ai-engine.service --since "5 minutes ago" | grep -c "BUY\|SELL" | xargs -I {} echo "  📈 Signals generated (last 5min): {}" &&
  
  # Check RL Agent
  echo "" &&
  echo "🎓 RL AGENT STATUS:" &&
  journalctl -u quantum-rl-agent.service --since "5 minutes ago" | grep -c "episode\|reward" | xargs -I {} echo "  🧠 Learning updates (last 5min): {}" &&
  
  # Check Execution
  echo "" &&
  echo "⚡ EXECUTION ENGINE:" &&
  journalctl -u quantum-execution.service --since "5 minutes ago" | grep -ic "testnet" | xargs -I {} echo "  🌐 Testnet mode: {} confirmations" &&
  journalctl -u quantum-execution.service --since "5 minutes ago" | grep -c "ORDER" | xargs -I {} echo "  📝 Orders placed (last 5min): {}" &&
  
  # Check Positions
  echo "" &&
  echo "💼 PORTFOLIO:" &&
  redis-cli HLEN "positions:open" | xargs -I {} echo "  📊 Open positions: {}" &&
  redis-cli GET "portfolio:total_pnl" | xargs -I {} echo "  💰 Total PnL: {} USDT" &&
  
  # Resource usage
  echo "" &&
  echo "💻 SYSTEM RESOURCES:" &&
  echo "  📊 CPU: $(top -bn1 | grep "Cpu(s)" | awk "{print \$2}" | cut -d"%" -f1)%" &&
  echo "  📊 RAM: $(free -h | grep Mem | awk "{print \$3\"/\"\$2}")" &&
  echo "  📊 Disk: $(df -h / | tail -1 | awk "{print \$5\" used\"}")" &&
  
  echo "" &&
  echo "========================================" &&
  echo "✅ SYSTEM STATUS CHECK COMPLETE" &&
  echo "========================================"
'
```

### 3.2 Test Binance Testnet Connection
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  cd /home/qt/quantum_trader &&
  source /opt/quantum/venvs/ai-engine/bin/activate &&
  
  python3 -c "
import os
from binance.client import Client

api_key = os.getenv(\"BINANCE_TESTNET_API_KEY\")
api_secret = os.getenv(\"BINANCE_TESTNET_SECRET_KEY\")

client = Client(api_key, api_secret, testnet=True)

# Test connection
account = client.futures_account()
balance = float(account[\"totalWalletBalance\"])

print(f\"✅ Binance Testnet Connected\")
print(f\"💰 Account Balance: {balance} USDT\")
print(f\"📊 Positions: {len(account.get(\"positions\", []))} symbols\")
print(f\"🔓 Trading Enabled: True\")
"
'
```

### 3.3 Overvåk Live Logs (Real-time monitoring)
```bash
# AI Engine logs (Trading signals)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-ai-engine.service -f | grep --line-buffered -i "signal\|buy\|sell"'

# Execution logs (Order placements)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-execution.service -f | grep --line-buffered -i "order\|filled\|executed"'

# RL Agent logs (Learning progress)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl -u quantum-rl-agent.service -f | grep --line-buffered -i "episode\|reward\|policy"'
```

---

## 🎯 STEG 4: START TRADING (GO LIVE)

### 4.1 Enable Trading Mode
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  # Set trading enabled flag in Redis
  redis-cli SET "system:trading_enabled" "true" &&
  redis-cli SET "system:trading_mode" "TESTNET" &&
  redis-cli SET "system:max_positions" "5" &&
  redis-cli SET "system:risk_per_trade" "0.02" &&
  
  echo "✅ Trading Mode: ENABLED" &&
  echo "🌐 Environment: TESTNET" &&
  echo "💼 Max Positions: 5" &&
  echo "⚠️  Risk Per Trade: 2%"
'
```

### 4.2 Manuell Trade Test (Verifiser før full auto)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  # Push test signal til AI Engine
  redis-cli XADD "quantum:stream:ai_signals" "*" \\
    symbol "BTCUSDT" \\
    action "BUY" \\
    confidence "0.85" \\
    reason "MANUAL_TEST" \\
    timestamp "$(date +%s)" &&
  
  echo "✅ Test signal sent to AI Engine" &&
  echo "⏳ Watch execution logs for order placement..." &&
  
  # Watch for 30 seconds
  timeout 30 journalctl -u quantum-execution.service -f | grep --line-buffered -i "order"
'
```

### 4.3 Full Auto Mode (Let It Run)
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "========================================" &&
  echo "🚀 FULL AUTO MODE - ACTIVATED" &&
  echo "========================================" &&
  
  # Bekreft alle systemer er klare
  redis-cli SET "system:auto_mode" "true" &&
  redis-cli SET "system:last_activated" "$(date -Iseconds)" &&
  
  echo "" &&
  echo "✅ System Status:" &&
  echo "  🤖 AI Engine: ACTIVE" &&
  echo "  🎓 RL Agent: LEARNING" &&
  echo "  ⚡ Execution: READY" &&
  echo "  🛡️  Risk Controls: ENABLED" &&
  echo "" &&
  echo "🌐 Trading on: Binance Testnet" &&
  echo "💰 Initial Capital: 10,000 USDT" &&
  echo "📊 Max Positions: 5" &&
  echo "⚠️  Risk Per Trade: 2%" &&
  echo "🎯 Confidence Threshold: 65%" &&
  echo "" &&
  echo "========================================" &&
  echo "🎉 QUANTUM TRADER IS NOW LIVE!" &&
  echo "========================================"
'
```

---

## 📍 DASHBOARDS & MONITORING URLS

### Grafana (Metrics & Charts)
**URL:** http://46.224.116.254:3000  
**User:** admin  
**Pass:** [Se VPS /etc/grafana/grafana.ini]

**Dashboards å sjekke:**
- Quantum Overview → Systemstatus
- AI Engine Performance → Signal quality
- RL Training Progress → Learning metrics
- Position Monitor → Åpne posisjoner
- Risk Dashboard → Risk metrics

### RL Dashboard (RL-specific metrics)
**URL:** http://46.224.116.254:8025 (Via SSH tunnel)
```bash
# Start SSH tunnel lokalt
wsl ssh -i ~/.ssh/hetzner_fresh -L 8025:localhost:8025 root@46.224.116.254 -N
# Åpne: http://localhost:8025
```

### Main Dashboard (System health)
**URL:** http://46.224.116.254:8080
**Features:**
- Service status (All quantum-*.service units)
- CPU/RAM/Disk usage
- Recent logs
- Quick actions (restart services, etc.)

---

## ⚠️ NOTFALLSTOPP (Emergency Stop)

### Stopp All Trading Umiddelbart
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  echo "🛑 EMERGENCY STOP - Disabling trading..." &&
  
  # Disable trading i Redis
  redis-cli SET "system:trading_enabled" "false" &&
  redis-cli SET "system:emergency_stop" "true" &&
  redis-cli SET "system:stop_reason" "MANUAL_EMERGENCY_STOP" &&
  redis-cli SET "system:stop_timestamp" "$(date -Iseconds)" &&
  
  # Stopp execution engine (stoppekk ordre plassering)
  systemctl stop quantum-execution.service &&
  
  echo "✅ Trading disabled" &&
  echo "✅ Execution engine stopped" &&
  echo "⚠️  Open positions still active (må lukkes manuelt)"
'
```

### Lukk Alle Åpne Posisjoner
```bash
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 '
  cd /home/qt/quantum_trader &&
  source /opt/quantum/venvs/ai-engine/bin/activate &&
  python3 cancel_all_orders_testnet.py &&
  python3 close_all_positions_testnet.py &&
  echo "✅ All positions closed"
'
```

---

## 🔄 DAGLIG RUTINE (Daily Operations)

### Morgen Check (Før markedet åpner)
```bash
# 1. Verifiser alle services kjører
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'systemctl list-units "quantum-*.service" --state=failed'

# 2. Sjekk nattprestasjon
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'redis-cli GET "portfolio:daily_pnl"'

# 3. Verifiser testnet balance
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cd /home/qt/quantum_trader && python3 check_testnet_balance.py'
```

### Kveldsstatus (Etter markedet stenger)
```bash
# 1. Generer daglig rapport
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cd /home/qt/quantum_trader && ./scripts/daily_report.sh'

# 2. Backup Redis data
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'redis-cli BGSAVE'

# 3. Rotate logs (hvis nødvendig)
wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'journalctl --vacuum-time=7d'
```

---

## 📚 NYTTIGE KOMMANDOER (Quick Reference)

### Service Management
```bash
# Start alle Quantum services
systemctl start quantum-*.service

# Restart spesifikk service
systemctl restart quantum-ai-engine.service

# Se status
systemctl status quantum-execution.service

# Se logs (live)
journalctl -u quantum-ai-engine.service -f

# Se logs (siste 1 time)
journalctl -u quantum-execution.service --since "1 hour ago"
```

### Redis Debugging
```bash
# Se alle keys
redis-cli KEYS "quantum:*"

# Hent verdi
redis-cli GET "system:trading_enabled"

# Se stream lengde
redis-cli XLEN "quantum:stream:ai_signals"

# Les siste 10 meldinger fra stream
redis-cli XREVRANGE "quantum:stream:ai_signals" + - COUNT 10
```

### Performance Monitoring
```bash
# CPU per service
systemd-cgtop

# Memory usage
systemctl status quantum-*.service | grep Memory

# Disk I/O
iotop -o
```

---

## ✅ SUCCESS CRITERIA (Systemet fungerer hvis...)

- [x] Redis: PONG response
- [x] AI Engine: Genererer BUY/SELL signals hver 5-10 min
- [x] RL Agent: Episode updates i logs
- [x] Execution: Testnet orders plasseres uten errors
- [x] Risk Controls: Ingen ordre over configured limits
- [x] Position Monitor: Tracking åpne posisjoner i Redis
- [x] Dashboards: Alle URLs tilgjengelige
- [x] No critical errors i journalctl logs
- [x] CPU < 80%, RAM < 90%, Disk < 80%

---

## 🆘 TROUBLESHOOTING

### Problem: Service won't start
```bash
# Check detailed status
systemctl status quantum-SERVICE.service -l

# Check logs
journalctl -u quantum-SERVICE.service --since "10 minutes ago"

# Check if port is already in use
netstat -tlnp | grep PORT_NUMBER

# Restart dependencies first
systemctl restart redis-server
systemctl restart quantum-ai-engine.service
systemctl restart quantum-SERVICE.service
```

### Problem: No trading signals
```bash
# Check AI Engine is processing
journalctl -u quantum-ai-engine.service -f | grep -i "signal\|inference"

# Check Redis connection
redis-cli PING

# Check market data flowing
redis-cli XLEN "quantum:stream:market_data"

# Restart AI Engine
systemctl restart quantum-ai-engine.service
```

### Problem: Orders not executing
```bash
# Check execution service
systemctl status quantum-execution.service

# Check testnet credentials
redis-cli GET "credentials:binance:testnet:valid"

# Check execution logs
journalctl -u quantum-execution.service --since "30 minutes ago"

# Test manual order
python3 test_binance_testnet_order.py
```

---

## 📞 SUPPORT & RESOURCES

- **VPS SSH:** `wsl ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254`
- **Grafana:** http://46.224.116.254:3000
- **Binance Testnet:** https://testnet.binancefuture.com
- **Service Matrix:** `/home/qt/quantum_trader/ops/systemd/SERVICE_MATRIX.md`
- **Architecture Docs:** `/home/qt/quantum_trader/docs/`

---

**🎉 QUANTUM TRADER ER NÅ KLAR FOR TESTNET TRADING!**

**Next Steps:**
1. Følg aktiveringssekvensen over (Fase 1-7)
2. Kjør full verifikasjon (Steg 3)
3. Start med manuell test (Steg 4.2)
4. Aktiver full auto mode (Steg 4.3)
5. Overvåk i Grafana dashboards

**Good luck! 🚀📈**
