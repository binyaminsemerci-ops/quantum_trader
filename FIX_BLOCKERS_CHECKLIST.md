# 🔧 PRODUCTION BLOCKER FIXES

## SYSTEM DISCOVERY
**Kjørende**: AI Engine, Execution (PAPER mode), Redis, Postgres, Nginx, Prometheus, Grafana, Alertmanager
**Stoppet**: Backend (import errors), Risk-Safety (ESS import error)
**Ikke deployed**: MarketData, Monitoring Health, Portfolio Intelligence, RL Training

---

## BLOCKER 1: NULL TRADE HISTORY ✅ LØSNING KLAR

**Problem**: System har 0 trades, kjører i PAPER mode men ingen historikk.

**Fix**:
```bash
# 1. Generer test signals for å trigge paper trades
docker exec quantum_ai_engine python3 -c "
from redis import Redis
import json, time
redis = Redis(host='quantum_redis', port=6379)
for i in range(5):
    signal = {
        'symbol': 'BTCUSDT',
        'action': 'BUY',
        'confidence': 0.75,
        'timestamp': time.time()
    }
    redis.xadd('quantum:stream:market_data', {'payload': json.dumps(signal)})
    time.sleep(2)
print('5 test signals published')
"

# 2. Monitor trades
docker logs -f quantum_execution | grep -i "trade\|order"

# 3. Verify trade history
redis-cli XLEN quantum:stream:trade.closed
```

**Status**: PAPER mode aktiv ✅, men må generere faktiske signals

---

## BLOCKER 2: BINANCE API KEY TOM ⚠️ MÅ FIKSES

**Problem**: BINANCE_API_KEY=<tom>

**Fix options**:
```bash
# OPTION A: Binance Testnet (ANBEFALT FØRST)
# 1. Gå til: https://testnet.binance.vision/
# 2. Logg inn / registrer
# 3. Create API Key → copy key + secret
# 4. Legg til i .env:
echo "BINANCE_API_KEY=<testnet_key>" >> ~/quantum_trader/.env
echo "BINANCE_API_SECRET=<testnet_secret>" >> ~/quantum_trader/.env
echo "BINANCE_TESTNET=true" >> ~/quantum_trader/.env

# OPTION B: Real Binance (KUN etter testnet success)
# 1. Binance.com → API Management
# 2. Create API Key
# 3. Permissions: Enable Trading, DISABLE Withdrawals
# 4. IP Whitelist: 46.224.116.254
# 5. Legg til i .env (uten TESTNET flag)

# Restart execution
docker restart quantum_execution
```

**Status**: Tom - BLOCKER

---

## BLOCKER 3: ALERTS IKKE TESTET 🔴 TEST NÅ

**Fix**:
```bash
# Test 1: AI Engine down
docker stop quantum_ai_engine
# VERIFY: Telegram message innen 90s
sleep 120
docker start quantum_ai_engine

# Test 2: Redis down  
docker stop quantum_redis
# VERIFY: Telegram message innen 60s
sleep 90
docker start quantum_redis

# Test 3: Manual alert
curl -X POST http://localhost:9093/-/reload

# Dokumenter results
echo "Alert test results:" > ~/alert_test_$(date +%Y%m%d).txt
echo "AI Engine down: <latency>s" >> ~/alert_test_$(date +%Y%m%d).txt
echo "Redis down: <latency>s" >> ~/alert_test_$(date +%Y%m%d).txt
```

**Status**: Alertmanager kjører, Telegram config finnes, IKKE TESTET

---

## BLOCKER 4: RISK LIMITS FOR HØY ⚠️ FIX NÅ

**Problem**:
- MAX_POSITION_USD=1000 (for høyt)
- MAX_LEVERAGE=10 (farlig)
- Risk-Safety service crashed

**Fix**:
```bash
# 1. Reduser limits i .env
ssh qt@46.224.116.254
cd ~/quantum_trader
cp .env .env.backup

cat >> .env << 'EOF'
# PRODUCTION SAFETY LIMITS (overrides)
MAX_POSITION_USD=50
MAX_LEVERAGE=1
MAX_CONCURRENT_POSITIONS=1
MAX_DAILY_TRADES=3
MAX_DAILY_LOSS_USD=200
REQUIRE_MANUAL_APPROVAL=true
EOF

# 2. Restart execution
docker restart quantum_execution

# 3. Verify limits applied
docker exec quantum_execution env | grep MAX_

# 4. (OPTIONAL) Fix Risk-Safety service
# Issue: ESS import error
# Quick fix: Deploy without ESS integration først
```

**Status**: Limits for høye - FIX KRITISK

---

## BLOCKER 5: HTTPS PUBLIC EXPOSURE 🔒 FIX NÅ

**Problem**: Nginx bundet til 0.0.0.0:443 (public internet)

**Fix**:
```bash
# 1. Edit systemctl.wsl.yml
cd ~/quantum_trader
cp systemctl.wsl.yml systemctl.wsl.yml.backup

# Change nginx ports:
sed -i 's/- "0.0.0.0:80:80"/- "127.0.0.1:80:80"/' systemctl.wsl.yml
sed -i 's/- "0.0.0.0:443:443"/- "127.0.0.1:443:443"/' systemctl.wsl.yml

# 2. Restart nginx
docker compose -f systemctl.wsl.yml up -d nginx

# 3. Verify localhost only
systemctl list-units | grep nginx
# Should show: 127.0.0.1:80->80/tcp, 127.0.0.1:443->443/tcp

# 4. Access via SSH tunnel
# From local machine:
ssh -L 8443:127.0.0.1:443 -i ~/.ssh/hetzner_fresh qt@46.224.116.254
# Then open: https://localhost:8443/health
```

**Status**: PUBLIC - FIX KRITISK

---

## STOPPEDE SERVICES SOM MÅ FIXES

### Risk-Safety Service 🔴
**Error**: `ImportError: cannot import name 'ESS' from 'backend.core.safety.ess'`
**Impact**: Ingen automatisk risk management
**Quick fix**: 
```bash
# Check if ESS exists
ls -la ~/quantum_trader/backend/core/safety/ess.py

# Fix import eller disable service temporarily
# Add to systemctl: restart: "no" for risk_safety
```

### Backend Service 🟡
**Error**: Multiple import errors (trades_routes, etc.)
**Impact**: Ingen web API for dashboard
**Fix**: Already wrapped imports in try-except, men fortsatt feiler
**Status**: IKKE KRITISK (AI Engine + Execution fungerer uten den)

---

## ADDITIONAL SERVICES (ikke deployed)

### MarketData Service
**Path**: ~/quantum_trader/microservices/marketdata/
**Status**: Exists but not deployed
**Purpose**: Real-time market data ingestion
**Priority**: MEDIUM (AI Engine har existing data feed)

### Monitoring Health Service
**Path**: ~/quantum_trader/microservices/monitoring_health/
**Status**: Exists but not deployed  
**Purpose**: Health checks + metrics
**Priority**: LOW (Prometheus + Grafana covers this)

### Portfolio Intelligence Service
**Path**: ~/quantum_trader/microservices/portfolio_intelligence/
**Status**: Exists but not deployed
**Purpose**: Portfolio analytics + optimization
**Priority**: LOW (nice-to-have)

### RL Training Service
**Path**: ~/quantum_trader/microservices/rl_training/
**Status**: Exists but not deployed
**Purpose**: Reinforcement learning model training
**Priority**: LOW (models already trained)

---

## EXECUTION PLAN (PRIORITY ORDER)

### KRITISK (Gjør NÅ - 2 timer)
1. ✅ **Risk limits** - Reduser til safe levels (10 min)
2. ✅ **HTTPS localhost** - Bind til 127.0.0.1 (10 min)
3. ⚠️ **Binance API** - Få testnet keys (20 min)
4. 🔴 **Alert test** - Test alle alerts (30 min)
5. ⚠️ **Paper trading** - Generate signals + monitor trades (1 time)

### VIKTIG (Gjør i dag - 4 timer)
6. Fix Risk-Safety ESS import
7. Test full trade lifecycle (open → close)
8. Backup restore test
9. Kill-all script + procedure
10. Document all fixes

### NICE-TO-HAVE (Gjør denne uka)
11. Deploy MarketData service
12. Fix Backend service imports
13. Enable Portfolio Intelligence
14. Setup proper logging aggregation

---

## VERIFICATION CHECKLIST

```bash
# Run this after fixes:
echo "=== BLOCKER 1: Trade History ==="
redis-cli XLEN quantum:stream:trade.closed
# MUST be >0 (ideally >5)

echo "=== BLOCKER 2: Binance API ==="
docker exec quantum_execution env | grep BINANCE_API_KEY
# MUST show key (not empty)

echo "=== BLOCKER 3: Alerts ==="
cat ~/alert_test_*.txt
# MUST show successful test results

echo "=== BLOCKER 4: Risk Limits ==="
docker exec quantum_execution env | grep -E "MAX_POSITION|MAX_LEVERAGE"
# MUST show: MAX_POSITION_USD=50, MAX_LEVERAGE=1

echo "=== BLOCKER 5: HTTPS ==="
systemctl list-units | grep nginx | grep "127.0.0.1:443"
# MUST show localhost binding
```

---

## GO/NO-GO AFTER FIXES

**IF ALL GREEN**:
- ✅ Limits set to safe levels
- ✅ HTTPS localhost only
- ✅ Alerts tested and working
- ✅ Paper trades flowing
- ✅ Binance testnet connected

**THEN**: Proceed to 48h testnet validation

**IF ANY RED**: DO NOT GO LIVE

