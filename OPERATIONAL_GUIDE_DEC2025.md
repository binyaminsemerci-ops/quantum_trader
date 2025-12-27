# QUANTUM TRADER - OPERASJONELL GUIDE
**Dato:** 17. desember 2025  
**Type:** Praktisk guide for drift og bruk av systemet  
**Målgruppe:** Operatører, utviklere, traders

---

## INNHOLDSFORTEGNELSE

1. [Oppstart og Nedstengning](#1-oppstart-og-nedstengning)
2. [Daglig Drift](#2-daglig-drift)
3. [Monitoring og Alarmhåndtering](#3-monitoring-og-alarmhåndtering)
4. [Troubleshooting](#4-troubleshooting)
5. [Konfigurasjon og Tuning](#5-konfigurasjon-og-tuning)
6. [Model Management](#6-model-management)
7. [Risk Management](#7-risk-management)
8. [Backup og Recovery](#8-backup-og-recovery)

---

## 1. OPPSTART OG NEDSTENGNING

### 1.1 Første gangs oppsett

#### Forhåndskrav
```bash
# Verify prerequisites
python --version  # Python 3.12+
node --version    # Node.js 18+
docker --version  # Docker 24+
```

#### Miljøvariabler Setup
```bash
# 1. Kopier .env.example til .env
cp .env.example .env

# 2. Rediger .env med dine API keys
nano .env

# Kritiske variabler:
BINANCE_API_KEY=***YOUR_KEY***
BINANCE_API_SECRET=***YOUR_SECRET***
BINANCE_TESTNET=true  # Start med testnet!
QT_ADMIN_TOKEN=***GENERATE_SECURE_TOKEN***
```

#### Database Initialisering
```bash
# Initialize database
python backend/database/init_db.py

# Verify tables created
sqlite3 backend/quantum_trader.db ".tables"
```

### 1.2 Start Systemet (Docker)

#### Full Stack Start
```bash
# Start alle services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f backend
```

**Forventet output:**
```
✅ Redis initialized
✅ PolicyStore v2 loaded
✅ EventBus v2 connected
✅ AI models loaded (4/4)
✅ Event-Driven Executor started
✅ Scheduler started
✅ Dashboard ready at http://localhost:5173
✅ Backend API at http://localhost:8000
```

#### Individuell Service Start
```bash
# Start kun backend
docker-compose up -d backend

# Start kun frontend
cd frontend && npm run dev

# Start AI engine separat
python ai_engine/ensemble_manager.py
```

### 1.3 Start Systemet (Lokal)

#### Backend Start
```bash
# Activate virtual environment
cd backend
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Start FastAPI
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Start
```bash
cd frontend
npm install
npm run dev
```

#### AI Engine Start
```bash
# Start i separat terminal
cd ai_engine
python ensemble_manager.py
```

### 1.4 Verifisering av Oppstart

#### Health Checks
```bash
# Backend health
curl http://localhost:8000/api/health

# Expected:
# {"status":"healthy","timestamp":"2025-12-17T10:00:00Z"}

# Architecture v2 health
curl http://localhost:8000/api/v2/health

# Expected:
# {
#   "status":"HEALTHY",
#   "dependencies":{
#     "redis":"HEALTHY",
#     "binance_rest":"HEALTHY"
#   }
# }

# Frontend
curl http://localhost:5173

# Expected: HTML response
```

#### System Status Check
```bash
# Check all components
python check_system_status.py

# Expected output:
# ✅ Backend: RUNNING
# ✅ AI Engine: LOADED (4 models)
# ✅ Redis: CONNECTED
# ✅ Database: OK
# ✅ Binance API: CONNECTED
# ✅ Event-Driven Executor: ACTIVE
```

### 1.5 Graceful Shutdown

#### Stop alle services
```bash
# Graceful shutdown (Docker)
docker-compose down

# With cleanup
docker-compose down -v  # Remove volumes

# Force stop
docker-compose kill
```

#### Stop individuell service
```bash
# Backend
pkill -f "uvicorn main:app"

# Frontend
pkill -f "npm run dev"
```

#### Pre-shutdown checklist
```bash
# 1. Check for open positions
curl http://localhost:8000/api/positions

# 2. Close all positions if needed
curl -X POST http://localhost:8000/api/trades/close-all

# 3. Stop executor
curl -X POST http://localhost:8000/api/executor/stop

# 4. Backup database
cp backend/quantum_trader.db backups/db_$(date +%Y%m%d_%H%M%S).db

# 5. Now safe to shutdown
docker-compose down
```

---

## 2. DAGLIG DRIFT

### 2.1 Morgen Rutine (Pre-Market)

```bash
# 1. System Health Check
curl http://localhost:8000/api/v2/health | jq

# 2. Check overnight performance
curl http://localhost:8000/api/stats/daily | jq

# 3. Review AI model status
curl http://localhost:8000/api/ai/models/status | jq

# 4. Check risk metrics
curl http://localhost:8000/api/risk/status | jq

# Expected checks:
# ✅ All models loaded
# ✅ Daily drawdown < 10%
# ✅ Open positions within limits
# ✅ No emergency brake active
```

### 2.2 Løpende Monitoring

#### Dashboard Monitoring
```
Open: http://localhost:5173

Monitor:
1. Positions Table
   - ✅ PnL green (positive)
   - ⚠️ Check for stuck positions (>24h old)
   
2. Signals Feed
   - ✅ New signals appearing (10-15/hour normal)
   - ⚠️ No signals = issue with AI engine
   
3. Risk Dashboard
   - ✅ Daily DD < 15%
   - ✅ Exposure < 200%
   - ✅ Win rate > 50%
   
4. System Metrics
   - ✅ API latency < 500ms
   - ✅ Model confidence > 70%
   - ✅ Execution success rate > 95%
```

#### Command Line Monitoring
```bash
# Check positions
python check_live_positions.py

# Monitor logs (live)
docker-compose logs -f --tail=50 backend

# Check trading activity
python check_todays_trading.py

# Monitor AI signals
python watch_signals.py
```

### 2.3 Kveld Rutine (Post-Market)

```bash
# 1. Generate daily report
python generate_trade_report.py

# 2. Review model performance
curl http://localhost:8000/api/ai/models/performance | jq

# 3. Check if retraining needed
curl http://localhost:8000/api/clm/status | jq

# 4. Backup important data
./scripts/daily_backup.sh

# 5. Review alerts (if any)
curl http://localhost:8000/api/alerts/today | jq
```

### 2.4 Ukentlig Rutine

```bash
# Sunday evening or Monday morning

# 1. Full system audit
python full_system_check.py

# 2. Review weekly performance
curl http://localhost:8000/api/stats/weekly | jq

# 3. Update universe (if needed)
python universe_analyzer.py

# 4. Check model drift
curl http://localhost:8000/api/clm/drift-report | jq

# 5. Review and adjust risk parameters
nano .env  # Update RM_MAX_DAILY_DD_PCT if needed

# 6. Restart services for fresh state
docker-compose restart
```

---

## 3. MONITORING OG ALARMHÅNDTERING

### 3.1 Prometheus Metrics

#### Konfigurer Prometheus
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'quantum_trader'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

#### Key Metrics å overvåke
```
# Trading Metrics
qt_trades_total                    # Total trades
qt_positions_count                 # Open positions
qt_pnl_total                       # Total PnL
qt_win_rate                        # Win rate

# AI Metrics
qt_model_predictions_total         # Predictions count
qt_model_latency_seconds           # Prediction latency
qt_signal_confidence               # Signal confidence

# Risk Metrics
qt_drawdown_pct                    # Current drawdown
qt_exposure_pct                    # Portfolio exposure
qt_emergency_brake                 # Emergency status

# System Metrics
qt_memory_usage_bytes              # Memory usage
qt_cpu_usage_pct                   # CPU usage
qt_api_latency_ms                  # API latency
```

### 3.2 Alert Rules

#### Critical Alerts (Immediate Action Required)

**Alert 1: High Drawdown**
```yaml
alert: HighDrawdown
expr: qt_drawdown_pct{period="daily"} > 15
for: 1m
severity: CRITICAL
action: |
  1. Check positions immediately
  2. Review recent trades
  3. Consider manual intervention
  4. Emergency brake may trigger
```

**Alert 2: Emergency Brake Active**
```yaml
alert: EmergencyBrakeActive
expr: qt_emergency_brake == 1
for: 0m
severity: CRITICAL
action: |
  1. Emergency Stop System activated
  2. No new trades allowed
  3. Review trigger reason
  4. Manual reset required
```

**Alert 3: Model Loading Failed**
```yaml
alert: ModelLoadingFailed
expr: qt_models_loaded < 4
for: 5m
severity: CRITICAL
action: |
  1. Check model files exist
  2. Review backend logs
  3. May need to retrain
  4. Trading may be impaired
```

#### Warning Alerts (Monitor Closely)

**Alert 4: Low Win Rate**
```yaml
alert: LowWinRate
expr: qt_win_rate < 0.50
for: 1h
severity: WARNING
action: |
  1. Review recent signals
  2. Check market conditions
  3. Consider lowering confidence threshold
  4. Monitor for improvement
```

**Alert 5: High API Latency**
```yaml
alert: HighAPILatency
expr: qt_api_latency_ms > 1000
for: 5m
severity: WARNING
action: |
  1. Check Binance API status
  2. Verify network connection
  3. May affect order execution
  4. Consider rate limit issues
```

### 3.3 Log Analysis

#### Viktige log patterns å se etter

**Normal Operation:**
```
INFO: Market scan complete, 50 symbols
INFO: AI ensemble prediction: BTCUSDT LONG @ 0.82 confidence
INFO: Risk check passed for BTCUSDT
INFO: Order placed: BTCUSDT LONG 0.1 @ 43250.50
INFO: Order filled: BTCUSDT LONG 0.1 @ 43251.00
```

**Warning Signs:**
```
WARNING: Model prediction latency high: 2.5s
WARNING: Binance API rate limit approaching
WARNING: Position BTCUSDT approaching TP
WARNING: Daily drawdown at 12%
```

**Error Conditions:**
```
ERROR: Failed to load model: xgboost_v20251213.pkl
ERROR: Binance API error: INSUFFICIENT_BALANCE
ERROR: Position not found: uuid-123
ERROR: Risk check failed: Exceeds max exposure
```

#### Log Monitoring Commands
```bash
# Follow logs in real-time
tail -f backend/logs/backend.log

# Search for errors
grep ERROR backend/logs/backend.log | tail -20

# Check specific symbol
grep "BTCUSDT" backend/logs/trading.log

# Monitor AI predictions
grep "ensemble prediction" backend/logs/ai_engine.log
```

---

## 4. TROUBLESHOOTING

### 4.1 Common Issues

#### Issue 1: "Model failed to load"

**Symptoms:**
```
ERROR: Failed to load model xgboost_v20251213.pkl
WARNING: Running with reduced model count
```

**Diagnose:**
```bash
# Check if model file exists
ls -la ai_engine/models/

# Check file permissions
ls -l ai_engine/models/xgboost_v*.pkl

# Try loading manually
python -c "import pickle; pickle.load(open('ai_engine/models/xgboost_v20251213.pkl', 'rb'))"
```

**Solution:**
```bash
# Option 1: Restore from backup
cp backups/models/xgboost_v20251213.pkl ai_engine/models/

# Option 2: Retrain model
python backend/train_model.py

# Option 3: Use older version
# Edit ensemble_manager.py to use previous version

# Restart backend
docker-compose restart backend
```

#### Issue 2: "No signals being generated"

**Symptoms:**
```
- Dashboard shows no signals
- Signals feed empty
- No new trades
```

**Diagnose:**
```bash
# Check AI engine status
curl http://localhost:8000/api/ai/status | jq

# Check market data
curl http://localhost:8000/api/prices/recent/BTCUSDT | jq

# Check model predictions manually
python test_ai_prediction.py BTCUSDT
```

**Solution:**
```bash
# Check if executor is running
curl http://localhost:8000/api/executor/status

# Restart executor
curl -X POST http://localhost:8000/api/executor/restart

# Check confidence threshold
# If too high (>80%), lower it
# Edit .env: QT_CONFIDENCE_THRESHOLD=0.65

# Restart
docker-compose restart backend
```

#### Issue 3: "Orders not executing"

**Symptoms:**
```
- Signals generated but no trades
- "Risk check failed" in logs
- Orders stuck in PENDING
```

**Diagnose:**
```bash
# Check risk limits
curl http://localhost:8000/api/risk/status | jq

# Check account balance
python check_account_balance.py

# Check open positions
python check_positions.py

# Review failed orders
grep "Risk check failed" backend/logs/backend.log
```

**Solution:**
```bash
# If max positions reached
# Close some positions manually or increase limit
# Edit .env: RM_MAX_CONCURRENT_TRADES=25

# If daily drawdown exceeded
# Wait for daily reset or increase limit (carefully!)
# Edit .env: RM_MAX_DAILY_DD_PCT=0.20

# If emergency brake active
# Investigate trigger and reset
curl -X POST http://localhost:8000/api/ess/reset

# Restart
docker-compose restart backend
```

#### Issue 4: "High memory usage"

**Symptoms:**
```
- System slow
- OOM errors in logs
- Docker containers restarting
```

**Diagnose:**
```bash
# Check memory usage
docker stats

# Check backend memory
curl http://localhost:8000/api/v2/health | jq '.system_metrics.memory_usage_bytes'

# Identify memory leaks
python -m memory_profiler backend/main.py
```

**Solution:**
```bash
# Option 1: Increase Docker memory limit
# Edit docker-compose.yml
services:
  backend:
    mem_limit: 4g

# Option 2: Reduce model count
# Disable heavy models (N-HiTS, PatchTST)
# Edit .env: AI_MODEL=xgb  # Use single model

# Option 3: Clear cache
curl -X POST http://localhost:8000/api/cache/clear

# Option 4: Restart with clean state
docker-compose down
docker system prune -f
docker-compose up -d
```

### 4.2 Emergency Procedures

#### Emergency Procedure 1: Close All Positions

```bash
# If system malfunction or extreme market event

# Step 1: Stop executor (no new trades)
curl -X POST http://localhost:8000/api/executor/stop

# Step 2: Close all losing positions
python emergency_close_losing.py

# Step 3: Close all positions
python close_all_positions_now.py

# Step 4: Verify all closed
python check_positions.py

# Step 5: Activate emergency brake
curl -X POST http://localhost:8000/api/ess/activate
```

#### Emergency Procedure 2: System Restart

```bash
# If system unresponsive or corrupted state

# Step 1: Backup current state
./scripts/emergency_backup.sh

# Step 2: Stop all services
docker-compose kill

# Step 3: Clear temporary data
rm -rf /tmp/qt_*
redis-cli FLUSHALL

# Step 4: Verify database integrity
sqlite3 backend/quantum_trader.db "PRAGMA integrity_check;"

# Step 5: Restart fresh
docker-compose up -d

# Step 6: Verify startup
python check_system_status.py
```

#### Emergency Procedure 3: Rollback to Previous Version

```bash
# If new deployment causing issues

# Step 1: Stop current version
docker-compose down

# Step 2: Checkout previous version
git log --oneline  # Find last stable commit
git checkout <commit-hash>

# Step 3: Restore previous models
cp backups/models_YYYYMMDD.tar.gz models/
tar -xzf models/models_YYYYMMDD.tar.gz

# Step 4: Restore previous database
cp backups/db_YYYYMMDD.db backend/quantum_trader.db

# Step 5: Start previous version
docker-compose up -d

# Step 6: Verify rollback success
python check_system_status.py
```

---

## 5. KONFIGURASJON OG TUNING

### 5.1 Risk Management Tuning

#### Conservative Profile (Low Risk)
```bash
# .env configuration
RM_MAX_POSITION_USD=1000          # Smaller positions
RM_MAX_LEVERAGE=10.0              # Lower leverage
RM_MAX_CONCURRENT_TRADES=10       # Fewer positions
RM_MAX_DAILY_DD_PCT=0.10          # 10% max daily DD
RM_RISK_PER_TRADE_PCT=0.01        # 1% risk per trade
QT_CONFIDENCE_THRESHOLD=0.80      # Higher confidence required
```

#### Moderate Profile (Balanced)
```bash
# .env configuration
RM_MAX_POSITION_USD=2000          # Default
RM_MAX_LEVERAGE=20.0              # Moderate leverage
RM_MAX_CONCURRENT_TRADES=15       # Default
RM_MAX_DAILY_DD_PCT=0.15          # 15% max daily DD
RM_RISK_PER_TRADE_PCT=0.02        # 2% risk per trade
QT_CONFIDENCE_THRESHOLD=0.70      # Default
```

#### Aggressive Profile (High Risk)
```bash
# .env configuration
RM_MAX_POSITION_USD=3000          # Larger positions
RM_MAX_LEVERAGE=30.0              # Max leverage
RM_MAX_CONCURRENT_TRADES=20       # More positions
RM_MAX_DAILY_DD_PCT=0.20          # 20% max daily DD
RM_RISK_PER_TRADE_PCT=0.05        # 5% risk per trade
QT_CONFIDENCE_THRESHOLD=0.65      # Lower confidence OK
```

### 5.2 AI Model Tuning

#### Enable/Disable Models
```bash
# Single model mode (fastest)
AI_MODEL=xgb

# Ensemble mode (best accuracy)
AI_MODEL=hybrid

# Custom ensemble weights
# Edit ai_engine/ensemble_manager.py:
weights = {
    'xgb': 0.30,      # Increase XGBoost weight
    'lgbm': 0.30,     # Increase LightGBM weight
    'nhits': 0.25,    # Reduce N-HiTS weight
    'patchtst': 0.15  # Reduce PatchTST weight
}
```

#### Confidence Threshold Tuning
```bash
# For more trades (but lower quality)
QT_CONFIDENCE_THRESHOLD=0.60

# For fewer, higher quality trades
QT_CONFIDENCE_THRESHOLD=0.80

# Adaptive threshold (by regime)
QT_POLICY_MIN_CONF_TRENDING=0.18
QT_POLICY_MIN_CONF_RANGING=0.22
QT_POLICY_MIN_CONF_NORMAL=0.20
```

### 5.3 Universe Selection Tuning

```bash
# Top 50 by volume (default)
QT_UNIVERSE=l1l2-top
QT_MAX_SYMBOLS=50

# Only megacap coins
QT_UNIVERSE=megacap
QT_MAX_SYMBOLS=20

# All available pairs
QT_UNIVERSE=all-usdt
QT_MAX_SYMBOLS=200

# Manual symbol list
QT_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT
```

### 5.4 Execution Parameters

```bash
# Fast execution (check every 5s)
QT_CHECK_INTERVAL=5
QT_COOLDOWN_SECONDS=60

# Moderate (default)
QT_CHECK_INTERVAL=10
QT_COOLDOWN_SECONDS=120

# Slow/conservative
QT_CHECK_INTERVAL=30
QT_COOLDOWN_SECONDS=300
```

---

## 6. MODEL MANAGEMENT

### 6.1 Manual Model Training

```bash
# Train all models
python backend/train_model.py --all

# Train specific model
python backend/train_model.py --model xgb
python backend/train_model.py --model lgbm
python backend/train_model.py --model nhits
python backend/train_model.py --model patchtst

# Training with custom parameters
python backend/train_model.py \
  --model xgb \
  --lookback-days 90 \
  --symbols 100 \
  --interval 5m
```

### 6.2 Model Evaluation

```bash
# Evaluate all models
python backend/domains/learning/evaluate_models.py

# Evaluate specific model
python test_model_performance.py --model xgb

# Backtest model
python backend/tools/backtest.py \
  --model xgb \
  --start-date 2025-11-01 \
  --end-date 2025-12-01
```

### 6.3 Model Promotion/Demotion

```bash
# Promote shadow model to production
curl -X POST http://localhost:8000/api/clm/promote \
  -H "Content-Type: application/json" \
  -d '{"model_id":"xgb_shadow_v123","reason":"Better performance"}'

# Demote model to shadow
curl -X POST http://localhost:8000/api/clm/demote \
  -H "Content-Type: application/json" \
  -d '{"model_id":"xgb_v122","reason":"Performance degradation"}'

# Rollback to previous version
python backend/tools/rollback_model.py --model xgb --version v120
```

### 6.4 Continuous Learning Configuration

```bash
# Enable/disable auto-retraining
QT_CLM_ENABLED=true
QT_CLM_AUTO_RETRAIN=true
QT_CLM_AUTO_PROMOTE=true

# Retraining schedule
QT_CLM_RETRAIN_HOURS=168  # Weekly (168h = 7 days)
QT_CLM_DRIFT_HOURS=24     # Check drift daily
QT_CLM_PERF_HOURS=6       # Check performance every 6h

# Drift detection threshold
QT_CLM_DRIFT_THRESHOLD=0.05  # Trigger retrain if drift > 5%

# Shadow testing
QT_CLM_SHADOW_MIN=100  # Min 100 predictions before promotion
```

---

## 7. RISK MANAGEMENT

### 7.1 Risk Limit Configuration

```bash
# Position limits
RM_MAX_POSITION_USD=2000        # Max margin per position
RM_MIN_POSITION_USD=100         # Min margin per position
RM_MAX_LEVERAGE=30.0            # Max leverage allowed

# Portfolio limits
RM_MAX_CONCURRENT_TRADES=20     # Max open positions
RM_MAX_EXPOSURE_PCT=2.00        # 200% max total exposure
RM_MAX_CORRELATION=0.7          # Max correlation between positions

# Drawdown limits
RM_MAX_DAILY_DD_PCT=0.15        # 15% max daily drawdown
RM_MAX_WEEKLY_DD_PCT=0.25       # 25% max weekly drawdown

# Per-trade risk
RM_RISK_PER_TRADE_PCT=0.02      # 2% base risk per trade
RM_MIN_RISK_PCT=0.01            # 1% minimum
RM_MAX_RISK_PCT=0.05            # 5% maximum
```

### 7.2 Emergency Stop System Configuration

```bash
# Enable/disable ESS
QT_ESS_ENABLED=true

# Evaluator thresholds
ESS_DRAWDOWN_THRESHOLD=0.15     # 15% daily DD triggers
ESS_EXECUTION_ERROR_THRESHOLD=5 # 5 consecutive errors triggers
ESS_DATA_FEED_TIMEOUT=300       # 5 min data feed outage triggers

# Actions on trigger
ESS_STOP_NEW_TRADES=true        # Stop new trades
ESS_CLOSE_LOSING=true           # Close losing positions
ESS_REDUCE_EXPOSURE=0.5         # Reduce exposure by 50%
ESS_ALERT_OPS=true              # Alert operators
```

### 7.3 Risk Monitoring

```bash
# Check current risk metrics
curl http://localhost:8000/api/risk/status | jq

# Check ESS status
curl http://localhost:8000/api/ess/status | jq

# Review risk events
curl http://localhost:8000/api/risk/events | jq

# Get risk report
python backend/tools/risk_report.py --period daily
```

---

## 8. BACKUP OG RECOVERY

### 8.1 Backup Strategy

#### Daglig Backup Script
```bash
#!/bin/bash
# daily_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/daily/$DATE"
mkdir -p $BACKUP_DIR

# 1. Database backup
cp backend/quantum_trader.db $BACKUP_DIR/db.sqlite

# 2. Models backup
tar -czf $BACKUP_DIR/models.tar.gz ai_engine/models/

# 3. Configuration backup
cp .env $BACKUP_DIR/env.backup

# 4. Logs backup
tar -czf $BACKUP_DIR/logs.tar.gz backend/logs/

# 5. Verify backups
ls -lh $BACKUP_DIR/

echo "✅ Daily backup complete: $BACKUP_DIR"
```

#### Ukentlig Backup Script
```bash
#!/bin/bash
# weekly_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/weekly/$DATE"
mkdir -p $BACKUP_DIR

# Full system backup
tar -czf $BACKUP_DIR/full_system.tar.gz \
  backend/ \
  ai_engine/ \
  frontend/ \
  .env \
  docker-compose.yml

# Rotate old backups (keep 4 weeks)
find /backups/weekly/ -type d -mtime +28 -exec rm -rf {} \;

echo "✅ Weekly backup complete: $BACKUP_DIR"
```

### 8.2 Recovery Procedures

#### Database Recovery
```bash
# Stop services
docker-compose down

# Restore database
cp /backups/daily/20251217/db.sqlite backend/quantum_trader.db

# Verify integrity
sqlite3 backend/quantum_trader.db "PRAGMA integrity_check;"

# Restart
docker-compose up -d
```

#### Model Recovery
```bash
# Restore models from backup
tar -xzf /backups/daily/20251217/models.tar.gz -C ai_engine/

# Verify models
ls -la ai_engine/models/

# Test load
python -c "from ai_engine.ensemble_manager import EnsembleManager; em = EnsembleManager()"

# Restart
docker-compose restart backend
```

#### Full System Recovery
```bash
# Extract full backup
tar -xzf /backups/weekly/20251215/full_system.tar.gz

# Restore configuration
cp .env.backup .env

# Rebuild containers
docker-compose build --no-cache

# Start system
docker-compose up -d

# Verify
python check_system_status.py
```

---

## VEDLEGG: QUICK REFERENCE

### Environment Variables Quick Reference

```bash
# Exchange
BINANCE_API_KEY=***
BINANCE_API_SECRET=***
BINANCE_TESTNET=true

# AI/ML
AI_MODEL=hybrid
QT_CONFIDENCE_THRESHOLD=0.70
QT_CLM_ENABLED=true

# Risk
RM_MAX_POSITION_USD=2000
RM_MAX_LEVERAGE=30.0
RM_MAX_DAILY_DD_PCT=0.15

# Execution
QT_EVENT_DRIVEN_MODE=true
QT_CHECK_INTERVAL=10
QT_COOLDOWN_SECONDS=120
```

### API Endpoints Quick Reference

```bash
# Health
GET  /api/health
GET  /api/v2/health

# Positions
GET  /api/positions
POST /api/trades/close-all

# Signals
GET  /api/signals
GET  /api/signals/recent

# Risk
GET  /api/risk/status
POST /api/ess/activate
POST /api/ess/reset

# AI
GET  /api/ai/models/status
POST /api/clm/trigger-retrain

# Stats
GET  /api/stats/daily
GET  /api/stats/weekly

# Executor
GET  /api/executor/status
POST /api/executor/start
POST /api/executor/stop
```

---

**Dokumentert av:** GitHub Copilot  
**Dato:** 17. desember 2025  
**Versjon:** 1.0  
**Status:** ✅ KOMPLETT
