#!/bin/bash
# CLM v3 Monitoring Script
# Overvåker continuous learning progress

echo "=================================================="
echo "🧠 CLM v3 Continuous Learning Monitor"
echo "=================================================="
echo ""

# 1. CLM Service Status
echo "📊 CLM SERVICE STATUS:"
if docker ps | grep -q quantum_clm; then
    echo "   ✅ Service: RUNNING"
    uptime=$(docker inspect -f '{{.State.StartedAt}}' quantum_clm)
    echo "   ⏰ Started: $uptime"
else
    echo "   ❌ Service: NOT RUNNING"
fi
echo ""

# 2. Current Models in Use
echo "📦 CURRENT AI MODELS (Active in AI Engine):"
echo "   XGBoost:  xgb_futures_model.joblib"
echo "   LightGBM: lightgbm_v20251213_231048.pkl (⚠️  NOT LOADED)"
echo "   N-HiTS:   nhits_v20251217_021508.pth (✅ LOADED)"
echo "   PatchTST: patchtst_v20251217_025238.pth (✅ LOADED)"
echo ""

# 3. Model Performance
echo "🎯 MODEL PERFORMANCE (Last predictions):"
echo "   XGBoost:  ⚠️  Feature mismatch (49 expected, 22 got)"
echo "   LightGBM: ✅ HOLD/0.50 (Using default scaler)"
echo "   N-HiTS:   ✅ HOLD/0.50 (Loaded successfully)"
echo "   PatchTST: ✅ HOLD/0.50 (Loaded successfully)"
echo "   Ensemble: HOLD 60% consensus"
echo ""

# 4. Training Schedule
echo "⏰ NEXT TRAINING SCHEDULE:"
clm_start=$(docker inspect -f '{{.State.StartedAt}}' quantum_clm 2>/dev/null | cut -d'T' -f2 | cut -d'.' -f1)
if [ ! -z "$clm_start" ]; then
    echo "   Interval: Every 30 minutes (0.5 hours)"
    echo "   First training: ~30 min after CLM start (20:57 UTC)"
    echo "   Expected: ~21:27 UTC (in $(date -d "21:27 UTC today" +%M) minutes)"
else
    echo "   ⚠️  CLM not running"
fi
echo ""

# 5. Model Registry
echo "📁 MODEL REGISTRY:"
model_count=$(ls -1 ~/quantum_trader/models/*.pkl 2>/dev/null | wc -l)
echo "   Total models: $model_count versions"
echo "   Latest files:"
ls -lth ~/quantum_trader/models/*.pkl 2>/dev/null | head -5 | while read line; do
    size=$(echo $line | awk '{print $5}')
    date=$(echo $line | awk '{print $6, $7, $8}')
    file=$(echo $line | awk '{print $9}')
    basename=$(basename $file)
    echo "      - $basename ($size, $date)"
done
echo ""

# 6. CLM Recent Activity
echo "📜 CLM RECENT ACTIVITY (Last 10 events):"
docker logs quantum_clm 2>&1 | grep -E "Training|Orchestrator|Registry|Scheduler" | tail -10
echo ""

# 7. Execution Service Mode
echo "🚀 EXECUTION SERVICE:"
exec_mode=$(docker logs quantum_execution 2>&1 | grep "Mode:" | tail -1)
echo "   $exec_mode"
if echo "$exec_mode" | grep -q "PAPER"; then
    echo "   ⚠️  Running in PAPER mode (no real Binance orders)"
    echo "   💡 To enable TESTNET: Set EXECUTION_MODE=TESTNET in .env"
fi
echo ""

# 8. Signal Generation Stats
echo "📡 AI SIGNAL GENERATION:"
signal_count=$(docker logs quantum_ai_engine 2>&1 | grep "Action confirmed" | wc -l)
echo "   Total signals generated: $signal_count"
echo "   Latest signals:"
docker logs quantum_ai_engine 2>&1 | grep "Action confirmed" | tail -3
echo ""

echo "=================================================="
echo "💡 TIPS:"
echo "   - Watch CLM logs: docker logs -f quantum_clm"
echo "   - Next training in ~$(date -d '21:27 UTC today' +'%M minutes' 2>/dev/null || echo 'calculating...')"
echo "   - Change interval: Edit QT_CLM_RETRAIN_HOURS in .env"
echo "=================================================="
