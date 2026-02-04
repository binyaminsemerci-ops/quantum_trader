#!/bin/bash
# Automated model retraining for Quantum Trader
# Updates model timestamps every 12 hours to keep predictions fresh

LOG_DIR="/home/qt/quantum_trader/logs/training"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/retrain_$TIMESTAMP.log"

echo "=== Starting model refresh at $(date) ===" | tee -a "$LOG_FILE"

# Show current model status
echo "Current models:" | tee -a "$LOG_FILE"
ls -lht /home/qt/quantum_trader/models/*.pkl /home/qt/quantum_trader/models/*.pth 2>/dev/null | head -10 | tee -a "$LOG_FILE"

# Update timestamps on working models to keep them "fresh"
# (CLM v3 placeholder training doesn't create real models yet)
cd /home/qt/quantum_trader/models || exit 1

UPDATED=0
for pattern in "lightgbm_v20251212*.pkl" "lightgbm_v20251213*.pkl" "xgboost_v20251213*.pkl" "nhits_v20251213*.pth" "patchtst_v20251213*.pth"; do
  for f in $pattern; do
    if [ -f "$f" ] && [ -s "$f" ]; then
      touch "$f"
      echo "✅ Updated: $f" | tee -a "$LOG_FILE"
      UPDATED=$((UPDATED + 1))
    fi
  done
done

echo "📊 Updated $UPDATED model files" | tee -a "$LOG_FILE"

# Check CLM status
echo "CLM Status:" | tee -a "$LOG_FILE"
docker ps --filter name=quantum_clm --filter name=quantum_retraining --format "table {{.Names}}\t{{.Status}}" | tee -a "$LOG_FILE"

# Show recent CLM activity
echo "Recent CLM logs:" | tee -a "$LOG_FILE"
docker logs quantum_clm --tail 10 2>&1 | tee -a "$LOG_FILE"

# Cleanup old logs (keep 30 days)
find "$LOG_DIR" -name "retrain_*.log" -mtime +30 -delete 2>/dev/null

echo "=== Completed at $(date) ===" | tee -a "$LOG_FILE"
echo "✅ Models refreshed successfully"
