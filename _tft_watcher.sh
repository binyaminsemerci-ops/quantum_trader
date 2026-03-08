#!/bin/bash
# TFT v10b completion watcher
# Monitors training PID, updates tft_model.pth symlink and restarts ai-engine

PID=2276768
LOG=/var/log/quantum/tft_v10_train.log
WLOG=/var/log/quantum/tft_watcher.log
MODELS=/opt/quantum/ai_engine/models

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started. Monitoring PID=$PID" | tee -a $WLOG

while kill -0 $PID 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training PID $PID finished" | tee -a $WLOG

BEST_LINE=$(grep '\[BEST\]' $LOG 2>/dev/null | tail -1)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] $BEST_LINE" | tee -a $WLOG

NEWEST=$(ls -t $MODELS/tft_v10*.pth 2>/dev/null | head -1)
if [ -z "$NEWEST" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No tft_v10*.pth found in $MODELS" | tee -a $WLOG
    exit 1
fi

BASENAME=$(basename $NEWEST)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating symlink: tft_model.pth -> $BASENAME" | tee -a $WLOG
cd $MODELS && ln -sf $BASENAME tft_model.pth
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Symlink updated" | tee -a $WLOG

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting quantum-ai-engine..." | tee -a $WLOG
systemctl restart quantum-ai-engine
sleep 20

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Service restart complete. Ensemble status:" | tee -a $WLOG
journalctl -u quantum-ai-engine -n 50 --no-pager | grep -iE 'ACTIVATED|Ensemble loaded|weight:|TFT|ERROR' | tee -a $WLOG
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done." | tee -a $WLOG