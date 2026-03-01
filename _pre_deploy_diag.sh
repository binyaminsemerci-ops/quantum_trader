#!/bin/bash
# Pre-deployment diagnostic: find process manager for services
echo "=== PROCESS MANAGER DETECTION ==="
which supervisorctl 2>/dev/null && echo "supervisord found" || echo "no supervisorctl"
which pm2 2>/dev/null && echo "pm2 found" || echo "no pm2"
ls /etc/supervisor/conf.d/ 2>/dev/null && echo "supervisor conf.d present" || true
ls /etc/supervisord.conf 2>/dev/null || true

echo ""
echo "=== SUPERVISOR STATUS if available ==="
supervisorctl status 2>/dev/null | head -30 || true

echo ""
echo "=== USER SYSTEMD ==="
sudo -u qt XDG_RUNTIME_DIR=/run/user/$(id -u qt) systemctl --user list-units --type=service 2>/dev/null | grep -E "intent|apply|ensemble|harvest" | head -20 || true

echo ""
echo "=== SYSTEM SYSTEMD (active services matching quantum) ==="
systemctl list-units --type=service --state=active 2>/dev/null | grep -E "intent|apply|ensemble|harvest|quantum" | head -30 || true

echo ""
echo "=== INTENT_EXECUTOR process environ ==="
IE_PID=$(ps aux | grep intent_executor | grep -v grep | awk '{print $2}' | head -1)
echo "IE PID: $IE_PID"
if [ -n "$IE_PID" ]; then
    cat /proc/$IE_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "ALLOWLIST\|SOURCE\|SYNTH\|RISK\|HARVEST" | head -20
fi

echo ""
echo "=== APPLY_LAYER process environ ==="
AL_PID=$(ps aux | grep apply_layer | grep -v grep | awk '{print $2}' | head -1)
echo "AL PID: $AL_PID"
if [ -n "$AL_PID" ]; then
    cat /proc/$AL_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "ALLOWLIST\|SOURCE\|SYNTH\|RISK\|HARVEST\|K_OPEN\|K_CLOSE\|DAILY_LOSS" | head -20
fi

echo ""
echo "=== ENSEMBLE_PREDICTOR process environ ==="
EP_PID=$(ps aux | grep ensemble_predictor | grep -v grep | awk '{print $2}' | head -1)
echo "EP PID: $EP_PID"
if [ -n "$EP_PID" ]; then
    cat /proc/$EP_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "SYNTH\|MODE\|ENSEMBLE\|CONFIDENCE" | head -20
fi

echo ""
echo "=== HARVEST_V2 process environ ==="
HV2_PID=$(ps aux | grep harvest_v2 | grep -v grep | awk '{print $2}' | head -1)
echo "HV2 PID: $HV2_PID"
if [ -n "$HV2_PID" ]; then
    cat /proc/$HV2_PID/environ 2>/dev/null | tr '\0' '\n' | grep -E "HARVEST\|STREAM\|CONFIG\|R_" | head -20
fi

echo ""
echo "=== Find .env files for services ==="
find /opt/quantum -name "*.env" 2>/dev/null | head -20
find /home/qt -name "*.env" 2>/dev/null | head -20
find /etc -name "quantum*" 2>/dev/null | head -10

echo ""
echo "=== Check supervisor conf.d files ==="
ls /etc/supervisor/conf.d/ 2>/dev/null
cat /etc/supervisor/conf.d/intent_executor.conf 2>/dev/null || cat /etc/supervisor/conf.d/intent-executor.conf 2>/dev/null || true
