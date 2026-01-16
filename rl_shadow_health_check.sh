#!/bin/bash
# RL Shadow System Health Check

ALERT_LOG="/var/log/quantum/rl_shadow_health.log"
ERRORS=0

echo "[$(date -Iseconds)] RL Shadow Health Check" >> "$ALERT_LOG"

# 1) Check RL Policy Publisher
if ! systemctl is-active --quiet quantum-rl-policy-publisher.service; then
  echo "  ❌ RL Policy Publisher is NOT active" >> "$ALERT_LOG"
  ((ERRORS++))
else
  echo "  ✅ RL Policy Publisher active" >> "$ALERT_LOG"
fi

# 2) Check AI Engine
if ! systemctl is-active --quiet quantum-ai-engine.service; then
  echo "  ❌ AI Engine is NOT active" >> "$ALERT_LOG"
  ((ERRORS++))
else
  echo "  ✅ AI Engine active" >> "$ALERT_LOG"
fi

# 3) Check Scorecard log freshness (should update every 15 min)
if [ -f /var/log/quantum/rl_shadow_scorecard.log ]; then
  LOG_AGE=$(( $(date +%s) - $(stat -c %Y /var/log/quantum/rl_shadow_scorecard.log) ))
  if [ $LOG_AGE -gt 1200 ]; then  # 20 minutes
    echo "  ⚠️  Scorecard log is stale (${LOG_AGE}s old)" >> "$ALERT_LOG"
    ((ERRORS++))
  else
    echo "  ✅ Scorecard log fresh (${LOG_AGE}s old)" >> "$ALERT_LOG"
  fi
else
  echo "  ❌ Scorecard log missing" >> "$ALERT_LOG"
  ((ERRORS++))
fi

# Summary
if [ $ERRORS -eq 0 ]; then
  echo "  ✅ All checks passed" >> "$ALERT_LOG"
else
  echo "  ⚠️  $ERRORS checks failed" >> "$ALERT_LOG"
fi

echo "" >> "$ALERT_LOG"

# Keep log at reasonable size (last 500 lines)
tail -500 "$ALERT_LOG" > "$ALERT_LOG.tmp" && mv "$ALERT_LOG.tmp" "$ALERT_LOG"
