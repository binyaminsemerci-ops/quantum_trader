#!/bin/bash
set -euo pipefail

ALERT_LOG="/var/log/quantum/rl_shadow_health.log"
TS="$(date -Iseconds)"

pub="$(systemctl is-active quantum-rl-policy-publisher.service || true)"
aie="$(systemctl is-active quantum-ai-engine.service || true)"
timer="$(systemctl is-active quantum-rl-shadow-scorecard.timer || true)"

log_age="NA"
if [ -f /var/log/quantum/rl_shadow_scorecard.log ]; then
  now="$(date +%s)"
  mtime="$(stat -c %Y /var/log/quantum/rl_shadow_scorecard.log)"
  log_age="$((now - mtime))"
fi

echo "[$TS] publisher=$pub ai_engine=$aie scorecard_timer=$timer scorecard_log_age_sec=$log_age" >> "$ALERT_LOG"
