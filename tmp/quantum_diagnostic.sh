#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/opt/quantum/audit/quantum_diagnostic.log"
API_ENDPOINT="https://app.quantumfond.com/grafana/api/quantum/core/diagnostics"
DATE_STR=$(date '+%Y-%m-%d %H:%M:%S')
GREEN="\033[1;32m"; RED="\033[1;31m"; YELLOW="\033[1;33m"; BLUE="\033[1;34m"; RESET="\033[0m"

echo -e "${BLUE}=== Quantum Diagnostic Check ($DATE_STR) ===${RESET}"

# --- Collect statuses ---
redis_status=$(systemctl is-active redis.service || true)
agent_status=$(systemctl is-active quantum-rl-agent.service || true)
trainer_status=$(systemctl is-active quantum-rl-trainer.service || true)
monitor_status=$(systemctl is-active quantum-rl-monitor.service || true)
policy_sync_status=$(systemctl is-active quantum-policy-sync.timer || true)
ensemble_verify_status=$(systemctl is-active quantum-verify-ensemble.timer || true)
rl_verify_status=$(systemctl is-active quantum-verify-rl.timer || true)
core_health_status=$(systemctl is-active quantum-core-health.timer || true)

# --- Gather logs ---
policy_tail=$(tail -n 2 /opt/quantum/ensemble/logs/policy_sync.log 2>/dev/null | tail -n 1)
ensemble_tail=$(tail -n 2 /opt/quantum/ensemble/logs/ensemble_health_auto.log 2>/dev/null | tail -n 1)
rl_tail=$(tail -n 2 /opt/quantum/rl/logs/rl_diag_auto.log 2>/dev/null | tail -n 1)

# --- Redis data ---
reward_len=$(redis-cli XLEN quantum:rl:reward 2>/dev/null || echo 0)
exp_len=$(redis-cli XLEN quantum:rl:experience 2>/dev/null || echo 0)

# --- Color print summary ---
echo -e "🧩 Redis: ${GREEN}${redis_status}${RESET} (${reward_len} rewards / ${exp_len} experiences)"
echo -e "🤖 RL Agent: ${GREEN}${agent_status}${RESET}"
echo -e "🧠 RL Trainer: ${GREEN}${trainer_status}${RESET}"
echo -e "📡 RL Monitor: ${GREEN}${monitor_status}${RESET}"
echo -e "⚙️ Policy Sync: ${GREEN}${policy_sync_status}${RESET}"
echo -e "📊 RL Verify Timer: ${GREEN}${rl_verify_status}${RESET}"
echo -e "📦 Ensemble Verify Timer: ${GREEN}${ensemble_verify_status}${RESET}"
echo -e "🩺 Core Health Timer: ${GREEN}${core_health_status}${RESET}"

# --- Build JSON payload ---
payload=$(cat <<EOF
{
  "timestamp": "$DATE_STR",
  "redis_status": "$redis_status",
  "agent_status": "$agent_status",
  "trainer_status": "$trainer_status",
  "monitor_status": "$monitor_status",
  "policy_sync_status": "$policy_sync_status",
  "ensemble_verify_status": "$ensemble_verify_status",
  "rl_verify_status": "$rl_verify_status",
  "core_health_status": "$core_health_status",
  "reward_len": $reward_len,
  "experience_len": $exp_len,
  "policy_tail": "${policy_tail//\"/\\\"}",
  "ensemble_tail": "${ensemble_tail//\"/\\\"}",
  "rl_tail": "${rl_tail//\"/\\\"}"
}
EOF
)

# --- Log + POST ---
echo "$payload" >> "$LOG_FILE"
if curl -s -X POST -H "Content-Type: application/json" -d "$payload" "$API_ENDPOINT" >/dev/null 2>&1; then
  echo -e "${GREEN}✅ Diagnostics posted to Grafana${RESET}"
else
  echo -e "${YELLOW}⚠️ Failed to post diagnostics${RESET}"
fi
echo -e "${BLUE}Diagnostics complete. Log saved to $LOG_FILE${RESET}"
