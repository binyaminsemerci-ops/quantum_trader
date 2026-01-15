#!/usr/bin/env bash
set -u -o pipefail

LOG_FILE="/opt/quantum/rl/logs/rl_diag_auto.log"
API_ENDPOINT="${API_ENDPOINT:-}"
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
BLUE="\033[34m"
RESET="\033[0m"

log() { printf "%b\n" "$1"; }
status_line() {
  local label="$1"; shift
  local val="$1"; shift
  case "$val" in
    active) log "${GREEN}${label}: ${val}${RESET}" ;;
    inactive|failed) log "${RED}${label}: ${val}${RESET}" ;;
    *) log "${YELLOW}${label}: ${val}${RESET}" ;;
  esac
}

mkdir -p "$(dirname "$LOG_FILE")"

(
  log "${BLUE}=== Quantum RL Verify $(date)${RESET}"

  monitor_status=$(systemctl is-active quantum-rl-monitor 2>/dev/null || true)
  trainer_status=$(systemctl is-active quantum-rl-trainer 2>/dev/null || true)
  agent_status=$(systemctl is-active quantum-rl-agent 2>/dev/null || true)

  status_line "Monitor" "${monitor_status:-unknown}"
  status_line "Trainer" "${trainer_status:-unknown}"
  status_line "Agent" "${agent_status:-unknown}"

  experience_count=$(redis-cli XLEN quantum:rl:experience 2>/dev/null || echo 0)
  reward_count=$(redis-cli XLEN quantum:rl:reward 2>/dev/null || echo 0)
  log "Experience entries: ${experience_count}"
  log "Reward entries: ${reward_count}"

  metrics_output="$(python3 /opt/quantum/metrics/rl_metrics.py 2>&1 || true)"
  log "Metrics: ${metrics_output}"

  if [ -f /opt/quantum/rl/policies/rl_policy.pt ]; then
    log "Policy file: $(ls -lh /opt/quantum/rl/policies/rl_policy.pt)"
    log "Policy sha256: $(sha256sum /opt/quantum/rl/policies/rl_policy.pt | awk '{print $1}')"
  else
    log "Policy file missing: /opt/quantum/rl/policies/rl_policy.pt"
  fi

  metrics_sanitized=${metrics_output//"/\\"}
  payload=$(printf '{"timestamp":"%s","monitor_status":"%s","trainer_status":"%s","agent_status":"%s","experience_count":%s,"reward_count":%s,"metrics":"%s"}' \
    "$TIMESTAMP" "${monitor_status:-unknown}" "${trainer_status:-unknown}" "${agent_status:-unknown}" \
    "${experience_count:-0}" "${reward_count:-0}" "$metrics_sanitized")
  log "Payload: ${payload}"

  if [ -n "$API_ENDPOINT" ]; then
    if curl -sS -X POST -H "Content-Type: application/json" -d "$payload" "$API_ENDPOINT" >/tmp/rl_verify_curl.log 2>&1; then
      log "${GREEN}JSON metrics posted to Grafana API${RESET}"
    else
      log "${RED}Failed to post metrics to API${RESET}"
      cat /tmp/rl_verify_curl.log
    fi
  else
    log "API_ENDPOINT not set; skipping POST"
  fi

  log "Verification complete. Log saved to ${LOG_FILE}"
) | tee -a "$LOG_FILE"
