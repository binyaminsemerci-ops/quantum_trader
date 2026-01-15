#!/usr/bin/env bash
set -euo pipefail

OUT_LOG="/opt/quantum/ensemble/logs/ensemble_health_auto.log"
API_ENDPOINT="http://localhost:3000/api/quantum/ensemble/metrics"
DATE_STR=$(date '+%Y-%m-%d %H:%M:%S')

# Colors
GREEN="\033[1;32m"; YELLOW="\033[1;33m"; RED="\033[1;31m"; BLUE="\033[1;34m"; RESET="\033[0m"

mkdir -p "$(dirname "$OUT_LOG")"

echo -e "${BLUE}=== Quantum Ensemble Health Verify ($DATE_STR) ===${RESET}"

# Locate latest Proof Pack V3
LOG_PATH=$(ls -t /opt/quantum/reports/ENSEMBLE_RECOVERY_V3_* 2>/dev/null | head -n 1 || true)
if [[ -z "$LOG_PATH" ]]; then
  echo -e "${RED}No Proof Pack V3 report found.${RESET}"
  exit 1
fi
echo -e "Using report: ${YELLOW}$LOG_PATH${RESET}"

# Extract key metrics
VOTING_HEALTH=$(grep -oP 'Voting Health\s*=\s*\K[0-9\.]+' "$LOG_PATH" | tail -n1)
ACTIVE_MODELS=$(grep -oP 'Active Models\s*=\s*\K[0-9]+' "$LOG_PATH" | tail -n1)
CONF_SPREAD=$(grep -oP 'Confidence Spread\s*=\s*\K[0-9\.]+' "$LOG_PATH" | tail -n1)
MODEL_HASH=$(grep -oP 'Model Hash\s*=\s*\K[\w\d]+' "$LOG_PATH" | tail -n1)

echo -e "🧠 Voting Health: ${GREEN}${VOTING_HEALTH:-N/A}${RESET}"
echo -e "📦 Active Models: ${YELLOW}${ACTIVE_MODELS:-N/A}${RESET}"
echo -e "📊 Confidence Spread: ${GREEN}${CONF_SPREAD:-N/A}${RESET}"
echo -e "🔐 Model Hash: ${BLUE}${MODEL_HASH:-N/A}${RESET}"

# Build JSON payload (avoid non-zero read exit under set -e)
JSON_PAYLOAD=$(cat <<EOF_JSON
{
  "timestamp": "$DATE_STR",
  "voting_health": ${VOTING_HEALTH:-0.0},
  "active_models": ${ACTIVE_MODELS:-0},
  "confidence_spread": ${CONF_SPREAD:-0.0},
  "model_hash": "$MODEL_HASH"
}
EOF_JSON
)

# Write log
{
  echo "=== Quantum Ensemble Health Verify ($DATE_STR) ==="
  echo "$JSON_PAYLOAD"
} > "$OUT_LOG"

# Post to Grafana API
if curl -s -X POST -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "$API_ENDPOINT" >/dev/null 2>&1; then
  echo -e "${GREEN}✅ Ensemble health metrics posted to Grafana API${RESET}"
else
  echo -e "${YELLOW}⚠️  Failed to post to Grafana API${RESET}"
fi

echo -e "${BLUE}Verification complete. Log saved to $OUT_LOG${RESET}"
