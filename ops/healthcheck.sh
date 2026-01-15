#!/usr/bin/env bash
set -euo pipefail

TRAIN_TMR="quantum-training-worker.timer"
CONTRACT_TMR="quantum-contract-check.timer"
MODELS_DIR="/data/quantum/models"
LATEST_JSON="${MODELS_DIR}/latest.json"
PY="/opt/quantum/venvs/ai-engine/bin/python3"
ALERT_ENV="/etc/quantum/alert.env"

fail=0
ok(){ echo "OK: $*"; }
bad(){ echo "FAIL: $*"; fail=1; }

# load webhook
if [[ -f "$ALERT_ENV" ]]; then source "$ALERT_ENV" || true; fi

post_alert() {
  local msg="$1"
  [[ -z "${ALERT_WEBHOOK_URL:-}" ]] && return 0
  curl -fsS -m 6 -H "Content-Type: application/json" \
    -d "{\"text\":\"${msg//\"/\\\"}\"}" \
    "$ALERT_WEBHOOK_URL" >/dev/null 2>&1 || true
}

# timers
for u in "$TRAIN_TMR" "$CONTRACT_TMR"; do
  systemctl is-enabled --quiet "$u" && systemctl is-active --quiet "$u" \
    && ok "$u enabled+active" || bad "$u not enabled/active"
done

# redis
systemctl is-active --quiet redis-server.service && ok "redis active" || bad "redis not active"

# disk
df -h "$MODELS_DIR" | head -2 || true
free_gb="$(df -BG --output=avail "$MODELS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
[[ "${free_gb:-0}" -ge 5 ]] && ok "disk ok (${free_gb}G free)" || bad "low disk (${free_gb}G free)"

# artifact schema + exit_code
if [[ -f "$LATEST_JSON" ]]; then
  "$PY" - <<'PY' || exit 1
import json,sys
d=json.load(open("/data/quantum/models/latest.json"))
for k in ("git_commit","exit_code","duration_seconds","hostname"):
    if k not in d: print("missing",k); sys.exit(2)
if d.get("exit_code")!=0: print("exit_code",d.get("exit_code")); sys.exit(3)
print("git_commit",d["git_commit"],"exit_code",d["exit_code"])
PY
  ok "latest.json valid"
else
  bad "missing latest.json"
fi

# contract tests
"$PY" /opt/quantum/tests/orchestrator_contract_test_clean.py >/dev/null 2>&1 \
  && ok "contract tests pass" || bad "contract tests fail"

if [[ "$fail" -eq 0 ]]; then
  exit 0
else
  post_alert "❌ Quantum healthcheck FAIL on $(hostname) at $(date -Is). See /var/log/quantum-healthcheck.log"
  exit 1
fi
