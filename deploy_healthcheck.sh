#!/usr/bin/env bash
# Deploy production healthcheck with rate-limiting
# Run from WSL: bash deploy_healthcheck.sh

set -euo pipefail

echo "=== Deploying Healthcheck to VPS ==="

# A) Webhook config
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo mkdir -p /etc/quantum'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo tee /etc/quantum/alert.env >/dev/null' <<'EOF'
# Sett webhook her (Slack/Discord/etc)
ALERT_WEBHOOK_URL=""
EOF
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo chmod 600 /etc/quantum/alert.env'
echo "✅ Webhook config created"

# B) Core healthcheck
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo mkdir -p /opt/quantum/ops'
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo tee /opt/quantum/ops/healthcheck.sh >/dev/null' <<'EOF'
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
EOF
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo chmod +x /opt/quantum/ops/healthcheck.sh'
echo "✅ Healthcheck script installed"

# C) Rate-limited cron wrapper
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo tee /opt/quantum/ops/healthcheck_cron.sh >/dev/null' <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

LOG="/var/log/quantum-healthcheck.log"
LOCK="/tmp/quantum-healthcheck.alert.lock"
COOLDOWN=600

mkdir -p /var/log

if /opt/quantum/ops/healthcheck.sh >>"$LOG" 2>&1; then
  echo "$(date -Is) OK" >>"$LOG"
  exit 0
fi

echo "$(date -Is) FAIL" >>"$LOG"

now=$(date +%s)
last=0
[[ -f "$LOCK" ]] && last=$(cat "$LOCK" 2>/dev/null || echo 0)

if (( now - last >= COOLDOWN )); then
  echo "$now" >"$LOCK"
  /opt/quantum/ops/healthcheck.sh >>"$LOG" 2>&1 || true
fi

exit 1
EOF
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo chmod +x /opt/quantum/ops/healthcheck_cron.sh'
echo "✅ Cron wrapper installed"

# D) Cron every 5 minutes
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo tee /etc/cron.d/quantum-healthcheck >/dev/null' <<'EOF'
*/5 * * * * root /opt/quantum/ops/healthcheck_cron.sh
EOF
echo "✅ Cron job configured"

echo ""
echo "=== Testing Healthcheck ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'sudo /opt/quantum/ops/healthcheck.sh; echo "Exit: $?"'

echo ""
echo "=== Recent Logs ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'tail -40 /var/log/quantum-healthcheck.log 2>/dev/null || echo "No logs yet"'

echo ""
echo "=== Cron Status ==="
ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cat /etc/cron.d/quantum-healthcheck'
