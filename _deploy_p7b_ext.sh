#!/usr/bin/env bash
# PATCH-7B-ext deploy script — run on VPS as root
set -e

# Step 1: secrets file
cat > /etc/quantum/exit-agent-secrets.env << 'SECRETS_EOF'
# Exit Management Agent — external inference secrets
# chmod 600 this file — NEVER commit to git
EXIT_AGENT_QWEN3_API_KEY=__SET_GROQ_API_KEY_HERE__
SECRETS_EOF
chmod 600 /etc/quantum/exit-agent-secrets.env
echo "[1] secrets.env: $(stat -c 'perms=%a owner=%U' /etc/quantum/exit-agent-secrets.env)"

# Step 2: patch service file
SERVICE=/etc/systemd/system/quantum-exit-management-agent.service
cp "$SERVICE" "${SERVICE}.bak.p7b_ext.$(date +%s)"

# Replace Ollama endpoint + model + timeout, add EnvironmentFile
sed -i \
  's|EXIT_AGENT_QWEN3_ENDPOINT=http://localhost:11434|EXIT_AGENT_QWEN3_ENDPOINT=https://api.groq.com/openai/v1|' \
  "$SERVICE"
sed -i \
  's|EXIT_AGENT_QWEN3_MODEL=qwen3:0.6b|EXIT_AGENT_QWEN3_MODEL=meta-llama/llama-4-scout-17b-16e-instruct|' \
  "$SERVICE"
sed -i \
  's|EXIT_AGENT_QWEN3_TIMEOUT_MS=2000|EXIT_AGENT_QWEN3_TIMEOUT_MS=5000|' \
  "$SERVICE"

# Add EnvironmentFile line after [Service] section start (after User=qt line)
if ! grep -q 'exit-agent-secrets.env' "$SERVICE"; then
  sed -i '/^User=qt/a EnvironmentFile=-/etc/quantum/exit-agent-secrets.env' "$SERVICE"
fi

echo "[2] Service file patched. Verifying key lines:"
grep -E 'EnvironmentFile|QWEN3_ENDPOINT|QWEN3_MODEL|QWEN3_TIMEOUT|QWEN3_SHADOW' "$SERVICE"
