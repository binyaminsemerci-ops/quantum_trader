#!/usr/bin/env bash
#
# Policy Refresh - Atomic Policy Generation with Validation
# ==========================================================
#
# Generates fresh AI policy, validates schema, writes atomically to Redis.
# Logs success/failure for monitoring.
#
# Usage:
#   bash scripts/policy_refresh.sh

set -euo pipefail

# Logging
LOG_PREFIX="[POLICY-REFRESH]"
log_info() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} INFO: $1"
}

log_error() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} ERROR: $1" >&2
}

log_ok() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} ✅ OK: $1"
}

log_fail() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') ${LOG_PREFIX} ❌ FAIL: $1" >&2
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AI_GENERATOR_SCRIPT="$SCRIPT_DIR/ai_universe_generator_v1.py"
FALLBACK_GENERATOR="$SCRIPT_DIR/generate_sample_policy.py"
TEMP_FILE="/tmp/policy_refresh_$$.json"

# Select generator (prefer AI, fallback to sample if AI unavailable)
if [ -f "$AI_GENERATOR_SCRIPT" ]; then
    GENERATOR_SCRIPT="$AI_GENERATOR_SCRIPT"
    log_info "Using AI universe generator: $AI_GENERATOR_SCRIPT"
else
    GENERATOR_SCRIPT="$FALLBACK_GENERATOR"
    log_info "AI generator not found, using fallback: $FALLBACK_GENERATOR"
fi

# Validate generator exists
if [ ! -f "$GENERATOR_SCRIPT" ]; then
    log_fail "Generator script not found: $GENERATOR_SCRIPT"
    exit 2
fi

# Step 1: Generate policy
log_info "Generating fresh policy..."
if python3 "$GENERATOR_SCRIPT" > /dev/null 2>&1; then
    log_ok "Policy generated successfully"
else
    log_fail "Policy generation failed (exit code: $?)"
    exit 2
fi

# Step 2: Validate policy exists in Redis
log_info "Validating policy in Redis..."
POLICY_VERSION=$(redis-cli HGET quantum:policy:current policy_version 2>/dev/null || echo "")
POLICY_HASH=$(redis-cli HGET quantum:policy:current policy_hash 2>/dev/null || echo "")
VALID_UNTIL=$(redis-cli HGET quantum:policy:current valid_until_epoch 2>/dev/null || echo "")

if [ -z "$POLICY_VERSION" ] || [ -z "$POLICY_HASH" ] || [ -z "$VALID_UNTIL" ]; then
    log_fail "Policy validation failed - missing required fields"
    log_error "  policy_version: ${POLICY_VERSION:-MISSING}"
    log_error "  policy_hash: ${POLICY_HASH:-MISSING}"
    log_error "  valid_until_epoch: ${VALID_UNTIL:-MISSING}"
    exit 2
fi

# Step 3: Validate expiry time (must be in future)
NOW=$(date +%s)
VALID_UNTIL_INT=$(echo "$VALID_UNTIL" | cut -d'.' -f1)  # Strip decimals for bash comparison
if [ "$VALID_UNTIL_INT" -le "$NOW" ]; then
    log_fail "Policy expired: valid_until=$VALID_UNTIL_INT now=$NOW"
    exit 2
fi

REMAINING_SEC=$((VALID_UNTIL_INT - NOW))
REMAINING_MIN=$((REMAINING_SEC / 60))

log_ok "Policy validated: version=$POLICY_VERSION hash=${POLICY_HASH:0:8} valid_for=${REMAINING_MIN}min"

# Log structured data for auditing
log_info "POLICY_STATE: version=$POLICY_VERSION hash=$POLICY_HASH valid_until=$VALID_UNTIL_INT remaining_sec=$REMAINING_SEC"

# Step 4: Verify universe count
UNIVERSE_COUNT=$(redis-cli HGET quantum:policy:current universe_symbols 2>/dev/null | python3 -c "import sys, json; print(len(json.loads(sys.stdin.read())))" 2>/dev/null || echo "0")
if [ "$UNIVERSE_COUNT" -eq 0 ]; then
    log_fail "Policy universe empty"
    exit 2
fi

log_ok "Policy universe: $UNIVERSE_COUNT symbols"

# Success
log_ok "Policy refresh completed successfully"
log_info "POLICY_AUDIT: version=$POLICY_VERSION hash=$POLICY_HASH universe_count=$UNIVERSE_COUNT valid_until=$VALID_UNTIL_INT"
log_info "Next refresh: $(date -d "@$((NOW + 1800))" '+%Y-%m-%d %H:%M:%S')"
exit 0
