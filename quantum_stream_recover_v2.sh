#!/usr/bin/env bash
set -euo pipefail

# Quantum Stream Recovery - Automated stale pending claim + zombie cleanup
# Designed for systemd timer execution (every 2 minutes)

STREAM="quantum:stream:trade.intent"
GROUP="quantum:group:execution:trade.intent"
LOG="/var/log/quantum/stream_recover.log"

# Ensure log directory exists
mkdir -p /var/log/quantum

# Safety: MODE detection from /etc/quantum/*.env
# Default to TESTNET unless explicitly LIVE
MODE="TESTNET"
if [ -f /etc/quantum/testnet.env ]; then
  # testnet.env exists -> TESTNET mode
  if grep -qiE "^(USE_BINANCE_TESTNET|BINANCE_TESTNET)=true" /etc/quantum/testnet.env 2>/dev/null; then
    MODE="TESTNET"
  fi
elif grep -qiE "^BINANCE_TESTNET=false|^TRADING_MODE=LIVE" /etc/quantum/*.env 2>/dev/null; then
  # Explicit LIVE mode flags
  MODE="LIVE"
fi

# LIVE mode: log and exit (safety)
if [[ "$MODE" == "LIVE" ]]; then
  echo "$(date -Is) [RECOVER] LIVE mode detected -> skipping automated recovery" >> "$LOG"
  exit 0
fi

# Consumer name for this recovery run
CONSUMER="recover-$(hostname)-$$"

# Ensure consumer group exists (idempotent)
redis-cli XGROUP CREATE "$STREAM" "$GROUP" 0 MKSTREAM >/dev/null 2>&1 || true

# CLAIM STALE PENDING (idle > 60 seconds)
# Use XAUTOCLAIM if available (Redis >= 6.2)
if redis-cli COMMAND INFO XAUTOCLAIM >/dev/null 2>&1; then
  # XAUTOCLAIM returns: [next_start_id, [[id, [field, value, ...]], ...]]
  OUT="$(redis-cli --raw XAUTOCLAIM "$STREAM" "$GROUP" "$CONSUMER" 60000 0 COUNT 200 2>/dev/null || true)"
  # Count claimed entries (lines matching Redis stream ID format)
  CLAIMED_COUNT="$(printf "%s\n" "$OUT" | grep -cE '^[0-9]+-[0-9]+$' || echo 0)"
  echo "$(date -Is) [RECOVER] XAUTOCLAIM claimed=$CLAIMED_COUNT consumer=$CONSUMER idle_threshold=60s" >> "$LOG"
else
  # Fallback: log that XAUTOCLAIM is unavailable
  echo "$(date -Is) [RECOVER] XAUTOCLAIM unavailable (requires Redis >= 6.2)" >> "$LOG"
  CLAIMED_COUNT=0
fi

# SAFE ZOMBIE CLEANUP
# Delete consumers with: idle > 1 hour (3600000ms) AND pending == 0
# Parse XINFO CONSUMERS output (--raw makes it line-based)
RAW="$(redis-cli --raw XINFO CONSUMERS "$STREAM" "$GROUP" 2>/dev/null || true)"

# State machine to parse consumer blocks
name=""
pending=""
idle=""
DELETED_COUNT=0

while IFS= read -r line; do
  case "$line" in
    name)
      # Next line is the consumer name
      read -r name
      ;;
    pending)
      # Next line is the pending count
      read -r pending
      ;;
    idle)
      # Next line is the idle time in milliseconds
      read -r idle
      
      # When we have all three fields, evaluate for deletion
      if [[ -n "${name}" && -n "${pending}" && -n "${idle}" ]]; then
        # Validate numeric fields
        if [[ "${idle}" =~ ^[0-9]+$ && "${pending}" =~ ^[0-9]+$ ]]; then
          # Delete if idle > 1 hour AND pending == 0
          if (( idle > 3600000 )) && (( pending == 0 )); then
            if redis-cli XGROUP DELCONSUMER "$STREAM" "$GROUP" "$name" >/dev/null 2>&1; then
              echo "$(date -Is) [CLEAN] deleted_consumer=$name idle_ms=$idle pending=$pending" >> "$LOG"
              DELETED_COUNT=$((DELETED_COUNT + 1))
            fi
          fi
        fi
        # Reset for next consumer
        name=""
        pending=""
        idle=""
      fi
      ;;
  esac
done <<< "$RAW"

# Summary log
if (( CLAIMED_COUNT > 0 || DELETED_COUNT > 0 )); then
  echo "$(date -Is) [SUMMARY] claimed=$CLAIMED_COUNT deleted=$DELETED_COUNT" >> "$LOG"
else
  echo "$(date -Is) [SUMMARY] no action needed (claimed=0 deleted=0)" >> "$LOG"
fi

exit 0
