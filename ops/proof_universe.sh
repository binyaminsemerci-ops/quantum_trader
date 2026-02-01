#!/usr/bin/env bash
set -euo pipefail

# Quantum Trading Universe Proof Script
# Displays current universe state from Redis

REDIS_CMD="redis-cli"

echo "╔════════════════════════════════════════════════╗"
echo "║   QUANTUM UNIVERSE SERVICE - STATUS            ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check if Redis keys exist
if ! $REDIS_CMD EXISTS quantum:cfg:universe:active > /dev/null 2>&1; then
    echo "❌ ERROR: quantum:cfg:universe:active key not found"
    echo "   Universe service may not be running or has never fetched successfully"
    exit 1
fi

# Fetch active universe
ACTIVE_JSON=$($REDIS_CMD GET quantum:cfg:universe:active)

# Parse JSON (requires jq, fallback to basic parsing)
if command -v jq > /dev/null 2>&1; then
    MODE=$(echo "$ACTIVE_JSON" | jq -r '.mode')
    ASOF=$(echo "$ACTIVE_JSON" | jq -r '.asof_epoch')
    SYMBOLS=$(echo "$ACTIVE_JSON" | jq -r '.symbols[]')
    COUNT=$(echo "$SYMBOLS" | wc -l)
else
    # Basic fallback without jq
    MODE=$(echo "$ACTIVE_JSON" | grep -oP '"mode"\s*:\s*"\K[^"]+' || echo "unknown")
    ASOF=$(echo "$ACTIVE_JSON" | grep -oP '"asof_epoch"\s*:\s*\K[0-9]+' || echo "0")
    COUNT=$(echo "$ACTIVE_JSON" | grep -oP '"symbols"\s*:\s*\[' | wc -l)
    SYMBOLS=$(echo "$ACTIVE_JSON" | grep -oP '"[A-Z0-9]+USDT"' | sed 's/"//g')
fi

# Get metadata
META_ASOF=$($REDIS_CMD HGET quantum:cfg:universe:meta asof_epoch || echo "0")
META_LAST_OK=$($REDIS_CMD HGET quantum:cfg:universe:meta last_ok_epoch || echo "0")
META_COUNT=$($REDIS_CMD HGET quantum:cfg:universe:meta count || echo "0")
META_STALE=$($REDIS_CMD HGET quantum:cfg:universe:meta stale || echo "0")
META_ERROR=$($REDIS_CMD HGET quantum:cfg:universe:meta error || echo "")

# Calculate age
NOW=$(date +%s)
AGE_SEC=$((NOW - ASOF))
AGE_MIN=$((AGE_SEC / 60))

# Display results
echo "Mode:           $MODE"
echo "Active Symbols: $META_COUNT"
echo "Last Update:    $(date -d @${ASOF} '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "epoch ${ASOF}")"
echo "Age:            ${AGE_MIN} minutes (${AGE_SEC}s)"
echo ""

# Stale warning
if [ "$META_STALE" = "1" ]; then
    echo "⚠️  STATUS: STALE (using last known good data)"
    if [ -n "$META_ERROR" ]; then
        echo "   Last Error: ${META_ERROR:0:100}"
    fi
    LAST_OK_AGE=$((NOW - META_LAST_OK))
    LAST_OK_MIN=$((LAST_OK_AGE / 60))
    echo "   Last OK:    ${LAST_OK_MIN} minutes ago"
else
    echo "✅ STATUS: FRESH (recently updated)"
fi

echo ""
echo "Sample Symbols (first 20):"
echo "$SYMBOLS" | head -20 | nl -w2 -s'. '

echo ""
echo "Redis Keys:"
echo "  active:  quantum:cfg:universe:active"
echo "  last_ok: quantum:cfg:universe:last_ok"
echo "  meta:    quantum:cfg:universe:meta"
echo ""
echo "Full symbol list available via:"
echo "  redis-cli GET quantum:cfg:universe:active | jq -r '.symbols[]'"
