#!/usr/bin/env bash
# Fix harvest-brain consumer group offset to start from new messages
# This prevents harvest-brain from getting stuck on old backlog events

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
STREAM_NAME="quantum:stream:execution.result"
GROUP_NAME="harvest_brain:execution"

echo "=== Harvest-Brain Stream Offset Fix ==="
echo "Stream: $STREAM_NAME"
echo "Group: $GROUP_NAME"
echo

# Check if stream exists
if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" EXISTS "$STREAM_NAME" | grep -q "1"; then
    echo "❌ Stream $STREAM_NAME does not exist"
    exit 1
fi

echo "✅ Stream exists"

# Check if group exists
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XINFO GROUPS "$STREAM_NAME" 2>/dev/null | grep -q "$GROUP_NAME"; then
    echo "✅ Consumer group exists"
    
    # Get current group info
    echo
    echo "Current group info:"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XINFO GROUPS "$STREAM_NAME" | grep -A5 "$GROUP_NAME" || true
    
    # Set group ID to $ (new messages only)
    echo
    echo "Setting group ID to $ (new messages only)..."
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XGROUP SETID "$STREAM_NAME" "$GROUP_NAME" '$'
    
    echo "✅ Group offset updated to $ (new messages)"
else
    echo "⚠️  Consumer group does not exist, creating with ID=$"
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XGROUP CREATE "$STREAM_NAME" "$GROUP_NAME" '$' MKSTREAM
    echo "✅ Consumer group created with offset $"
fi

echo
echo "New group info:"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XINFO GROUPS "$STREAM_NAME" | grep -A5 "$GROUP_NAME" || true

echo
echo "=== Fix Complete ==="
echo "harvest-brain will now start consuming from NEW messages only"
echo "Old backlog (EGLDUSDT, NEOUSDT qty=0.0 events) will be skipped"
