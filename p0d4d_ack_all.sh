#!/bin/bash
set -euo pipefail

STREAM=quantum:stream:trade.intent
GROUP=quantum:group:execution:trade.intent  
CONSUMER=execution-quantumtrader-prod-1-417524

echo "=== ACK ALL PENDING MESSAGES FOR CONSUMER ==="
echo "Consumer: $CONSUMER"
echo

# Get ALL pending
redis-cli XPENDING "$STREAM" "$GROUP" - + 50000 "$CONSUMER" | awk '{print $1}' > /tmp/pending_ids.txt
COUNT=$(wc -l < /tmp/pending_ids.txt)
echo "Found $COUNT pending messages"

if [ "$COUNT" -gt 0 ]; then
  echo "ACKing all..."
  cat /tmp/pending_ids.txt | xargs -n 1000 redis-cli XACK "$STREAM" "$GROUP" >/dev/null 2>&1
  echo "✅ ACKed $COUNT messages"
fi

AFTER=$(redis-cli XPENDING "$STREAM" "$GROUP" "$CONSUMER" | wc -l)
echo "Pending lines after: $AFTER"

# Wait 10s and check if processing starts
echo
echo "Waiting 10s for consumer to start processing NEW messages..."
BEFORE=$(redis-cli XINFO GROUPS "$STREAM" | grep -A40 "entries-read" | head -1)
sleep 10
AFTER=$(redis-cli XINFO GROUPS "$STREAM" | grep -A40 "entries-read" | head -1)
echo "entries-read before: $BEFORE"
echo "entries-read after:  $AFTER"

echo
echo "Checking execution.result..."
redis-cli XINFO STREAM quantum:stream:execution.result | grep -A1 "last-generated-id"
