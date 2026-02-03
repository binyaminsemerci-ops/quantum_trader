#!/bin/bash
# Backlog Drain Runner
# Safely processes historical trade.intent events

set -e

CONTAINER=${CONTAINER:-quantum_backend}
MODE=${MODE:-dry-run}
THROTTLE=${THROTTLE:-2}
MAX_AGE_HOURS=${MAX_AGE_HOURS:-24}
MIN_CONFIDENCE=${MIN_CONFIDENCE:-0.6}
MAX_EVENTS=${MAX_EVENTS:-}
SYMBOLS=${SYMBOLS:-}

echo "═══════════════════════════════════════════════════════════"
echo "🔧 BACKLOG DRAIN SERVICE"
echo "═══════════════════════════════════════════════════════════"
echo "Mode: $MODE"
echo "Throttle: $THROTTLE events/sec"
echo "Max age: $MAX_AGE_HOURS hours"
echo "Min confidence: $MIN_CONFIDENCE"
[[ -n "$MAX_EVENTS" ]] && echo "Max events: $MAX_EVENTS"
[[ -n "$SYMBOLS" ]] && echo "Symbol filter: $SYMBOLS"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Build command
CMD="python -m backend.services.execution.backlog_drain_service"
CMD="$CMD --mode=$MODE"
CMD="$CMD --throttle=$THROTTLE"
CMD="$CMD --max-age-hours=$MAX_AGE_HOURS"
CMD="$CMD --min-confidence=$MIN_CONFIDENCE"
[[ -n "$MAX_EVENTS" ]] && CMD="$CMD --max-events=$MAX_EVENTS"
[[ -n "$SYMBOLS" ]] && CMD="$CMD --symbols=$SYMBOLS"

# Execute in container
echo "Running: docker exec $CONTAINER $CMD"
echo ""
docker exec -it $CONTAINER $CMD

echo ""
echo "✅ Drain complete"
