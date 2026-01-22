#!/bin/bash
# P0.D.5 Throughput Monitor
# Tracks backlog processing and calculates ETA

DURATION=${1:-60}

echo "=== P0.D.5 Throughput Monitor ==="
echo "Timestamp: $(date)"
echo "Duration: ${DURATION}s"
echo ""

# Initial metrics
INITIAL_LAG=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="lag"{getline; print}')
INITIAL_PENDING=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="pending"{getline; print}')
INITIAL_ENTRIES=$(redis-cli XINFO STREAM quantum:stream:execution.result | awk '$0=="entries-added"{getline; print}')

echo "Initial State:"
echo "  Lag: $INITIAL_LAG"
echo "  Pending: $INITIAL_PENDING"
echo "  Execution Results: $INITIAL_ENTRIES"

echo ""
echo "Waiting ${DURATION} seconds..."
sleep $DURATION

# Final metrics
FINAL_LAG=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="lag"{getline; print}')
FINAL_PENDING=$(redis-cli XINFO GROUPS quantum:stream:trade.intent | awk '$0=="pending"{getline; print}')
FINAL_ENTRIES=$(redis-cli XINFO STREAM quantum:stream:execution.result | awk '$0=="entries-added"{getline; print}')

echo ""
echo "After ${DURATION} seconds:"
echo "  Lag: $FINAL_LAG"
echo "  Pending: $FINAL_PENDING"
echo "  Execution Results: $FINAL_ENTRIES"

# Calculate changes
LAG_CHANGE=$((FINAL_LAG - INITIAL_LAG))
PENDING_CHANGE=$((FINAL_PENDING - INITIAL_PENDING))
NEW_ENTRIES=$((FINAL_ENTRIES - INITIAL_ENTRIES))

echo ""
echo "Changes:"
echo "  Lag: $LAG_CHANGE"
echo "  Pending: $PENDING_CHANGE"
echo "  New Executions: $NEW_ENTRIES"

# Throughput calculation (messages per minute)
THROUGHPUT=$((NEW_ENTRIES * 60 / DURATION))

echo ""
echo "📊 Throughput: $THROUGHPUT executions/minute"

if [ $THROUGHPUT -gt 0 ]; then
    echo "✅ Pipeline is processing messages"
    
    if [ $LAG_CHANGE -lt 0 ]; then
        LAG_DECREASE=$((-LAG_CHANGE))
        echo "✅ Lag decreasing: -$LAG_DECREASE messages"
        
        # ETA calculation
        if [ $THROUGHPUT -gt 0 ]; then
            ETA_MINUTES=$((FINAL_LAG / THROUGHPUT))
            ETA_HOURS=$((ETA_MINUTES / 60))
            ETA_DAYS=$((ETA_HOURS / 24))
            
            if [ $ETA_DAYS -gt 0 ]; then
                echo "📅 ETA to clear backlog: ~$ETA_DAYS days ($ETA_HOURS hours)"
            else
                echo "📅 ETA to clear backlog: ~$ETA_HOURS hours"
            fi
        fi
    else
        echo "⚠️ Lag not decreasing - likely processing stale intents (TTL drops)"
    fi
else
    echo "⚠️ No messages processed - check logs"
fi

# Check for stale drops
echo ""
echo "=== Stale Intent Drops ==="
STALE_COUNT=$(journalctl -u quantum-execution --since "${DURATION} seconds ago" | grep -c "STALE_INTENT_DROP" 2>/dev/null || echo "0")
echo "Stale intents dropped: $STALE_COUNT"

if [ "$STALE_COUNT" != "0" ]; then
    echo ""
    echo "Recent stale drops:"
    journalctl -u quantum-execution --since "${DURATION} seconds ago" | grep "STALE_INTENT_DROP" | tail -5
fi

# Current configuration
echo ""
echo "=== Current Configuration ==="
echo "Config from logs:"
journalctl -u quantum-execution -n 100 | grep "P0.D.5 Config" | tail -1 || echo "Config not found in recent logs"

echo ""
echo "=== Next Steps ==="
if [ $THROUGHPUT -lt 10 ]; then
    echo "⚠️ Low throughput - check for errors or increase XREADGROUP_COUNT"
elif [ $THROUGHPUT -lt 30 ]; then
    echo "💡 Moderate throughput - consider increasing XREADGROUP_COUNT to 20"
elif [ $THROUGHPUT -lt 60 ]; then
    echo "💡 Good throughput - can increase XREADGROUP_COUNT to 30-50"
else
    echo "✅ Excellent throughput - monitor for stability"
fi
