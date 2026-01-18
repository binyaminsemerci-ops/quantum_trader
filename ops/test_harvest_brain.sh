#!/bin/bash
#
# test_harvest_brain.sh - Test HarvestBrain with simulated price movements
# Usage: bash test_harvest_brain.sh
#

set -e

echo "🧪 HarvestBrain Test Suite"
echo "=========================="
echo ""

# Config
SYMBOL="ETHUSDT"
ENTRY_PRICE=3300.0
STOP_LOSS=3200.0
TAKE_PROFIT=3500.0
POSITION_SIZE=1.0
RISK=$(echo "$ENTRY_PRICE - $STOP_LOSS" | bc)

echo "Test Setup:"
echo "  Symbol: $SYMBOL"
echo "  Entry: $ENTRY_PRICE"
echo "  Stop Loss: $STOP_LOSS"
echo "  Take Profit: $TAKE_PROFIT"
echo "  Position: $POSITION_SIZE"
echo "  Risk: $RISK"
echo ""

# Test 1: Entry execution
echo "=== Test 1: Entry Execution ==="
redis-cli XADD quantum:stream:execution.result "*" \
  symbol "$SYMBOL" \
  side "BUY" \
  qty "$POSITION_SIZE" \
  price "$ENTRY_PRICE" \
  status "FILLED" \
  entry_price "$ENTRY_PRICE" \
  stop_loss "$STOP_LOSS" \
  take_profit "$TAKE_PROFIT" \
  timestamp "$(date +%s)" \
  order_id "test_entry_$(date +%s)"

echo "✅ Entry execution injected: BUY $POSITION_SIZE $SYMBOL @ $ENTRY_PRICE"
sleep 2

# Test 2: Price at R=0.5 (should trigger 25% close)
PRICE_R05=$(echo "$ENTRY_PRICE + ($RISK * 0.5)" | bc)
echo ""
echo "=== Test 2: Price reaches R=0.5 ($PRICE_R05) ==="
echo "Expected: Harvest 25% ($POSITION_SIZE * 0.25 = $(echo "$POSITION_SIZE * 0.25" | bc))"

redis-cli XADD quantum:stream:execution.result "*" \
  symbol "$SYMBOL" \
  side "BUY" \
  qty "0" \
  price "$PRICE_R05" \
  status "PRICE_UPDATE" \
  current_price "$PRICE_R05" \
  timestamp "$(date +%s)"

sleep 3
SUGGESTIONS_COUNT=$(redis-cli XLEN quantum:stream:harvest.suggestions)
echo "Harvest suggestions count: $SUGGESTIONS_COUNT"
if [[ $SUGGESTIONS_COUNT -gt 0 ]]; then
    echo "Latest suggestion:"
    redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 1
fi

# Test 3: Price at R=1.0 (should trigger another 25%)
PRICE_R10=$(echo "$ENTRY_PRICE + ($RISK * 1.0)" | bc)
echo ""
echo "=== Test 3: Price reaches R=1.0 ($PRICE_R10) ==="
echo "Expected: Harvest another 25%"

redis-cli XADD quantum:stream:execution.result "*" \
  symbol "$SYMBOL" \
  side "BUY" \
  qty "0" \
  price "$PRICE_R10" \
  status "PRICE_UPDATE" \
  current_price "$PRICE_R10" \
  timestamp "$(date +%s)"

sleep 3
SUGGESTIONS_COUNT=$(redis-cli XLEN quantum:stream:harvest.suggestions)
echo "Harvest suggestions count: $SUGGESTIONS_COUNT"
if [[ $SUGGESTIONS_COUNT -gt 0 ]]; then
    echo "Latest suggestion:"
    redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 1
fi

# Test 4: Price at R=1.5 (should trigger another 25%)
PRICE_R15=$(echo "$ENTRY_PRICE + ($RISK * 1.5)" | bc)
echo ""
echo "=== Test 4: Price reaches R=1.5 ($PRICE_R15) ==="
echo "Expected: Harvest another 25%"

redis-cli XADD quantum:stream:execution.result "*" \
  symbol "$SYMBOL" \
  side "BUY" \
  qty "0" \
  price "$PRICE_R15" \
  status "PRICE_UPDATE" \
  current_price "$PRICE_R15" \
  timestamp "$(date +%s)"

sleep 3
SUGGESTIONS_COUNT=$(redis-cli XLEN quantum:stream:harvest.suggestions)
echo "Harvest suggestions count: $SUGGESTIONS_COUNT"
if [[ $SUGGESTIONS_COUNT -gt 0 ]]; then
    echo "All harvest suggestions:"
    redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 5
fi

# Test 5: Dedup check (inject same price again)
echo ""
echo "=== Test 5: Dedup Test ==="
echo "Injecting same R=1.5 price again (should be deduplicated)"

redis-cli XADD quantum:stream:execution.result "*" \
  symbol "$SYMBOL" \
  side "BUY" \
  qty "0" \
  price "$PRICE_R15" \
  status "PRICE_UPDATE" \
  current_price "$PRICE_R15" \
  timestamp "$(date +%s)"

sleep 3
SUGGESTIONS_COUNT_AFTER=$(redis-cli XLEN quantum:stream:harvest.suggestions)
echo "Harvest suggestions before: $SUGGESTIONS_COUNT"
echo "Harvest suggestions after: $SUGGESTIONS_COUNT_AFTER"
if [[ $SUGGESTIONS_COUNT == $SUGGESTIONS_COUNT_AFTER ]]; then
    echo "✅ Dedup working - no duplicate suggestions"
else
    echo "⚠️  Dedup may not be working - suggestions increased"
fi

# Test 6: Check dedup keys
echo ""
echo "=== Test 6: Dedup Keys ==="
DEDUP_COUNT=$(redis-cli KEYS "quantum:dedup:harvest:*" | wc -l)
echo "Active dedup keys: $DEDUP_COUNT"
redis-cli KEYS "quantum:dedup:harvest:*" | head -5

# Test 7: Check HarvestBrain logs
echo ""
echo "=== Test 7: Service Logs ==="
echo "Recent HarvestBrain activity:"
tail -20 /var/log/quantum/harvest_brain.log | grep -E "(Harvest|position|ERROR|WARNING)" || echo "No recent harvest activity"

# Summary
echo ""
echo "=== Test Summary ==="
echo "Entry Price: $ENTRY_PRICE"
echo "R=0.5 Price: $PRICE_R05 (expect 25% harvest)"
echo "R=1.0 Price: $PRICE_R10 (expect 25% harvest)"
echo "R=1.5 Price: $PRICE_R15 (expect 25% harvest)"
echo ""
echo "Total harvest suggestions: $SUGGESTIONS_COUNT_AFTER"
echo "Active dedup keys: $DEDUP_COUNT"
echo ""
echo "✅ Test suite completed"
echo ""
echo "Next steps:"
echo "1. Review harvest.suggestions stream: redis-cli XREVRANGE quantum:stream:harvest.suggestions + - COUNT 10"
echo "2. Check service logs: tail -f /var/log/quantum/harvest_brain.log"
echo "3. Monitor consumer group: redis-cli XINFO GROUPS quantum:stream:execution.result"
