#!/bin/bash
echo "📤 Injecting test market ticks..."
echo "====================================="

INITIAL_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent)
echo "Initial trade.intent: $INITIAL_LEN"

# Inject 10 identical ETHUSDT ticks
echo "Injecting 10x ETHUSDT ticks..."
for i in {1..10}; do
  docker exec quantum_redis redis-cli XADD quantum:stream:market.tick '*' \
    timestamp "2026-01-19T00:00:00Z" \
    symbol "ETHUSDT" \
    price "3330.51" \
    volume "100" \
    bid "3330.50" \
    ask "3330.52" > /dev/null
  echo "  ✓ Tick $i/10"
  sleep 0.1
done

echo ""
echo "⏳ Waiting 5 seconds..."
sleep 5

NEW_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent)
NEW_SIGNALS=$((NEW_LEN - INITIAL_LEN))

echo ""
echo "Final trade.intent: $NEW_LEN"
echo "New signals: $NEW_SIGNALS"
echo ""
echo "✅ Test complete"
