#!/bin/bash
BEFORE=$(redis-cli XLEN quantum:stream:trade.intent)
echo "📊 Before: $BEFORE signals"
echo ""
echo "📤 Injecting 10 identical ETHUSDT ticks..."

for i in {1..10}; do
  redis-cli XADD quantum:stream:market.tick '*' \
    timestamp "2026-01-19T00:00:00Z" \
    symbol "ETHUSDT" \
    price "3330.51" \
    volume "100" > /dev/null
  echo "  ✓ Injected tick $i"
  sleep 0.05
done

echo ""
echo "⏳ Waiting 5 seconds for AI Engine processing..."
sleep 5

AFTER=$(redis-cli XLEN quantum:stream:trade.intent)
NEW=$((AFTER - BEFORE))

echo ""
echo "📊 After: $AFTER signals"
echo "📈 New signals: $NEW (from 10 identical ticks)"
echo ""

if [ $NEW -le 2 ]; then
  echo "✅ DEDUP WORKING! Only $NEW signals created from 10 identical ticks"
else
  echo "⚠️ DEDUP FAILED! Created $NEW signals (expected 1-2)"
fi
