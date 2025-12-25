#!/bin/bash
echo "🔍 UNDERSØKER DET RIKTIGE SYSTEMET"
echo "===================================="
echo ""

echo "1️⃣ Sjekker om Trading Bot kjører:"
docker ps | grep trading_bot
echo ""

echo "2️⃣ Sjekker om Trading Bot publiserer signaler (siste 5):"
docker logs quantum_trading_bot --tail 100 | grep -i "Published trade.intent" | tail -5
if [ $? -ne 0 ]; then
    echo "❌ INGEN published signals funnet!"
fi
echo ""

echo "3️⃣ Sjekker EventBus stream lengde:"
STREAM_LEN=$(docker exec quantum_redis redis-cli XLEN quantum:stream:trade.intent 2>/dev/null)
echo "Stream length: $STREAM_LEN"
if [ "$STREAM_LEN" = "0" ] || [ -z "$STREAM_LEN" ]; then
    echo "❌ Stream er tom eller finnes ikke!"
fi
echo ""

echo "4️⃣ Sjekker sample fra stream (siste 2 meldinger):"
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 2
echo ""

echo "5️⃣ Sjekker om TradeIntentSubscriber kjører i AI Engine:"
docker logs quantum_ai_engine --tail 200 | grep -i "TradeIntent\|Subscribed to.*trade.intent"
if [ $? -ne 0 ]; then
    echo "❌ INGEN TradeIntentSubscriber logs funnet!"
fi
echo ""

echo "6️⃣ Sjekker hvilke containere som kjører:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep quantum
echo ""

echo "==================================="
echo "✅ UNDERSØKELSE FERDIG"
