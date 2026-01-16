#!/bin/bash
NOW=$(date +%s)

redis-cli SET quantum:rl:policy:BTCUSDT "{\"action\":\"BUY\",\"confidence\":0.85,\"version\":\"v2.0\",\"timestamp\":$NOW,\"reason\":\"fresh_test\"}"
redis-cli SET quantum:rl:policy:ETHUSDT "{\"action\":\"SELL\",\"confidence\":0.78,\"version\":\"v2.0\",\"timestamp\":$NOW,\"reason\":\"fresh_test\"}"
redis-cli SET quantum:rl:policy:SOLUSDT "{\"action\":\"BUY\",\"confidence\":0.82,\"version\":\"v2.0\",\"timestamp\":$NOW,\"reason\":\"fresh_test\"}"

echo "--- Fresh policies set at $NOW ---"
redis-cli KEYS "quantum:rl:policy:*" | while read key; do
    echo "$key:"
    redis-cli GET "$key"
done
