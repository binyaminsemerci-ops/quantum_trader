#!/bin/bash
redis-cli DEL quantum:position:BNBUSDT quantum:position:ADAUSDT 2>/dev/null
redis-cli DEL quantum:cooldown:last_exec_ts:ADAUSDT 2>/dev/null
redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | while read k; do redis-cli DEL "$k" 2>/dev/null; done
sleep 5
echo "pos=$(redis-cli KEYS 'quantum:position:[A-Z]*' 2>/dev/null | wc -l)"
echo "cooldowns=$(redis-cli KEYS 'quantum:cooldown:*' 2>/dev/null | wc -l)"
echo "BNB=$(redis-cli HGET quantum:position:snapshot:BNBUSDT position_amt 2>/dev/null)"
echo "ADA=$(redis-cli HGET quantum:position:snapshot:ADAUSDT position_amt 2>/dev/null)"
echo "BTC=$(redis-cli HGET quantum:position:snapshot:BTCUSDT position_amt 2>/dev/null)"
echo "ETH=$(redis-cli HGET quantum:position:snapshot:ETHUSDT position_amt 2>/dev/null)"
echo "SOL=$(redis-cli HGET quantum:position:snapshot:SOLUSDT position_amt 2>/dev/null)"
