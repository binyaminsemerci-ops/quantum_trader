#!/bin/bash
echo "=== VERIFY: P3 snapshot keys ==="
COUNT=$(redis-cli KEYS "quantum:position:snapshot:*" | grep -c "snapshot" 2>/dev/null || echo 0)
echo "  Snapshot keys remaining: $COUNT (expect 0)"

echo ""
echo "=== VERIFY: P4 cooldowns ==="
COOLS=$(redis-cli KEYS "quantum:cooldown:last_exec_ts:*" 2>/dev/null)
if [ -z "$COOLS" ]; then
    echo "  No permanent cooldown keys [GOOD]"
else
    echo "$COOLS" | while read -r k; do
        TTL=$(redis-cli TTL "$k")
        echo "  $k TTL=$TTL"
    done
fi

echo ""
echo "=== VERIFY: P6 permit keys type ==="
PERMITS=$(redis-cli KEYS "quantum:permit:p33:*" 2>/dev/null | head -10)
if [ -z "$PERMITS" ]; then
    echo "  No permit keys currently [OK]"
else
    echo "$PERMITS" | while read -r k; do
        T=$(redis-cli TYPE "$k")
        echo "  $k TYPE=$T"
    done
fi

echo ""
echo "=== VERIFY: Ledger symbol fields ==="
echo "  BNB symbol: $(redis-cli HGET quantum:position:ledger:BNBUSDT symbol)"
echo "  BTC symbol: $(redis-cli HGET quantum:position:ledger:BTCUSDT symbol)"
echo "  ETH symbol: $(redis-cli HGET quantum:position:ledger:ETHUSDT symbol)"

echo ""
echo "=== VERIFY: Service status ==="
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service)"
echo "  harvest-v2:      $(systemctl is-active quantum-harvest-v2.service)"

echo ""
echo "=== VERIFY: HV2 skip counts (last 5 min) ==="
journalctl -u quantum-harvest-v2.service --since "5 minutes ago" --no-pager 2>/dev/null \
  | grep -E "SKIP_INVALID|skipped_invalid|total_keys" | tail -10

echo ""
echo "=== VERIFY: Intent-executor errors ==="
journalctl -u quantum-intent-executor.service --since "10 minutes ago" --no-pager 2>/dev/null \
  | grep -iE "WRONGTYPE|permit.*error|hgetall.*err" | tail -10

echo ""
echo "=== Position keys (clean) ==="
redis-cli KEYS "quantum:position:*" 2>/dev/null | grep -vE "snapshot|ledger" | sort
