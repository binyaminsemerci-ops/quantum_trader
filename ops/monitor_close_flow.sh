#!/bin/bash
# Monitor CLOSE orders flow after SKIP_FLAT_SELL=false fix
# Run on VPS: bash /home/qt/quantum_trader/ops/monitor_close_flow.sh

echo "═══════════════════════════════════════════════════════════════"
echo "🔍 CLOSE ORDER FLOW MONITOR - FASE A VERIFICATION"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# 1. Check Intent Bridge config
echo "1️⃣ Intent Bridge Configuration:"
echo "────────────────────────────────────────────────────────────────"
grep SKIP_FLAT_SELL /etc/quantum/intent-bridge.env
journalctl -u quantum-intent-bridge -n 200 --no-pager | grep "Skip flat SELL" | tail -1
echo ""

# 2. Count CLOSE orders in apply.plan stream (last 100 entries)
echo "2️⃣ CLOSE Orders in apply.plan (last 100 entries):"
echo "────────────────────────────────────────────────────────────────"
close_count=$(redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -c "FULL_CLOSE_PROPOSED\|PARTIAL_CLOSE")
reduce_only_count=$(redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | grep -c "reduceOnly")
echo "CLOSE actions: $close_count"
echo "reduceOnly orders: $reduce_only_count"
echo ""

# 3. Recent CLOSE orders (last 5)
echo "3️⃣ Recent CLOSE Orders (last 5):"
echo "────────────────────────────────────────────────────────────────"
redis-cli XREVRANGE quantum:stream:apply.plan + - COUNT 100 | \
  awk '/^[0-9]+-[0-9]+$/ {id=$0} /^symbol$/ {getline; sym=$0} /^action$/ {getline; if ($0 ~ /CLOSE/) {print id " | " sym " | " $0}}' | \
  head -5
echo ""

# 4. Trade.intent to apply.plan flow (Intent Bridge activity)
echo "4️⃣ Intent Bridge Activity (last 10 minutes):"
echo "────────────────────────────────────────────────────────────────"
journalctl -u quantum-intent-bridge --since "10 minutes ago" --no-pager | \
  grep -E "Bridge success|Published plan.*SELL|Skip publish.*SELL" | \
  tail -10
echo ""

# 5. Execution results from apply.result stream
echo "5️⃣ Recent Executions (apply.result, last 10):"
echo "────────────────────────────────────────────────────────────────"
redis-cli XREVRANGE quantum:stream:apply.result + - COUNT 10 | \
  awk '/^[0-9]+-[0-9]+$/ {id=$0} /^symbol$/ {getline; sym=$0} /^side$/ {getline; side=$0} /^status$/ {getline; print id " | " sym " | " side " | " $0}' | \
  head -10
echo ""

# 6. Check for "Skip publish" events (should be ZERO after fix)
echo "6️⃣ Filtered SELL Orders (should be ZERO after fix):"
echo "────────────────────────────────────────────────────────────────"
skipped=$(journalctl -u quantum-intent-bridge --since "1 hour ago" --no-pager | grep -c "Skip publish.*SELL")
echo "SELL orders skipped in last hour: $skipped"
if [ "$skipped" -eq 0 ]; then
  echo "✅ NO SELL orders filtered - SKIP_FLAT_SELL=false is working!"
else
  echo "⚠️  WARNING: Some SELL orders still being filtered"
  journalctl -u quantum-intent-bridge --since "1 hour ago" --no-pager | \
    grep "Skip publish.*SELL" | tail -5
fi
echo ""

# 7. Live monitoring mode
echo "7️⃣ Live Monitoring (Ctrl+C to stop):"
echo "────────────────────────────────────────────────────────────────"
echo "Watching for new CLOSE orders in apply.plan..."
echo ""

redis-cli --csv XREAD BLOCK 5000 STREAMS quantum:stream:apply.plan $ | \
  while read line; do
    if echo "$line" | grep -q "CLOSE\|reduceOnly"; then
      timestamp=$(date '+%Y-%m-%d %H:%M:%S')
      echo "[$timestamp] CLOSE ORDER DETECTED: $line"
    fi
  done
