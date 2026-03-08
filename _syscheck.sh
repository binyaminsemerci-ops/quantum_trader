#!/bin/bash
echo "=== SERVICES ==="
for s in quantum-harvest-v2 quantum-intent-executor quantum-apply-layer quantum-position-state-brain quantum-ai-engine quantum-backend quantum-market-publisher; do
  printf "  %-42s %s\n" "$s" "$(systemctl is-active $s 2>/dev/null)"
done

echo ""
echo "=== HV2 LAST TICK ==="
journalctl -u quantum-harvest-v2 -n 50 --no-pager 2>/dev/null | grep HV2_TICK | tail -3

echo ""
echo "=== APPLY LAYER LAST DECISIONS ==="
journalctl -u quantum-apply-layer -n 200 --no-pager 2>/dev/null | grep "decision=" | tail -6

echo ""
echo "=== INTENT EXECUTOR METRICS ==="
journalctl -u quantum-intent-executor -n 200 --no-pager 2>/dev/null | grep "Metrics:" | tail -2

echo ""
echo "=== UNKNOWN_VARIANT errors (last 200 lines) ==="
CNT=$(journalctl -u quantum-apply-layer -n 200 --no-pager 2>/dev/null | grep -c "unknown_variant")
echo "  unknown_variant count: $CNT"

echo ""
echo "=== WRONGTYPE errors (last 5 min) ==="
CNT2=$(journalctl -u quantum-intent-executor --since "5 minutes ago" --no-pager 2>/dev/null | grep -c WRONGTYPE)
echo "  WRONGTYPE count: $CNT2"

echo ""
echo "=== OPEN POSITIONS ==="
redis-cli KEYS "quantum:position:*" 2>/dev/null | grep -v snapshot | grep -v ledger | sort

echo ""
echo "=== COOLDOWN TTLs ==="
for k in $(redis-cli KEYS "quantum:cooldown:last_exec_ts:*" 2>/dev/null); do
  echo "  $k TTL=$(redis-cli TTL $k)"
done

echo ""
echo "=== REDIS HEALTH ==="
redis-cli ping
echo "  DB size: $(redis-cli dbsize)"
redis-cli info memory 2>/dev/null | grep used_memory_human

echo ""
echo "=== DISK ==="
df -h / | tail -1

echo ""
echo "=== RAM ==="
free -h | grep Mem

echo ""
echo "=== P5 LAST BNBUSDT DECISION ==="
journalctl -u quantum-apply-layer -n 400 --no-pager 2>/dev/null | grep BNBUSDT | tail -3
