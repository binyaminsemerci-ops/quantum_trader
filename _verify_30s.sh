#!/bin/bash
echo "=== Waiting 30s for HV2 to cycle with new code ==="
sleep 30

echo ""
echo "=== HV2_TICK after patch (last 1 min) ==="
journalctl -u quantum-harvest-v2.service --since "1 minute ago" --no-pager 2>/dev/null | grep "HV2_TICK" | tail -5

echo ""
echo "=== Active position keys (expect: ADA, BNB, BTC, ETH, SOL if all exist) ==="
redis-cli KEYS "quantum:position:*" 2>/dev/null | grep -vE "snapshot|ledger" | sort

echo ""
echo "=== Cooldown keys with TTL ==="
for k in $(redis-cli KEYS "quantum:cooldown:last_exec_ts:*" 2>/dev/null); do
    TTL=$(redis-cli TTL "$k")
    echo "  $k TTL=$TTL"
done

echo ""
echo "=== WRONGTYPE errors in last 5 min ==="
journalctl -u quantum-intent-executor.service --since "5 minutes ago" --no-pager 2>/dev/null | grep "WRONGTYPE" | wc -l
echo " ^ (expect 0)"

echo ""
echo "=== Proposals current state ==="
for k in $(redis-cli KEYS "quantum:harvest:proposal:*" 2>/dev/null); do
    SYM=$(echo "$k" | sed 's/quantum:harvest:proposal://')
    ACT=$(redis-cli HGET "$k" action 2>/dev/null)
    R=$(redis-cli HGET "$k" R_net 2>/dev/null)
    TYPE=$(redis-cli TYPE "$k" 2>/dev/null)
    echo "  $SYM: action=$ACT R_net=$R TYPE=$TYPE"
done
