#!/bin/bash
echo "=== All permit key types in Redis ==="
redis-cli keys "quantum:permit:*" | sed 's/quantum:permit://' | grep -oE "^[^:]+:" | sort | uniq -c | sort -rn | head -10

echo ""
echo "=== Recent ANTI_CHURN logs in intent_bridge ==="
journalctl -u quantum-intent-bridge --no-pager --since "5 minutes ago" -q 2>&1 | grep -iE "ANTI_CHURN|churn" | tail -10

echo ""
echo "=== EXIT_OWNERSHIP enforcement in apply_layer ==="
journalctl -u quantum-apply-layer --no-pager --since "2 minutes ago" -q 2>&1 | grep -iE "exit_owner|DENY_NOT_EXIT|ALLOW_EXIT" | tail -10

echo ""
echo "=== System summary ==="
echo "Governor bypass: $(redis-cli get quantum:global:governor_bypass 2>/dev/null || echo not-set)"
echo "Kill switch: $(redis-cli get quantum:global:kill_switch 2>/dev/null || echo not-set)"
echo "Permit keys in Redis: $(redis-cli keys 'quantum:permit:*' | wc -l)"
