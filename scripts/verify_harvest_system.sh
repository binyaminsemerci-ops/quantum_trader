#!/usr/bin/env bash
# Comprehensive verification script for harvest system
# Shows: CLOSE plans generated, CLOSE executions, position changes

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"

echo "=================================================="
echo "    HARVEST SYSTEM VERIFICATION"
echo "=================================================="
echo

# (1) CLOSE plans in apply.plan (last 200 entries)
echo "=== (1) EXIT PLANS PRODUCED (last 200 entries) ==="
CLOSE_COUNT=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.plan + - COUNT 200 | grep -c "FULL_CLOSE_PROPOSED" || echo "0")
echo "FULL_CLOSE_PROPOSED plans: $CLOSE_COUNT"
echo

# Show last 3 CLOSE plans with details
echo "Last 3 CLOSE plans:"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.plan + - COUNT 100 | \
    grep -B5 -A10 "FULL_CLOSE_PROPOSED" | head -60 || echo "No CLOSE plans found"
echo

# (2) CLOSE executions in apply.result
echo "=== (2) EXIT EXECUTIONS (last 100 results) ==="
EXECUTED_TRUE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.result + - COUNT 100 | grep -c "^True$" || echo "0")
REDUCE_ONLY_TRUE=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.result + - COUNT 100 | grep -B1 "^reduceOnly$" | grep -c "^True$" || echo "0")
echo "executed=True: $EXECUTED_TRUE"
echo "reduceOnly=True: $REDUCE_ONLY_TRUE"
echo

# Show last executed close
echo "Last executed CLOSE:"
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.result + - COUNT 200 | \
    awk '/^True$/{found=1} found{print; count++; if(count>=20) exit}' | head -40 || echo "No executed=True entries found"
echo

# (3) Current open positions
echo "=== (3) CURRENT OPEN POSITIONS ==="
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" KEYS "quantum:position:*" | while read -r key; do
    if [ -n "$key" ]; then
        symbol=$(echo "$key" | sed 's/quantum:position://')
        side=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" HGET "$key" side)
        qty=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" HGET "$key" quantity)
        echo "  $symbol: $side qty=$qty"
    fi
done
echo

# (4) Duplicate blocks (anti-dup gate working)
echo "=== (4) DUPLICATE BLOCKS (last 5 min) ==="
if command -v journalctl &>/dev/null; then
    SKIP_COUNT=$(journalctl -u quantum-apply-layer --since "5 minutes ago" 2>/dev/null | grep -c "SKIP_OPEN_DUPLICATE" || echo "0")
    echo "SKIP_OPEN_DUPLICATE: $SKIP_COUNT blocks"
    
    if [ "$SKIP_COUNT" -gt 0 ]; then
        echo "Sample (last 5):"
        journalctl -u quantum-apply-layer --since "5 minutes ago" 2>/dev/null | grep "SKIP_OPEN_DUPLICATE" | tail -5 || true
    fi
else
    echo "journalctl not available (not on VPS)"
fi
echo

# (5) Apply-layer CLOSE logs
echo "=== (5) APPLY-LAYER CLOSE ACTIVITY (last 10 min) ==="
if command -v journalctl &>/dev/null; then
    CLOSE_COUNT=$(journalctl -u quantum-apply-layer --since "10 minutes ago" 2>/dev/null | grep -c "\[CLOSE\]" || echo "0")
    echo "CLOSE log lines: $CLOSE_COUNT"
    
    if [ "$CLOSE_COUNT" -gt 0 ]; then
        echo "Recent CLOSE activity:"
        journalctl -u quantum-apply-layer --since "10 minutes ago" 2>/dev/null | grep "\[CLOSE\]" | tail -20 || true
    else
        echo "No CLOSE activity in last 10 minutes"
    fi
else
    echo "journalctl not available (not on VPS)"
fi
echo

# (6) Action breakdown (last 300 plans)
echo "=== (6) ACTION BREAKDOWN (last 300 plans) ==="
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" XREVRANGE quantum:stream:apply.plan + - COUNT 300 | \
    grep -A1 "^action$" | grep -v "^action$" | grep -v "^--$" | sort | uniq -c | sort -rn | head -10
echo

echo "=================================================="
echo "    VERIFICATION COMPLETE"
echo "=================================================="
echo
echo "Expected healthy state:"
echo "  - CLOSE plans: >5 (exitbrain generating)"
echo "  - executed=True: >0 (apply-layer executing)"
echo "  - reduceOnly=True: >0 (closes working)"
echo "  - Duplicate blocks: >10 (anti-dup working)"
echo "  - CLOSE logs: >0 (apply-layer processing)"
