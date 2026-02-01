#!/usr/bin/env bash
set -euo pipefail

# Quantum Trading P3.3 Universe Source Proof Script
# Shows which allowlist source P3.3 Position State Brain is using

echo "╔════════════════════════════════════════════════╗"
echo "║   P3.3 ALLOWLIST SOURCE VERIFICATION           ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Check if P3.3 service is running
if ! systemctl is-active quantum-position-state-brain > /dev/null 2>&1; then
    echo "❌ ERROR: quantum-position-state-brain service not running"
    exit 1
fi

echo "✅ P3.3 Service: active"
echo ""

# Get recent logs showing allowlist source
echo "Allowlist Source (from logs):"
journalctl -u quantum-position-state-brain --since "5 minutes ago" --no-pager 2>/dev/null | \
    grep -E "Allowlist source=" | tail -1 | \
    sed 's/.*\[INFO\] /  /' || echo "  (no recent source logs found)"

echo ""

# Check Universe Service status
echo "Universe Service Status:"
if systemctl is-active quantum-universe-service > /dev/null 2>&1; then
    echo "  Service: active"
    
    if command -v redis-cli > /dev/null 2>&1; then
        STALE=$(redis-cli HGET quantum:cfg:universe:meta stale 2>/dev/null || echo "unknown")
        COUNT=$(redis-cli HGET quantum:cfg:universe:meta count 2>/dev/null || echo "unknown")
        
        echo "  Stale: $STALE"
        echo "  Symbols: $COUNT"
        
        if [ "$STALE" = "0" ]; then
            echo "  ✅ Universe FRESH - P3.3 should use universe symbols"
        elif [ "$STALE" = "1" ]; then
            echo "  ⚠️  Universe STALE - P3.3 should fallback to P33_ALLOWLIST"
        fi
    else
        echo "  (redis-cli not available for detailed check)"
    fi
else
    echo "  Service: inactive"
    echo "  ❌ Universe not running - P3.3 using P33_ALLOWLIST fallback"
fi

echo ""

# Show P33_ALLOWLIST env var
echo "Fallback Allowlist (P33_ALLOWLIST):"
if [ -f /etc/quantum/position-state-brain.env ]; then
    P33_ALLOWLIST=$(grep "^P33_ALLOWLIST=" /etc/quantum/position-state-brain.env 2>/dev/null | cut -d'=' -f2 || echo "not set")
    echo "  $P33_ALLOWLIST"
else
    echo "  (config file not found)"
fi

echo ""

# Show recent P3.3 permit activity
echo "Recent P3.3 Permit Activity (last 10):"
journalctl -u quantum-position-state-brain --since "10 minutes ago" --no-pager 2>/dev/null | \
    grep -E "P3.3 (ALLOW|DENY)" | tail -10 | \
    sed 's/.*quantum-position-state-brain\[[0-9]*\]: /  /' || echo "  (no recent permit activity)"

echo ""

# Summary
echo "═══════════════════════════════════════════════"
echo ""
echo "Expected Behavior:"
echo "  • If Universe active + stale=0 → P3.3 uses universe symbols"
echo "  • If Universe inactive or stale=1 → P3.3 uses P33_ALLOWLIST"
echo ""
echo "To check logs:"
echo "  journalctl -u quantum-position-state-brain -f | grep 'Allowlist source'"
