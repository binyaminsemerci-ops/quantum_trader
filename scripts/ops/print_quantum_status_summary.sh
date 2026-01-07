#!/bin/bash
# print_quantum_status_summary.sh
# Prints formatted status of all quantum systemd services
# Usage: bash scripts/ops/print_quantum_status_summary.sh

set -euo pipefail

echo "=== Quantum Services Status Summary ==="
echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo

# Print header
printf "%-45s %-10s %-20s\n" "SERVICE" "STATUS" "ACTIVE SINCE"
printf "%-45s %-10s %-20s\n" "-------" "------" "------------"

# Get all quantum services
systemctl list-units 'quantum*.service' --all --no-pager --no-legend | while read -r line; do
    SERVICE=$(echo "$line" | awk '{print $1}')
    LOAD=$(echo "$line" | awk '{print $2}')
    ACTIVE=$(echo "$line" | awk '{print $3}')
    SUB=$(echo "$line" | awk '{print $4}')
    
    # Get active timestamp if running
    if [ "$ACTIVE" = "active" ]; then
        TIMESTAMP=$(systemctl show "$SERVICE" --property=ActiveEnterTimestamp --value | awk '{print $1, $2, $3}')
        STATUS="✅ running"
    elif [ "$ACTIVE" = "inactive" ]; then
        TIMESTAMP="-"
        STATUS="⚪ inactive"
    elif [ "$ACTIVE" = "failed" ]; then
        TIMESTAMP="-"
        STATUS="❌ FAILED"
    elif [ "$ACTIVE" = "activating" ]; then
        TIMESTAMP="-"
        STATUS="🔄 starting"
    else
        TIMESTAMP="-"
        STATUS="$ACTIVE"
    fi
    
    printf "%-45s %-10s %-20s\n" "$SERVICE" "$STATUS" "$TIMESTAMP"
done

echo
echo "=== Summary ==="
TOTAL=$(systemctl list-units 'quantum*.service' --all --no-pager --no-legend | wc -l)
RUNNING=$(systemctl list-units 'quantum*.service' --state=running --no-pager --no-legend | wc -l)
FAILED=$(systemctl --failed --no-pager --no-legend 'quantum*.service' 2>/dev/null | wc -l)

echo "Total services: $TOTAL"
echo "Running: $RUNNING"
echo "Failed: $FAILED"

if [ "$FAILED" -eq 0 ] && [ "$RUNNING" -eq 12 ]; then
    echo "✅ System healthy (12 expected services running)"
elif [ "$FAILED" -gt 0 ]; then
    echo "❌ System has failed services"
else
    echo "⚠️  Unexpected running count (expected: 12, actual: $RUNNING)"
fi
