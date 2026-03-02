#!/bin/bash
echo "=== Waiting 60s for intent-executor to process plans ==="
sleep 60

echo ""
echo "=== WRONGTYPE errors in last 60s (expect 0) ==="
COUNT=$(journalctl -u quantum-intent-executor.service --since "60 seconds ago" --no-pager 2>/dev/null | grep -c "WRONGTYPE" || echo 0)
echo "  WRONGTYPE errors: $COUNT"

echo ""
echo "=== Recent intent-executor log (last 60s) ==="
journalctl -u quantum-intent-executor.service --since "60 seconds ago" --no-pager 2>/dev/null \
  | grep -E "Processing plan|Permit found|Waiting.*permit|WRONGTYPE|ALLOW|executed=|filled=" | tail -20

echo ""
echo "=== Summary ==="
echo "  WRONGTYPE: $COUNT (pass if 0)"
echo "  intent-executor: $(systemctl is-active quantum-intent-executor.service)"
