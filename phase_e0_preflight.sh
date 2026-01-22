#!/bin/bash
# PHASE E0: Preflight Evidence Capture

DIR=/tmp/phase_e_$(date +%Y%m%d_%H%M%S)
mkdir -p $DIR/{before,after,proof,backup,logs}

echo "=== PHASE E0: PREFLIGHT EVIDENCE CAPTURE ==="
echo "Evidence directory: $DIR"
echo ""
echo "=== Step 1: Verify Phase D streams ==="
/usr/bin/redis-cli --scan --pattern "quantum:stream:*" 2>/dev/null | sort > $DIR/before/streams.list.txt
echo "Found streams:"
cat $DIR/before/streams.list.txt
echo ""
echo "=== Step 2: List services ==="
systemctl list-units --type=service | grep -E "quantum|router|execution|ai-engine" | sort > $DIR/before/services.list.txt
echo "Active services (first 15):"
head -15 $DIR/before/services.list.txt
echo ""
echo "=== Step 3: Critical stream verification ==="
echo "Checking for required streams..."
for stream in "quantum:stream:execution.result" "quantum:stream:position.snapshot" "quantum:stream:pnl.unrealized" "quantum:stream:pnl.snapshot" "quantum:stream:trade.intent"; do
  xlen=$(/usr/bin/redis-cli XLEN "$stream" 2>/dev/null || echo "NOT_FOUND")
  echo "$stream: $xlen"
done > $DIR/before/critical_streams.txt
cat $DIR/before/critical_streams.txt
echo ""
echo "=== PREFLIGHT SUMMARY ==="
streams_found=$(cat $DIR/before/streams.list.txt | wc -l)
echo "Total streams found: $streams_found"

# Check critical streams
echo ""
echo "Critical streams status:"
grep "NOT_FOUND" $DIR/before/critical_streams.txt | wc -l > $DIR/before/missing_count.txt
missing=$(cat $DIR/before/missing_count.txt)

if [ "$missing" -gt 0 ]; then
  echo "⚠️  WARNING: $missing critical streams missing!"
  echo "Missing streams:"
  grep "NOT_FOUND" $DIR/before/critical_streams.txt
  echo ""
  echo "Found streams:"
  grep -v "NOT_FOUND" $DIR/before/critical_streams.txt
else
  echo "✅ All critical streams present"
fi

echo ""
echo "Evidence saved to: $DIR"
echo "$DIR"
