#!/bin/bash
# XGBoost Confidence Sanity Check
# Parses recent trade.intent payloads and analyzes XGB confidence distribution

echo "=== XGBoost Confidence Sanity Check ==="
echo "Fetching last 50 trade.intent payloads..."

# Fetch payloads from Redis
PAYLOADS=$(docker exec quantum_redis redis-cli XREVRANGE quantum:stream:trade.intent + - COUNT 50 2>/dev/null)

if [ -z "$PAYLOADS" ]; then
    echo "❌ No trade.intent payloads found in Redis"
    exit 1
fi

# Extract XGB confidences using grep/awk
echo "$PAYLOADS" | grep -oP '"model_breakdown":\{[^}]+' | \
    grep -oP '"xgb":\{[^}]+' | \
    grep -oP '"confidence":[0-9.]+' | \
    awk -F: '{print $2}' > /tmp/xgb_confidences.txt

COUNT=$(wc -l < /tmp/xgb_confidences.txt)

if [ "$COUNT" -eq 0 ]; then
    echo "⚠️  No XGB confidences found (model may be returning xgb_no_model)"
    echo "Checking for xgb_no_model occurrences..."
    XGB_NO_MODEL=$(echo "$PAYLOADS" | grep -c "xgb_no_model")
    echo "Found xgb_no_model: $XGB_NO_MODEL times"
    exit 1
fi

echo "Found $COUNT XGB predictions"
echo ""

# Calculate stats using awk
awk '
BEGIN {
    min=999; max=0; sum=0; count=0
}
{
    val=$1
    sum+=val
    count++
    if(val<min) min=val
    if(val>max) max=val
    values[count]=val
}
END {
    # Sort for median
    n=asort(values)
    if(n%2) median=values[(n+1)/2]
    else median=(values[n/2]+values[n/2+1])/2
    
    avg=sum/count
    
    print "📊 XGB Confidence Statistics:"
    print "  Min:     " min
    print "  Median:  " median
    print "  Mean:    " sprintf("%.4f", avg)
    print "  Max:     " max
    print ""
    
    # Check for anomalies
    if(min==max) {
        print "❌ ANOMALY: All confidences identical (" min ")"
        print "   This indicates a mapping/feature bug"
    } else if(min > 0.95) {
        print "⚠️  WARNING: All confidences > 0.95"
        print "   Model may be overfitting or feature issue"
    } else if(max < 0.55) {
        print "⚠️  WARNING: All confidences < 0.55"
        print "   Model may be underconfident or broken"
    } else {
        print "✅ Confidence distribution looks healthy"
    }
}
' /tmp/xgb_confidences.txt

# Show unique values and counts
echo ""
echo "📋 Unique Confidence Values (top 10):"
sort /tmp/xgb_confidences.txt | uniq -c | sort -rn | head -10

# Cleanup
rm -f /tmp/xgb_confidences.txt
