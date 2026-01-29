#!/bin/bash
set +e

# Source config (env vars from systemd EnvironmentFile)
: "${P28A3_TRUTH_PUSH_ENABLED:=false}"
: "${P28A3_TRUTH_PUSHGATEWAY_URL:=http://localhost:9091}"
: "${P28A3_TRUTH_PUSH_JOB:=quantum_p28a3_latency_proof}"
: "${P28A3_TRUTH_PUSH_TIMEOUT_SEC:=2}"
: "${P28A3_TRUTH_LOKI_STRUCTURED_ENABLED:=true}"

# Run Python script
output=$(/usr/bin/python3 /home/qt/quantum_trader/scripts/p28a3_verify_latency.py 2>&1)
rc=$?

# Extract TRUTH line (ASCII-safe, handles emoji prefix)
truth=$(printf "%s\n" "$output" | grep -m1 -F "[TRUTH]")

# Emit original TRUTH line (unchanged)
if [ -n "$truth" ]; then
    printf "%s\n" "$truth"
elif [ $rc -ne 0 ]; then
    printf "%s\n" "[TRUTH] error running script (exit=$rc)"
    exit 0
else
    printf "%s\n" "[TRUTH] missing (script output did not contain TRUTH line)"
    exit 0
fi

# Parse numeric fields from TRUTH line
# Expected format: [TRUTH] p99=<N>ms max=<N>ms samples=<N> negative_outliers=<N> headroom=<X>x (max_wait=<N>ms)
p99_ms=$(echo "$truth" | grep -oP 'p99=\K[0-9]+' || echo "0")
max_ms=$(echo "$truth" | grep -oP 'max=\K[0-9]+' || echo "0")
samples=$(echo "$truth" | grep -oP 'samples=\K[0-9]+' || echo "0")
negative_outliers=$(echo "$truth" | grep -oP 'negative_outliers=\K[0-9]+' || echo "0")
headroom_x=$(echo "$truth" | grep -oP 'headroom=\K[0-9.]+' || echo "0")
max_wait_ms=$(echo "$truth" | grep -oP 'max_wait=\K[0-9]+' || echo "0")

# Emit Loki-structured line (if enabled)
if [ "$P28A3_TRUTH_LOKI_STRUCTURED_ENABLED" = "true" ]; then
    printf "P28A3_TRUTH p99_ms=%s max_ms=%s samples=%s negative_outliers=%s headroom_x=%s max_wait_ms=%s\n" \
        "$p99_ms" "$max_ms" "$samples" "$negative_outliers" "$headroom_x" "$max_wait_ms"
fi

# Push to Pushgateway (if enabled)
if [ "$P28A3_TRUTH_PUSH_ENABLED" = "true" ]; then
    hostname=$(hostname -s || echo "unknown")
    push_url="${P28A3_TRUTH_PUSHGATEWAY_URL}/metrics/job/${P28A3_TRUTH_PUSH_JOB}/instance/${hostname}"
    
    # Build metrics payload
    metrics=$(cat <<EOF
# TYPE p28a3_latency_p99_ms gauge
p28a3_latency_p99_ms $p99_ms
# TYPE p28a3_latency_max_ms gauge
p28a3_latency_max_ms $max_ms
# TYPE p28a3_latency_samples gauge
p28a3_latency_samples $samples
# TYPE p28a3_latency_negative_outliers gauge
p28a3_latency_negative_outliers $negative_outliers
# TYPE p28a3_latency_headroom_x gauge
p28a3_latency_headroom_x $headroom_x
# TYPE p28a3_latency_max_wait_ms gauge
p28a3_latency_max_wait_ms $max_wait_ms
EOF
)
    
    # Push with timeout (fail-open)
    if ! curl -s --max-time "$P28A3_TRUTH_PUSH_TIMEOUT_SEC" \
         --data-binary "$metrics" \
         -H "Content-Type: text/plain" \
         "$push_url" >/dev/null 2>&1; then
        printf "P28A3_TRUTH_PUSH status=fail reason=unreachable url=%s\n" "$push_url"
    fi
fi

# Always exit 0 (fail-open)
exit 0
