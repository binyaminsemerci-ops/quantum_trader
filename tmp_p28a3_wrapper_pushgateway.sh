#!/bin/bash
set +e

# Source config (env vars from systemd EnvironmentFile)
: "${P28A3_TRUTH_PUSH_ENABLED:=false}"
: "${P28A3_TRUTH_PUSHGATEWAY_URL:=http://localhost:9091}"
: "${P28A3_TRUTH_PUSH_JOB:=quantum_p28a3_latency_proof}"
: "${P28A3_TRUTH_PUSH_TIMEOUT_SEC:=2}"

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

# Parse numeric fields from TRUTH line using POSIX tools
# Expected format: [TRUTH] p99=<N>ms max=<N>ms samples=<N> negative_outliers=<N> headroom=<X>x (max_wait=<N>ms)

# Use awk to parse key=value pairs
p99_ms=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^p99=/) {sub(/^p99=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')
max_ms=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^max=[0-9]/) {sub(/^max=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')
samples=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^samples=/) {sub(/^samples=/,"",$i); print $i; exit}}')
negative_outliers=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^negative_outliers=/) {sub(/^negative_outliers=/,"",$i); print $i; exit}}')
headroom_x=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^headroom=/) {sub(/^headroom=/,"",$i); sub(/x.*/,"",$i); print $i; exit}}')
max_wait_ms=$(echo "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/max_wait=/) {sub(/.*max_wait=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')

# Set defaults for missing values
: "${p99_ms:=0}"
: "${max_ms:=0}"
: "${samples:=0}"
: "${negative_outliers:=0}"
: "${headroom_x:=0}"
: "${max_wait_ms:=0}"

# Emit Loki-structured line (always)
printf "P28A3_TRUTH p99_ms=%s max_ms=%s samples=%s negative_outliers=%s headroom_x=%s max_wait_ms=%s\n" \
    "$p99_ms" "$max_ms" "$samples" "$negative_outliers" "$headroom_x" "$max_wait_ms"

# Push to Pushgateway (if enabled)
if [ "$P28A3_TRUTH_PUSH_ENABLED" = "true" ]; then
    # Verify Pushgateway (check for pushgateway-specific signature, not Prometheus)
    # Pushgateway has "push_time_seconds" metric with TYPE declaration
    if ! curl -fsS --max-time 1 "${P28A3_TRUTH_PUSHGATEWAY_URL}/metrics" 2>/dev/null | grep -qE '(^push_time_seconds\{|^# TYPE push_time_seconds )'; then
        printf "P28A3_TRUTH_PUSH status=fail reason=not_pushgateway_or_down url=%s\n" "$P28A3_TRUTH_PUSHGATEWAY_URL"
    else
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
        
        # Push with timeout and proper error handling
        if printf "%s\n" "$metrics" | curl -fsS --max-time "$P28A3_TRUTH_PUSH_TIMEOUT_SEC" \
             --data-binary @- \
             -H "Content-Type: text/plain" \
             "$push_url" >/dev/null 2>&1; then
            printf "P28A3_TRUTH_PUSH status=ok url=%s job=%s\n" "$P28A3_TRUTH_PUSHGATEWAY_URL" "$P28A3_TRUTH_PUSH_JOB"
        else
            printf "P28A3_TRUTH_PUSH status=fail reason=push_http_failed url=%s job=%s\n" "$P28A3_TRUTH_PUSHGATEWAY_URL" "$P28A3_TRUTH_PUSH_JOB"
        fi
    fi
fi

# Always exit 0 (fail-open)
exit 0
