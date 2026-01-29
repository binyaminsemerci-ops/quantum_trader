#!/bin/bash
set +e

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

# Emit Loki-structured line
printf "P28A3_TRUTH p99_ms=%s max_ms=%s samples=%s negative_outliers=%s headroom_x=%s max_wait_ms=%s\n" \
    "$p99_ms" "$max_ms" "$samples" "$negative_outliers" "$headroom_x" "$max_wait_ms"

# Always exit 0 (fail-open)
exit 0
