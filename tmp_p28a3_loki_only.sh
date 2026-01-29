#!/bin/bash
set +e

output=$(/usr/bin/python3 /home/qt/quantum_trader/scripts/p28a3_verify_latency.py 2>&1)
rc=$?

truth=$(printf "%s\n" "$output" | grep -m1 -F "[TRUTH]")

if [ -n "$truth" ]; then
  printf "%s\n" "$truth"
else
  if [ $rc -ne 0 ]; then
    printf "%s\n" "[TRUTH] error running script (exit=$rc)"
  else
    printf "%s\n" "[TRUTH] missing (script output did not contain TRUTH line)"
  fi
  # still emit structured line with zeros (fail-open observability)
  printf "%s\n" "P28A3_TRUTH p99_ms=0 max_ms=0 samples=0 negative_outliers=0 headroom_x=0 max_wait_ms=0"
  exit 0
fi

# POSIX parse tokens
p99_ms=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^p99=/){gsub(/^p99=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')
max_ms=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^max=/){gsub(/^max=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')
samples=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^samples=/){gsub(/^samples=/,"",$i); print $i; exit}}')
negative_outliers=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^negative_outliers=/){gsub(/^negative_outliers=/,"",$i); print $i; exit}}')
headroom_x=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/^headroom=/){gsub(/^headroom=/,"",$i); sub(/x.*/,"",$i); print $i; exit}}')
max_wait_ms=$(printf "%s\n" "$truth" | awk '{for(i=1;i<=NF;i++) if($i~/max_wait=/){sub(/.*max_wait=/,"",$i); sub(/ms.*/,"",$i); print $i; exit}}')

# defaults if empty
p99_ms=${p99_ms:-0}
max_ms=${max_ms:-0}
samples=${samples:-0}
negative_outliers=${negative_outliers:-0}
headroom_x=${headroom_x:-0}
max_wait_ms=${max_wait_ms:-0}

printf "%s\n" "P28A3_TRUTH p99_ms=${p99_ms} max_ms=${max_ms} samples=${samples} negative_outliers=${negative_outliers} headroom_x=${headroom_x} max_wait_ms=${max_wait_ms}"
exit 0
