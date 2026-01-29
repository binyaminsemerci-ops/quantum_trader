#!/bin/bash
set +e
output=$(/usr/bin/python3 /home/qt/quantum_trader/scripts/p28a3_verify_latency.py 2>&1)
rc=$?
# Match [TRUTH] anywhere in line (ASCII-safe, handles emoji prefix)
truth=$(printf "%s\n" "$output" | grep -m1 -F "[TRUTH]")
if [ -n "$truth" ]; then
    printf "%s\n" "$truth"
elif [ $rc -ne 0 ]; then
    printf "%s\n" "[TRUTH] error running script (exit=$rc)"
else
    printf "%s\n" "[TRUTH] missing (script output did not contain TRUTH line)"
fi
exit 0
