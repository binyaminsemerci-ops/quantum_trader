#!/usr/bin/env bash
set -euo pipefail

redis-cli --raw XREVRANGE quantum:stream:apply.heat.observed + - COUNT 1200 | \
python3 - <<'PY'
import sys
from collections import Counter

lines = [l.rstrip("\n") for l in sys.stdin]
obs = None
counts = Counter()

i = 0
while i < len(lines):
    if lines[i].lower() == "obs_point" and i+1 < len(lines):
        obs = lines[i+1]
        i += 2
        continue
    if lines[i].lower() == "heat_found" and i+1 < len(lines) and obs is not None:
        hf = lines[i+1]
        counts[(obs, hf)] += 1
        i += 2
        continue
    i += 1

for (op, hf), n in sorted(counts.items()):
    print(f"{op} heat_found={hf} {n}")
PY
