#!/usr/bin/env python3
import sys
from collections import Counter

lines = [l.rstrip("\n") for l in sys.stdin]
obs = None
reason = None
counts = Counter()

i = 0
while i < len(lines):
    if lines[i].lower() == "obs_point" and i+1 < len(lines):
        obs = lines[i+1]
        i += 2
        continue
    if lines[i].lower() == "heat_reason" and i+1 < len(lines) and obs is not None:
        reason = lines[i+1]
        counts[(obs, reason)] += 1
        i += 2
        continue
    i += 1

for (op, r), n in sorted(counts.items()):
    if op == "publish_plan_post":
        print(f"{op} reason={r} {n}")
