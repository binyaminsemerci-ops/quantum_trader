#!/usr/bin/env python3
"""Clean zombie Redis stream consumers."""
import subprocess

STREAM = "quantum:stream:apply.result"
GROUP = "quantum:group:execution:trade.intent"

def redis_cmd(*args):
    result = subprocess.run(["redis-cli"] + list(args), capture_output=True, text=True)
    return result.stdout

# Get all consumers
output = redis_cmd("XINFO", "CONSUMERS", STREAM, GROUP)
lines = [l.strip() for l in output.split("\n")]

# Parse consumers
consumers = []
for i, line in enumerate(lines):
    if line == "name" and i + 1 < len(lines):
        consumers.append(lines[i + 1])

print(f"Total consumers: {len(consumers)}")

# Find active (highest PID = newest)
def get_pid(name):
    try:
        return int(name.split("-")[-1])
    except:
        return 0

if not consumers:
    print("No consumers found")
    exit(0)

active = max(consumers, key=get_pid)
print(f"Active (keeping): {active}")

# Delete zombies
deleted = 0
for c in consumers:
    if c != active:
        print(f"Delete: {c}")
        redis_cmd("XGROUP", "DELCONSUMER", STREAM, GROUP, c)
        deleted += 1

print(f"\nDeleted {deleted} zombie consumers")
print(f"Remaining: {len(consumers) - deleted}")
