#!/usr/bin/env python3
"""Print exact repr of daemon lines to verify anchor"""
with open("/home/qt/quantum_trader/microservices/rl_sizing_agent/rl_agent_daemon.py") as f:
    lines = f.readlines()
for i, l in enumerate(lines[162:202], start=163):
    print(f"{i}: {repr(l)}")
