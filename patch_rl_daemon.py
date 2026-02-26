#!/usr/bin/env python3
"""Patch rl_agent_daemon.py: fix wrong field names causing reward=-1.0"""
import shutil

path = "/home/qt/quantum_trader/microservices/rl_sizing_agent/rl_agent_daemon.py"
backup = path + ".bak"

with open(path) as f:
    src = f.read()

# Verify the bug is present
assert 'fields.get("realized_pnl", 0)' in src, "realized_pnl not found — already patched?"
assert 'fields.get("close_price", 0)' in src, "close_price not found — already patched?"

# Backup
shutil.copy(path, backup)

# Fix: use correct field names matching quantum:stream:trade.closed format
src = src.replace('fields.get("realized_pnl", 0)', 'fields.get("pnl_usd", 0)')
src = src.replace('fields.get("close_price", 0)', 'fields.get("exit_price", 0)')

with open(path, "w") as f:
    f.write(src)

# Verify
with open(path) as f:
    result = f.read()

assert 'fields.get("pnl_usd", 0)' in result
assert 'fields.get("exit_price", 0)' in result
assert 'fields.get("realized_pnl", 0)' not in result
assert 'fields.get("close_price", 0)' not in result

print("✅ Patched successfully:")
print("  realized_pnl → pnl_usd")
print("  close_price  → exit_price")
print(f"  Backup: {backup}")
