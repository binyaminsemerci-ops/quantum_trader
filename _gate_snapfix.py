#!/usr/bin/env python3
"""
Fix intent_executor Gate 0+1 to use snapshot keys instead of direct-hash keys.
The old gate counted quantum:position:{symbol} hash keys which were deleted.
New gate: 
  Gate 0 - block if same symbol snapshot has position_amt != 0
  Gate 1 - count snapshots with |position_amt| > 0, block if >= 10
"""
import re

filepath = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(filepath, "r") as f:
    content = f.read()

# Find the block between "# Gate 1" and "# Gate 2"
pattern = r'([ \t]+# Gate 1: max 10 active positions\n.*?)(\n[ \t]+# Gate 2:)'
match = re.search(pattern, content, re.DOTALL)
if not match:
    print("ERROR: Could not find Gate 1 block via regex")
    # Show lines near "Gate 1" for debugging
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'Gate 1' in line or 'Gate 2' in line or 'Gate 3' in line:
            print(f"  Line {i+1}: {repr(line)}")
    exit(1)

print(f"Found Gate 1 block ({len(match.group(1))} chars)")
print("Block content:")
print(match.group(1))

# Get indentation from first line of block
first_line = match.group(1).split('\n')[0]
indent = len(first_line) - len(first_line.lstrip())
pad = ' ' * indent

new_gate01 = f"""{pad}# Gate 0: block if same symbol already has an open position (prevents doubling/churning)
{pad}_snap_key = f"quantum:position:snapshot:{{symbol}}"
{pad}_snap_amt_raw = self.redis.hget(_snap_key, "position_amt")
{pad}try:
{pad}    _snap_amt = float(_snap_amt_raw) if _snap_amt_raw else 0.0
{pad}except (ValueError, TypeError):
{pad}    _snap_amt = 0.0
{pad}if abs(_snap_amt) > 0.0:
{pad}    logger.warning(f"\\U0001f6ab SYMBOL_OPEN {{symbol}}: already open position_amt={{_snap_amt}} — REJECTED")
{pad}    self._write_result(plan_id, symbol, executed=False,
{pad}                       error=f"symbol_already_open_{{_snap_amt}}",
{pad}                       side=side, qty=qty_to_use)
{pad}    self._mark_done(plan_id)
{pad}    return True

{pad}# Gate 1: max 10 active positions (uses Binance-confirmed snapshot keys)
{pad}_snap_keys = self.redis.keys("quantum:position:snapshot:*")
{pad}_open_count = 0
{pad}for _sk in _snap_keys:
{pad}    try:
{pad}        _amt_raw = self.redis.hget(_sk, "position_amt")
{pad}        if _amt_raw and abs(float(_amt_raw)) > 0.0:
{pad}            _open_count += 1
{pad}    except (ValueError, TypeError):
{pad}        pass
{pad}if _open_count >= 10:
{pad}    logger.warning(f"\\U0001f6ab POSITION_LIMIT {{symbol}}: {{_open_count}}/10 active — REJECTED")
{pad}    self._write_result(plan_id, symbol, executed=False,
{pad}                       error=f"position_limit_{{_open_count}}/10",
{pad}                       side=side, qty=qty_to_use)
{pad}    self._mark_done(plan_id)
{pad}    return True"""

# Replace the old Gate 1 block
new_content = content[:match.start(1)] + new_gate01 + match.group(2) + content[match.end(2):]

# Write backup
with open(filepath + ".bak_snapgate", "w") as f:
    f.write(content)
print(f"Backup saved: {filepath}.bak_snapgate")

with open(filepath, "w") as f:
    f.write(new_content)
print("SUCCESS: Gate 0 (symbol dedup) + Gate 1 (snapshot-based 10-pos limit) deployed")
print(f"Old code ({len(match.group(1))} chars) replaced with {len(new_gate01)} chars")
