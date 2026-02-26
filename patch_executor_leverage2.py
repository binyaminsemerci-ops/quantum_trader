"""Fix only anchor2 (plan_leverage extraction) — run after patch_executor_leverage.py"""
import sys

path = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(path, "r") as f:
    src = f.read()

# Find the exact text around the reduce_only line
idx = src.find('reduce_only = reduce_only_str in ("true", "1", "yes")')
if idx == -1:
    print("FATAL: reduce_only line not found at all")
    sys.exit(1)

snippet = src[idx:idx+200]
print("SNIPPET:", repr(snippet))

# Already applied?
if "plan_leverage" in src:
    print("CHANGE2 already present — skipping")
    sys.exit(0)

old2 = (
    'reduce_only = reduce_only_str in ("true", "1", "yes")\n'
    '\n'
    '            # Log warning if field was missing (indicates old/malformed plan)'
)
new2 = (
    'reduce_only = reduce_only_str in ("true", "1", "yes")\n'
    '\n'
    '            # Extract leverage from plan (LeverageEngine value via intent-bridge)\n'
    '            leverage_str = event_data.get(b"leverage", b"1").decode()\n'
    '            try:\n'
    '                plan_leverage = max(1, min(125, int(float(leverage_str))))\n'
    '            except (ValueError, TypeError):\n'
    '                plan_leverage = 1\n'
    '\n'
    '            # Log warning if field was missing (indicates old/malformed plan)'
)

if old2 in src:
    src = src.replace(old2, new2, 1)
    print("CHANGE2 applied")
    with open(path, "w") as f:
        f.write(src)
    print("File written OK")
else:
    print("ANCHOR2 not found with exact text, trying fallback...")
    # Try inserting after the reduce_only line directly
    marker = 'reduce_only = reduce_only_str in ("true", "1", "yes")'
    insert_after = (
        '\n'
        '\n'
        '            # Extract leverage from plan (LeverageEngine value via intent-bridge)\n'
        '            leverage_str = event_data.get(b"leverage", b"1").decode()\n'
        '            try:\n'
        '                plan_leverage = max(1, min(125, int(float(leverage_str))))\n'
        '            except (ValueError, TypeError):\n'
        '                plan_leverage = 1\n'
    )
    pos = src.find(marker)
    if pos != -1:
        end = pos + len(marker)
        src = src[:end] + insert_after + src[end:]
        print("CHANGE2 fallback applied")
        with open(path, "w") as f:
            f.write(src)
        print("File written OK")
    else:
        print("FATAL: cannot apply CHANGE2")
        sys.exit(1)
