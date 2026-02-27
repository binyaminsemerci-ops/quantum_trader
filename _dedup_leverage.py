PATH = '/home/qt/quantum_trader/microservices/intent_executor/main.py'
with open(PATH, 'r') as f:
    src = f.read()

# Find and remove the first (old) duplicate leverage extraction block
OLD = (
    '            # Extract leverage from plan (LeverageEngine value via intent-bridge)\n'
    '            leverage_str = event_data.get(b"leverage", b"1").decode()\n'
    '            try:\n'
    '                plan_leverage = max(1, min(125, int(float(leverage_str))))\n'
    '            except (ValueError, TypeError):\n'
    '                plan_leverage = 1\n'
)

if OLD in src:
    count = src.count(OLD)
    print(f"Found {count} occurrence(s). Removing first one...")
    src = src.replace(OLD, '', 1)
    with open(PATH, 'w') as f:
        f.write(src)
    print("Duplicate removed OK")
else:
    idx = src.find('LeverageEngine value via intent-bridge')
    if idx >= 0:
        print("Found block at:", idx)
        chunk = src[idx-5:idx+350]
        print(repr(chunk))
    else:
        print("Old block not found - may already be removed")
        idx2 = src.find('_lev_str')
        print("New block (_lev_str) at:", idx2)
