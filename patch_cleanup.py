"""Remove duplicate plan_leverage block and do final syntax check."""
import py_compile, sys

path = "/home/qt/quantum_trader/microservices/intent_executor/main.py"
with open(path) as f:
    src = f.read()

dup = (
    '\n\n            # Extract leverage from plan (LeverageEngine via intent-bridge)\n'
    '            _lev_str = event_data.get(b"leverage", b"1").decode()\n'
    '            try:\n'
    '                plan_leverage = max(1, min(125, int(float(_lev_str))))\n'
    '            except (ValueError, TypeError):\n'
    '                plan_leverage = 1'
)

if dup in src:
    src = src.replace(dup, '', 1)
    with open(path, 'w') as f:
        f.write(src)
    print("Duplicate removed")
else:
    print("No duplicate found (already clean)")

# Syntax check
try:
    py_compile.compile(path, doraise=True)
    print("Syntax OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX ERROR: {e}")
    sys.exit(1)
