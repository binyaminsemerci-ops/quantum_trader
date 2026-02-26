#!/usr/bin/env python3
"""
TWO FIXES for harvest_brain duplicate order problem:

FIX-1: DedupManager.build_key() — remove r_level and unrealized_pnl from key.
  Both are floats that change every tick → unique key every second → dedup bypass.
  Result was 20 identical ACTUSDT orders at 00:29:03 with $0.035 profit each.
  Fix: key = symbol:intent_type only. TTL=900s prevents re-fire for 15 min.

FIX-2: Minimum notional guard in evaluate() — skip harvest if:
  qty * current_price < HARVEST_MIN_NOTIONAL (default $5.0)
  Prevents sub-$1 orders that burn fees for nothing.
  EMERGENCY_SL_CLOSE is exempt (always executes regardless of size).
"""
import py_compile
import sys

path = "/opt/quantum/microservices/harvest_brain/harvest_brain.py"

with open(path) as f:
    lines = f.readlines()

content = "".join(lines)
fixes = []

# ── FIX-1: Dedup key ─────────────────────────────────────────────────────────
# Find build_key method and replace the else branch
new_lines = []
i = 0
in_build_key = False
brace_depth = 0
replaced_dedup = False

while i < len(lines):
    line = lines[i]

    if not replaced_dedup and 'def build_key(self, intent: HarvestIntent)' in line:
        in_build_key = True
        # Collect entire method
        method_lines = [line]
        i += 1
        while i < len(lines):
            l = lines[i]
            method_lines.append(l)
            i += 1
            # Method ends when next def/class at same indent level is found
            if l.strip() and not l.strip().startswith('#') and not l[0].isspace() and l.strip().startswith('def '):
                # Overshot — put this line back
                i -= 1
                method_lines.pop()
                break
            # Or: next method at same indentation (4 spaces)
            if l.startswith('    def ') and 'build_key' not in l:
                i -= 1
                method_lines.pop()
                break

        # Replace with fixed version
        indent = '    '
        new_method = (
            f'{indent}def build_key(self, intent: HarvestIntent) -> str:\n'
            f'{indent}    """Build dedup key from intent.\n'
            f'{indent}    Key = symbol:intent_type ONLY.\n'
            f'{indent}    r_level and unrealized_pnl excluded: both change every tick\n'
            f'{indent}    (float drift), bypassing dedup and causing 20+ duplicate orders.\n'
            f'{indent}    TTL (900s default) prevents same action re-firing within 15 min.\n'
            f'{indent}    """\n'
            f'{indent}    return f"quantum:dedup:harvest:{{intent.symbol}}:{{intent.intent_type}}"\n'
        )
        new_lines.append(new_method)
        replaced_dedup = True
        fixes.append("FIX-1: dedup key = symbol:intent_type only")
        continue

    new_lines.append(line)
    i += 1

if not replaced_dedup:
    print("FIX-1 FAILED: build_key method not found")
    sys.exit(1)

content = "".join(new_lines)

# ── FIX-2: Minimum notional guard ────────────────────────────────────────────
# Add guard right before "return intents" at the end of evaluate()
# Find the "return intents" that closes the evaluate() method (not other returns)
OLD_RETURN = '        return intents\n\n\n# =='
NEW_RETURN = (
    '        # ── Minimum notional guard ──────────────────────────────────────\n'
    '        # Skip harvest orders too small to matter (fees > profit).\n'
    '        # EMERGENCY_SL_CLOSE always executes regardless of size.\n'
    '        min_notional = float(os.getenv("HARVEST_MIN_NOTIONAL", "5.0"))\n'
    '        filtered = []\n'
    '        for _intent in intents:\n'
    '            if _intent.intent_type == "EMERGENCY_SL_CLOSE":\n'
    '                filtered.append(_intent)  # always allow SL\n'
    '            elif _intent.qty * position.current_price >= min_notional:\n'
    '                filtered.append(_intent)\n'
    '            else:\n'
    '                logger.debug(\n'
    '                    f"SKIP_MIN_NOTIONAL {position.symbol}: "\n'
    '                    f"notional={_intent.qty * position.current_price:.2f} < {min_notional}"\n'
    '                )\n'
    '        return filtered\n'
    '\n'
    '\n# =='
)

if OLD_RETURN in content:
    content = content.replace(OLD_RETURN, NEW_RETURN, 1)
    fixes.append("FIX-2: minimum notional guard added ($5.0 default, env: HARVEST_MIN_NOTIONAL)")
else:
    print("FIX-2 FAILED: anchor not found")
    idx = content.rfind('        return intents')
    if idx >= 0:
        print(f"Context around last 'return intents': {repr(content[idx:idx+80])}")
    sys.exit(1)

with open(path, "w") as f:
    f.write(content)

print("Fixes applied:")
for f in fixes:
    print(f"  {f}")

try:
    py_compile.compile(path, doraise=True)
    print("SYNTAX_OK")
except py_compile.PyCompileError as e:
    print(f"SYNTAX_ERROR: {e}")
    sys.exit(1)
