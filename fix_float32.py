#!/usr/bin/env python3
"""Fix float32 JSON serialization issue in ensemble_manager.py"""

file_path = "/home/qt/quantum_trader/ai_engine/ensemble_manager.py"

with open(file_path, "r") as f:
    content = f.read()

# Track changes
changes = 0

# Fix 1: Line ~1134 - wrap v.get('confidence') in float()
old1 = "'confidence': v.get('confidence')"
new1 = "'confidence': float(v.get('confidence', 0.5))"
if old1 in content:
    content = content.replace(old1, new1)
    changes += 1
    print(f"✅ Fixed: {old1}")

# Fix 2: Line ~1136 - wrap v[1] in float()
old2 = "'confidence': v[1]"
new2 = "'confidence': float(v[1])"
if old2 in content:
    content = content.replace(old2, new2)
    changes += 1
    print(f"✅ Fixed: {old2}")

# Fix 3: Line ~930 - wrap conf in float()
old3 = "'confidence': conf,"
new3 = "'confidence': float(conf) if conf is not None else 0.5,"
if old3 in content:
    content = content.replace(old3, new3)
    changes += 1
    print(f"✅ Fixed: {old3}")

# Fix 4: Line ~719 - pred.get('confidence')
old4 = "'confidence': pred.get('confidence', 0.5)"
new4 = "'confidence': float(pred.get('confidence', 0.5))"
if old4 in content:
    content = content.replace(old4, new4)
    changes += 1
    print(f"✅ Fixed: {old4}")

# Fix 5: Line ~724 - pred[1]
old5 = "'confidence': pred[1]"
new5 = "'confidence': float(pred[1])"
if old5 in content:
    content = content.replace(old5, new5)
    changes += 1
    print(f"✅ Fixed: {old5}")

if changes > 0:
    with open(file_path, "w") as f:
        f.write(content)
    print(f"\n✅ Applied {changes} fixes to ensemble_manager.py")
else:
    print("⚠️ No changes needed (already fixed or patterns not found)")
