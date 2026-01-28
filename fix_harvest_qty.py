#!/usr/bin/env python3
"""Fix HarvestBrain to calculate qty from position_size_usd"""

with open("/opt/quantum/microservices/harvest_brain/harvest_brain.py", "r") as f:
    lines = f.readlines()

# Find line with qty extraction
for i, line in enumerate(lines):
    if "qty = float(exec_event.get('qty', 0))" in line:
        # Insert calculation code after entry_price line (i+2)
        insert_at = i + 3
        new_lines = [
            "\n",
            "            # If qty not provided, calculate from position_size_usd / entry_price\n",
            "            if qty == 0 and entry_price > 0:\n",
            "                position_size_usd = float(exec_event.get('position_size_usd', 0))\n",
            "                if position_size_usd > 0:\n",
            "                    qty = position_size_usd / entry_price\n",
            "                    logger.debug(f\"Calculated qty={qty:.4f} from ${position_size_usd:.2f}\")\n",
        ]
        lines = lines[:insert_at] + new_lines + lines[insert_at:]
        print(f"✅ Inserted qty calculation at line {insert_at}")
        break
else:
    print("❌ Could not find qty extraction line")
    exit(1)

with open("/opt/quantum/microservices/harvest_brain/harvest_brain.py", "w") as f:
    f.writelines(lines)

print("✅ Fixed HarvestBrain qty calculation")
