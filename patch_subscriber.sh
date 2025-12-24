#!/bin/bash
# Patch trade_intent_subscriber.py to disable ILF integration temporarily

FILE="/app/backend/events/subscribers/trade_intent_subscriber.py"

# Backup original
cp "$FILE" "${FILE}.bak"

# Patch: Comment out v35_integration import
sed -i 's/^from backend\.domains\.exits\.exit_brain_v3\.v35_integration import ExitBrainV35Integration$/# ILF disabled - import failed/' "$FILE"

# Patch: Disable exitbrain_v35 initialization  
sed -i 's/self\.exitbrain_v35 = ExitBrainV35Integration(enabled=True)/self.exitbrain_v35 = None  # Disabled/' "$FILE"

echo "✅ Patched trade_intent_subscriber.py"
