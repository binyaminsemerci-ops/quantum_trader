#!/usr/bin/env python3
"""Add TFT prediction code to ensemble_manager.py predict method"""

with open('/home/qt/quantum_trader/ai_engine/ensemble_manager.py', 'r') as f:
    content = f.read()

# The code to insert
tft_predict_code = '''        # TFT: Only predict if agent is loaded
        if self.tft_agent is not None:
            try:
                predictions['tft'] = self.tft_agent.predict(symbol, features)
            except Exception as e:
                logger.error(f"TFT prediction failed: {e} - excluding from ensemble (FAIL-CLOSED)")
                # Don't add to predictions - let ensemble work with remaining models

'''

# Insert before QSC FAIL-CLOSED line
if "predictions['tft']" not in content:
    content = content.replace(
        "        # 🔍 QSC FAIL-CLOSED: Exclude degraded models from voting",
        tft_predict_code + "        # 🔍 QSC FAIL-CLOSED: Exclude degraded models from voting"
    )
    print("✅ Added TFT predict code")
else:
    print("⚠️ TFT predict code already exists")

# Also update the CHART display line to include TFT - skip for now, predictions work first
print("Skipping CHART display update")

with open('/home/qt/quantum_trader/ai_engine/ensemble_manager.py', 'w') as f:
    f.write(content)

# Verify
with open('/home/qt/quantum_trader/ai_engine/ensemble_manager.py', 'r') as f:
    new_content = f.read()
    if "predictions['tft']" in new_content:
        print("✅ Verified TFT predict code")
    else:
        print("❌ TFT predict code not found")

print("Done!")
