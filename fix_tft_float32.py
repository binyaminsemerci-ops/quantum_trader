#!/usr/bin/env python3
"""Fix TFT float32 JSON serialization issue"""

file_path = '/home/qt/quantum_trader/ai_engine/agents/unified_agents.py'

with open(file_path, 'r') as f:
    content = f.read()

# Find TFTAgent predict method and ensure float conversion
# Look for the return statement in TFTAgent
old_return = '''return action, confidence, {
            "model": f"v{self.model_version}",
            "predicted_pnl": predicted_pnl,
        }'''

new_return = '''return action, float(confidence), {  # Convert np.float32 to float
            "model": f"v{self.model_version}",
            "predicted_pnl": float(predicted_pnl),  # Ensure JSON serializable
        }'''

if old_return in content:
    content = content.replace(old_return, new_return)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Fixed TFT float32 serialization")
elif "float(confidence)" in content and "TFTAgent" in content:
    print("✅ Already fixed")
else:
    # Alternative approach - find the TFTAgent class and fix it
    import re
    
    # Find TFTAgent predict return
    pattern = r'(class TFTAgent.*?)(return action, confidence, \{.*?"predicted_pnl": predicted_pnl,.*?\})'
    
    if 'class TFTAgent' in content:
        # Simple fix - replace any np.float32 confidence
        old2 = "return action, confidence, {"
        new2 = "return action, float(confidence), {"
        
        if old2 in content:
            content = content.replace(old2, new2, 1)  # Only first instance might be wrong
            
        # Also fix predicted_pnl
        old3 = '"predicted_pnl": predicted_pnl,'
        new3 = '"predicted_pnl": float(predicted_pnl) if hasattr(predicted_pnl, "item") else predicted_pnl,'
        
        if old3 in content:
            content = content.replace(old3, new3)
            
        with open(file_path, 'w') as f:
            f.write(content)
        print("✅ Applied alternative float32 fix")
    else:
        print("⚠️ TFTAgent not found in file")

print("Done!")
