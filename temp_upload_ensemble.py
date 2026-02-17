import base64
import sys

# Read local file
with open(r'c:\quantum_trader\ai_engine\ensemble_manager.py', 'rb') as f:
    content = f.read()

# Encode to base64
encoded = base64.b64encode(content).decode('ascii')

# Print for SSH transfer
print(encoded)
