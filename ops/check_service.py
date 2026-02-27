import os

# Check service file
with open('/tmp/quantum-paper-trade-controller.service', 'rb') as f:
    data = f.read()
print(repr(data[:400]))
print("has_cr:", b'\r' in data)
print("lines with cr:", sum(1 for l in data.split(b'\n') if b'\r' in l))
