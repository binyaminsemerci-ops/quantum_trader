import re
path = "/opt/quantum/microservices/exit_management_agent/perception.py"
with open(path) as f:
    txt = f.read()

# Show return PerceptionResult block
idx = txt.find("        return PerceptionResult(")
print("=== return block ===")
print(repr(txt[idx:idx+600]))
print()

# Show imports section
idx2 = txt.find("from .models import")
print("=== imports ===")
print(repr(txt[idx2:idx2+200]))
print()

# Show __init__ body (peak_prices line)
idx3 = txt.find("self._peak_prices")
print("=== _peak_prices init ===")
print(repr(txt[idx3:idx3+150]))
print()

# Show forget method
idx4 = txt.find("def forget(")
print("=== forget() ===")
print(repr(txt[idx4:idx4+200]))
