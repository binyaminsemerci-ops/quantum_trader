#!/usr/bin/env python3
"""Fix the sys.path in risk_proposal_publisher main.py"""

file_path = "/home/qt/quantum_trader/microservices/risk_proposal_publisher/main.py"

with open(file_path, 'r') as f:
    content = f.read()

# The current line adds parent directory (microservices), but we need quantum_trader root
old_line = '# Add parent directory to path\nsys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'

new_line = '''# Add quantum_trader root to path
_qt_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _qt_root)'''

if old_line in content:
    content = content.replace(old_line, new_line)
    with open(file_path, 'w') as f:
        f.write(content)
    print("✅ Fixed sys.path in risk_proposal_publisher/main.py")
else:
    if "_qt_root" in content:
        print("⏭️ Already fixed")
    else:
        print("⚠️ Could not find expected line to fix")
        print("Looking for:")
        print(old_line[:100])
