#!/usr/bin/env python3
"""
QSC MODE FIX: Make NHiTS FAIL-CLOSED on feature dimension mismatch
"""

file_path = 'ai_engine/agents/nhits_agent.py'

with open(file_path, 'r') as f:
    lines = f.readlines()

# Find the _match_sequence function
in_function = False
modified = False

for i, line in enumerate(lines):
    if 'def _match_sequence(self, sequence' in line:
        in_function = True
        continue
    
    if in_function and 'if sequence.shape[-1] < target_len:' in line:
        # Replace silent padding with FAIL-CLOSED exception
        indent = ' ' * (len(line) - len(line.lstrip()))
        
        # Insert FAIL-CLOSED logic
        lines[i] = f'{indent}if sequence.shape[-1] != target_len:\n'
        lines.insert(i+1, f'{indent}    raise ValueError(\n')
        lines.insert(i+2, f'{indent}        f"[NHITS] FAIL-CLOSED: Feature dimension mismatch {{sequence.shape[-1]}} != {{target_len}}. "\n')
        lines.insert(i+3, f'{indent}        f"Fix feature engineering to produce correct dimension. "\n')
        lines.insert(i+4, f'{indent}        f"Model expects {{target_len}} features, got {{sequence.shape[-1]}}"\n')
        lines.insert(i+5, f'{indent}    )\n')
        
        modified = True
        break

if modified:
    with open(file_path, 'w') as f:
        f.writelines(lines)
    print("✅ NHiTS FAIL-CLOSED enforcement ADDED")
    print("⚠️  System will now CRASH on feature mismatch (expected behavior)")
else:
    print("⚠️  Could not find feature dimension padding code")

print(f"✅ File checked: {file_path}")
