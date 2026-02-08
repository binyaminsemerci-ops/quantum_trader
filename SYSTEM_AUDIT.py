#!/usr/bin/env python3
"""
FULL SYSTEM AUDIT - Map entire trading flow
Find all components that touch TP/SL and position management
"""

import os
from pathlib import Path

print("="*80)
print("QUANTUM TRADER SYSTEM AUDIT")
print("="*80)
print()

# Find all Python files with relevant keywords
keywords = {
    'stop_loss': [],
    'take_profit': [],
    'position_size': [],
    'leverage': [],
    'close': [],
    'exit': []
}

base_dir = Path('.')
for py_file in base_dir.rglob('*.py'):
    if 'venv' in str(py_file) or '__pycache__' in str(py_file):
        continue
    
    try:
        content = py_file.read_text(encoding='utf-8', errors='ignore')
        for keyword in keywords:
            if keyword in content.lower():
                keywords[keyword].append(str(py_file))
    except:
        pass

print("FILES THAT HANDLE TP/SL/POSITION MANAGEMENT:")
print("-"*80)
for keyword, files in keywords.items():
    unique_files = set(files)
    print(f"\n{keyword.upper()} mentioned in {len(unique_files)} files:")
    for f in sorted(unique_files)[:10]:  # Top 10
        print(f"  - {f}")

print()
print("="*80)
print()

# Find active services
print("ACTIVE MICROSERVICES:")
print("-"*80)

services_dir = Path('microservices')
if services_dir.exists():
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir() and not service_dir.name.startswith('.'):
            main_files = list(service_dir.glob('*.py'))
            if main_files:
                print(f"  - {service_dir.name}/")
                for f in main_files[:3]:
                    print(f"      {f.name}")

print()
print("="*80)
