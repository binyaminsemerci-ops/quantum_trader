import pathlib
import sys
from typing import List, Set

art_root = pathlib.Path('artifacts_run_17930603338')
if not art_root.exists():
    print('Artifacts directory not found:', art_root)
    sys.exit(2)

results = {}
for sub in art_root.iterdir():
    if not sub.is_dir():
        continue
    before = sub / 'pip-list-before.txt'
    after = sub / 'pip-list-after.txt'
    if not before.exists() or not after.exists():
        print(f'Skipping {sub.name}: missing before/after files')
        continue
    b = {l.strip().split('==')[0].lower() for l in before.read_text().splitlines() if l.strip()}
    a = {l.strip().split('==')[0].lower() for l in after.read_text().splitlines() if l.strip()}
    added = sorted(a - b)
    removed = sorted(b - a)
    results[sub.name] = {'added': added, 'removed': removed, 'before_count': len(b), 'after_count': len(a)}

for name, data in results.items():
    print('---', name)
    print('before_count:', data['before_count'], 'after_count:', data['after_count'])
    if data['added']:
        print('added:')
        for p in data['added']:
            print('  ', p)
    else:
        print('added: (none)')
    if data['removed']:
        print('removed:')
        for p in data['removed']:
            print('  ', p)
    print()
