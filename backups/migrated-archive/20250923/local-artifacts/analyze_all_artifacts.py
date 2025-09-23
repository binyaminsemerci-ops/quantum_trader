import pathlib
root = pathlib.Path('.')
arts = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith('artifacts_run_')])
if not arts:
    print('No artifacts_run_* directories found')
    raise SystemExit(0)
for art in arts:
    print('\n==', art.name)
    results = {}
    for sub in art.iterdir():
        if not sub.is_dir():
            continue
        before = sub / 'pip-list-before.txt'
        after = sub / 'pip-list-after.txt'
        if not before.exists() or not after.exists():
            print('  Skipping', sub.name, '(missing before/after)')
            continue
        b = {l.strip().split('==')[0].lower() for l in before.read_text().splitlines() if l.strip()}
        a = {l.strip().split('==')[0].lower() for l in after.read_text().splitlines() if l.strip()}
        added = sorted(a - b)
        removed = sorted(b - a)
        results[sub.name] = {'added': added, 'removed': removed, 'before_count': len(b), 'after_count': len(a)}
    if not results:
        print('  No pip-list artifacts in', art.name)
        continue
    for name,data in results.items():
        print('  ---', name)
        print('    before_count:', data['before_count'], 'after_count:', data['after_count'])
        if data['added']:
            print('    added:', ', '.join(data['added'][:10]) + (', ...' if len(data['added'])>10 else ''))
        else:
            print('    added: (none)')
        if data['removed']:
            print('    removed:', ', '.join(data['removed'][:10]) + (', ...' if len(data['removed'])>10 else ''))
