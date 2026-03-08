with open(r'c:\quantum_trader\ops\retrain\train_tft_v7.py', 'r', encoding='ascii') as f:
    lines = f.readlines()

# Binary search for syntax error
lo, hi = 0, len(lines)
while lo < hi - 1:
    mid = (lo + hi) // 2
    chunk = ''.join(lines[:mid])
    try:
        compile(chunk, 'test.py', 'exec')
        lo = mid
    except SyntaxError:
        hi = mid

print(f"Error first appears between line {lo} and {hi}")
# Show lines around hi
for i in range(max(0, hi-5), min(len(lines), hi+3)):
    print(f"L{i+1}: {repr(lines[i])}")

# Try to compile up to hi
try:
    compile(''.join(lines[:hi]), 'test.py', 'exec')
    print("OK at", hi)
except SyntaxError as e:
    print(f"ERROR at {hi}: line={e.lineno}, col={e.offset}: {e.msg}")
    print(f"  text: {repr(e.text)}")
