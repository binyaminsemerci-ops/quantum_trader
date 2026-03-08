import py_compile, sys, tokenize, io

with open(r'c:\quantum_trader\ops\retrain\train_tft_v7.py', 'r', encoding='ascii') as f:
    lines = f.readlines()

# Print lines around error
for i in range(430, 441):
    if i < len(lines):
        print(f"L{i+1}: {repr(lines[i])}")

# Try compile
try:
    compile(''.join(lines), 'train_tft_v7.py', 'exec')
    print("COMPILE_OK")
except SyntaxError as e:
    print(f"SYNTAX_ERROR: line {e.lineno}, col {e.offset}: {e.msg}")
    print(f"  text: {repr(e.text)}")
