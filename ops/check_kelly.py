path = '/opt/quantum/microservices/layer4_portfolio_optimizer/layer4_portfolio_optimizer.py'
src = open(path).read()
marker = 'def kelly_fraction'
idx = src.find(marker)
if idx >= 0:
    print("=== kelly_fraction FUNCTION ===")
    print(src[idx:idx+400])
else:
    print("ERROR: def kelly_fraction not found!")
    # Show what's near kelly_fraction references
    for i, line in enumerate(src.splitlines()):
        if 'kelly' in line.lower():
            print(f"  line {i+1}: {line}")
