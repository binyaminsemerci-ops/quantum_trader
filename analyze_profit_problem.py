"""
Analyser HVORFOR systemet taper penger
"""
import subprocess
import re
from collections import defaultdict

def run_docker_command(cmd):
    """Run docker command and return output"""
    result = subprocess.run(
        f'docker logs quantum_backend 2>&1 | {cmd}',
        shell=True,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    return result.stdout

print("=" * 80)
print("🔍 PROFIT PROBLEM ANALYSE")
print("=" * 80)

# 1. Sjekk Take Profit nivåer
print("\n1️⃣ TAKE PROFIT NIVÅER (siste 10 trades):")
print("-" * 80)
tp_data = run_docker_command('Select-String "TP: \\$" | Select-Object -Last 10')
for line in tp_data.split('\n'):
    if 'TP:' in line:
        # Extract TP percentage
        match = re.search(r'TP:.*?\(([\+\-]\d+\.\d+)%\)', line)
        symbol_match = re.search(r'(TRADE APPROVED:|Order placed:)\s+(\w+)', line)
        if match and symbol_match:
            tp_pct = float(match.group(1))
            symbol = symbol_match.group(2)
            print(f"   {symbol}: TP = {tp_pct:+.2f}%")

# 2. Sjekk Stop Loss nivåer
print("\n2️⃣ STOP LOSS NIVÅER (siste 10 trades):")
print("-" * 80)
sl_data = run_docker_command('Select-String "SL: \\$" | Select-Object -Last 10')
for line in sl_data.split('\n'):
    if 'SL:' in line:
        # Extract SL percentage
        match = re.search(r'SL:.*?\(([\+\-]\d+\.\d+)%\)', line)
        symbol_match = re.search(r'(TRADE APPROVED:|Order placed:)\s+(\w+)', line)
        if match and symbol_match:
            sl_pct = float(match.group(1))
            symbol = symbol_match.group(2)
            print(f"   {symbol}: SL = {sl_pct:+.2f}%")

# 3. Sjekk faktiske PnL fra closed trades
print("\n3️⃣ LUKKEDE POSISJONER (siste 24t):")
print("-" * 80)
closed_trades = run_docker_command('Select-String "Position closed|realizedPnl" | Select-Object -Last 20')
winners = 0
losers = 0
total_pnl = 0.0

for line in closed_trades.split('\n'):
    pnl_match = re.search(r'realizedPnl["\']?\s*:\s*["\']?([\-\d\.]+)', line)
    if pnl_match:
        pnl = float(pnl_match.group(1))
        total_pnl += pnl
        if pnl > 0:
            winners += 1
            print(f"   ✅ Profit: +${pnl:.2f}")
        else:
            losers += 1
            print(f"   ❌ Loss: ${pnl:.2f}")

if winners + losers > 0:
    winrate = (winners / (winners + losers)) * 100
    print(f"\n   📊 Win Rate: {winrate:.1f}% ({winners}W / {losers}L)")
    print(f"   💰 Total PnL: ${total_pnl:.2f}")
else:
    print("   ⚠️  Ingen lukkede trades funnet i loggene")

# 4. Sjekk nåværende åpne posisjoner
print("\n4️⃣ ÅPNE POSISJONER (current PnL):")
print("-" * 80)
current_positions = run_docker_command('Select-String "\\[CHART\\].*PnL" | Select-Object -Last 10')
for line in current_positions.split('\n'):
    match = re.search(r'\[CHART\]\s+(\w+):\s+PnL\s+([\-\d\.]+)%', line)
    if match:
        symbol = match.group(1)
        pnl_pct = float(match.group(2))
        status = "🟢" if pnl_pct > 0 else "🔴"
        print(f"   {status} {symbol}: {pnl_pct:+.2f}%")

# 5. Sjekk R:R ratio
print("\n5️⃣ RISK:REWARD RATIO (siste 10 trades):")
print("-" * 80)
rr_data = run_docker_command('Select-String "R:R =" | Select-Object -Last 10')
rr_values = []
for line in rr_data.split('\n'):
    match = re.search(r'R:R\s*=\s*([\d\.]+)', line)
    if match:
        rr = float(match.group(1))
        rr_values.append(rr)
        symbol_match = re.search(r'(\w+USDT)', line)
        symbol = symbol_match.group(1) if symbol_match else "Unknown"
        print(f"   {symbol}: R:R = {rr:.2f}")

if rr_values:
    avg_rr = sum(rr_values) / len(rr_values)
    print(f"\n   📊 Average R:R: {avg_rr:.2f}")

# 6. Sjekk confidence levels
print("\n6️⃣ AI CONFIDENCE NIVÅER:")
print("-" * 80)
conf_data = run_docker_command('Select-String "confidence.*%" | Select-Object -Last 10')
conf_values = []
for line in conf_data.split('\n'):
    match = re.search(r'confidence[:\s]+([\d\.]+)%', line, re.IGNORECASE)
    if match:
        conf = float(match.group(1))
        conf_values.append(conf)
        symbol_match = re.search(r'(\w+USDT)', line)
        symbol = symbol_match.group(1) if symbol_match else "Unknown"
        status = "🟢" if conf >= 50 else "🟡" if conf >= 45 else "🔴"
        print(f"   {status} {symbol}: {conf:.1f}%")

if conf_values:
    avg_conf = sum(conf_values) / len(conf_values)
    print(f"\n   📊 Average Confidence: {avg_conf:.1f}%")

print("\n" + "=" * 80)
print("🎯 KONKLUSJON:")
print("=" * 80)

# Identify problems
problems = []

if rr_values and avg_rr < 2.0:
    problems.append(f"⚠️  R:R ratio for lav ({avg_rr:.2f}) - bør være 2:1 eller bedre")

if conf_values and avg_conf < 50:
    problems.append(f"⚠️  AI confidence for lav ({avg_conf:.1f}%) - bør være 50%+")

if losers > winners and winners + losers > 0:
    problems.append(f"⚠️  Flere tap enn gevinster ({losers}L vs {winners}W)")

if total_pnl < 0:
    problems.append(f"⚠️  Negativ total profit (${total_pnl:.2f})")

if problems:
    for problem in problems:
        print(problem)
    
    print("\n📝 ANBEFALINGER:")
    if avg_rr < 2.0:
        print("   1. Øk TP nivåer (f.eks. 2-3% i stedet for 0.3%)")
        print("   2. Reduser SL nivåer (tighter stops)")
    if avg_conf < 50:
        print("   3. Øk minimum confidence threshold til 50-55%")
    if losers > winners:
        print("   4. Bruk strengere entry filters")
else:
    print("✅ Ingen kritiske problemer funnet - fortsett monitoring")

print("=" * 80)
