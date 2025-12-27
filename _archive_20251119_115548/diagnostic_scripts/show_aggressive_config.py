"""Display aggressive configuration status"""
import os

print("\n" + "="*60)
print("[OK] AGGRESSIVE KONFIGURASJON AKTIVERT")
print("="*60)

print("\n[CHART] TRADING SETTINGS:")
leverage = os.getenv("QT_DEFAULT_LEVERAGE", "10")
max_pos = os.getenv("QT_MAX_POSITIONS", "8")
per_trade = os.getenv("QT_MAX_NOTIONAL_PER_TRADE", "250")
total_exp = os.getenv("QT_MAX_GROSS_EXPOSURE", "2000")
pos_size = float(total_exp) * float(leverage)

print(f"   Leverage: {leverage}x")
print(f"   Max Positions: {max_pos}")
print(f"   Per Trade: ${per_trade}")
print(f"   Total Exposure: ${total_exp}")
print(f"   Position Size (w/leverage): ${pos_size:.0f}")

print("\n[TARGET] TP/SL SETTINGS (TIGHTER):")
tp = float(os.getenv("QT_TP_PCT", "0.5")) * 100
sl = float(os.getenv("QT_SL_PCT", "0.75")) * 100
trail = float(os.getenv("QT_TRAIL_PCT", "0.2")) * 100
partial = float(os.getenv("QT_PARTIAL_TP", "0.6")) * 100

print(f"   Take Profit: {tp:.1f}%")
print(f"   Stop Loss: {sl:.2f}%")
print(f"   Trailing: {trail:.1f}%")
print(f"   Partial TP: {partial:.0f}%")

print("\n🤖 AI SETTINGS (MORE AGGRESSIVE):")
conf = float(os.getenv("QT_CONFIDENCE_THRESHOLD", "0.35")) * 100
check = os.getenv("QT_CHECK_INTERVAL", "10")
cooldown = os.getenv("QT_COOLDOWN_SECONDS", "120")

print(f"   Confidence Threshold: {conf:.0f}%")
print(f"   Check Interval: {check}s")
print(f"   Cooldown: {cooldown}s")

print("\n[MONEY] PROFIT CALCULATION:")
print(f"   Position Size: ${pos_size:.0f} (with {leverage}x leverage)")
print(f"   For $1,500 profit: Trenger {(1500/pos_size)*100:.1f}% movement")
print(f"   Timeframe: 14 timer (til kl 11:00)")
print(f"   Strategy: Flere trades, raskere exits")

print("\n[CHART] DASHBOARD:")
print("   🌐 qt-agent-ui: http://localhost:5174")
print("   [OK] Backend API: http://localhost:8000")

print("\n[WARNING]  VIKTIG RISIKO ADVARSEL:")
print(f"   - {leverage}x leverage = {leverage}x profit OG {leverage}x tap!")
print("   - $1,500 profit mulig, men også $1,500+ tap")
print("   - Følg nøye med på dashboard!")

print("\n[OK] Alt kjører nå - overvåk på dashboard!")
print("="*60 + "\n")
