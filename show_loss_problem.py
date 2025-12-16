"""Show the massive loss percentage on LONG positions"""

print("\n" + "=" * 80)
print("[WARNING] CRITICAL: MASSIVE LOSS ON LONG POSITIONS")
print("=" * 80)

print("\n[CHART] From your Binance UI data:\n")

print("1️⃣ APTUSDT LONG:")
print("   Margin:      175.72 BNFCR")
print("   Unrealized:  -19.88 BNFCR")
print("   Loss:        -11.31% 📉")
print()
print("   Price movement: Entry $2.87548 → Mark $2.87691")
print("   That's only +0.05% price move UP (should be profit!)")
print("   But showing -11.31% loss on margin! [WARNING]")

print("\n2️⃣ SOLUSDT LONG:")
print("   Margin:      141.71 BNFCR")
print("   Unrealized:  -22.11 BNFCR")
print("   Loss:        -15.60% 📉")
print()
print("   Price movement: Entry $138.17 → Mark $138.239")
print("   That's only +0.05% price move UP (should be profit!)")
print("   But showing -15.60% loss on margin! [WARNING]")

print("\n" + "=" * 80)
print("[SEARCH] TOTAL DAMAGE:")
print("=" * 80)

total_margin = 175.72 + 141.71
total_loss = -19.88 + -22.11
total_loss_pct = (total_loss / total_margin) * 100

print(f"\nCombined LONG positions:")
print(f"   Total Margin:  {total_margin:.2f} BNFCR")
print(f"   Total Loss:    {total_loss:.2f} BNFCR")
print(f"   Loss %:        {total_loss_pct:.2f}% 📉📉📉")

print("\n" + "=" * 80)
print("❓ WHY IS THIS HAPPENING?")
print("=" * 80)

print("""
Dette er IKKE normale fees! Med 20x leverage:

[RED_CIRCLE] Mulige årsaker til -11% til -16% tap på +0.05% pris-bevegelse:

1. 💸 FUNDING FEES (mest sannsynlig)
   → Long posisjon betaler funding hver 8. time
   → Med høy funding rate kan dette være 0.01-0.05% per 8h
   → Over tid: Store summer med 20x leverage
   
2. [WARNING] ENTRY SLIPPAGE
   → Kanskje du ble filled til dårligere pris enn vist?
   → Entry price kan være feil registrert
   
3. 🔻 LIQUIDATION FEES
   → Noen ganger hvis posisjon var nær liquidation før
   
4. [CHART] UNREALIZED P&L CALCULATION BUG
   → Binance UI kan ha feil i beregningen

MEN: Dette er UNORMALT høyt for bare 0.05% pris-bevegelse!
""")

print("\n" + "=" * 80)
print("[TARGET] SAMMENLIGNING MED SHORT POSISJONER:")
print("=" * 80)

print("\nDYMUSDT SHORT:")
print("   Entry: $0.078944 → Current: $0.078361")
print("   Price move: -0.74% (i din favør)")
print("   P&L: +36.12 BNFCR (+18.25%) [OK]")
print("   → Dette er KORREKT med 20x leverage!")

print("\nPORTALUSDT SHORT:")
print("   Entry: $0.018711 → Current: $0.018631")
print("   Price move: -0.43% (i din favør)")
print("   P&L: +17.19 BNFCR (+0.43% shown in earlier check) [OK]")

print("\nAPTUSDT LONG:")
print("   Entry: $2.87548 → Current: $2.87691")
print("   Price move: +0.05% (i din favør)")
print("   P&L: -19.88 BNFCR (-11.31%) ❌❌❌")
print("   → DETTE ER FEIL! Burde være ~+1% med 20x leverage")

print("\nSOLUSDT LONG:")
print("   Entry: $138.17 → Current: $138.239")
print("   Price move: +0.05% (i din favør)")
print("   P&L: -22.11 BNFCR (-15.60%) ❌❌❌")
print("   → DETTE ER FEIL! Burde være ~+1% med 20x leverage")

print("\n" + "=" * 80)
print("💡 KONKLUSJON:")
print("=" * 80)
print("""
SHORT posisjonene fungerer PERFEKT:
[OK] 0.74% pris-endring = 18% ROI (ca 20x multiplikator)
[OK] 0.43% pris-endring = små gains

LONG posisjonene viser UNORMAL tap:
❌ 0.05% pris-endring = -11% til -16% tap
❌ Dette er IKKE normal funding fee
❌ Noe er galt med disse LONG posisjonene!

ANBEFALING:
1. Sjekk Binance trade history for faktisk entry price
2. Sjekk funding fee history for disse posisjonene
3. Vurder å lukke LONG hvis de fortsetter å tape på positive bevegelser
""")

print("\n" + "=" * 80)
