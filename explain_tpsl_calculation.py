"""Explain TP/SL calculation - based on PRICE movement, not profit amount"""

print("\n" + "=" * 80)
print("📐 TP/SL CALCULATION EXPLAINED")
print("=" * 80)

print("\n[TARGET] KEY POINT: TP/SL er basert på PRIS-endring, IKKE profit beløp!")
print("=" * 80)

# Example with DYMUSDT SHORT
print("\n[CHART] Eksempel: DYMUSDT SHORT Position")
print("-" * 80)

entry_price = 0.078944
position_size = 50588.0
leverage = 20
margin = 197.87  # BNFCR

print(f"Entry Price:    ${entry_price:.6f}")
print(f"Position Size:  {position_size:,.0f} DYM")
print(f"Leverage:       {leverage}x")
print(f"Margin:         {margin:.2f} BNFCR")

notional = position_size * entry_price
print(f"Notional Value: ${notional:.2f}")

print("\n" + "=" * 80)
print("🔢 BEREGNING AV TP OG SL PRISER:")
print("=" * 80)

# For SHORT: profit når prisen GÅR NED
tp_pct = 0.03  # 3%
sl_pct = 0.02  # 2%

print(f"\n1️⃣ TAKE PROFIT (+{tp_pct*100:.0f}% PROFIT):")
print(f"   For SHORT: Prisen må gå NED {tp_pct*100:.0f}%")
print(f"   ")
print(f"   TP Price = Entry × (1 - {tp_pct})")
tp_price = entry_price * (1 - tp_pct)
print(f"   TP Price = ${entry_price:.6f} × {1-tp_pct}")
print(f"   TP Price = ${tp_price:.6f}")
print(f"   ")
print(f"   [OK] Når prisen når ${tp_price:.6f}:")
print(f"      → Prisen har beveget seg {tp_pct*100:.0f}% NED")
print(f"      → Dette gir {tp_pct*100:.0f}% PROFIT på posisjonen")

print(f"\n2️⃣ STOP LOSS (-{sl_pct*100:.0f}% LOSS):")
print(f"   For SHORT: Prisen må gå OPP {sl_pct*100:.0f}%")
print(f"   ")
print(f"   SL Price = Entry × (1 + {sl_pct})")
sl_price = entry_price * (1 + sl_pct)
print(f"   SL Price = ${entry_price:.6f} × {1+sl_pct}")
print(f"   SL Price = ${sl_price:.6f}")
print(f"   ")
print(f"   ❌ Når prisen når ${sl_price:.6f}:")
print(f"      → Prisen har beveget seg {sl_pct*100:.0f}% OPP")
print(f"      → Dette gir {sl_pct*100:.0f}% TAP på posisjonen")

print("\n" + "=" * 80)
print("[MONEY] HVORDAN DETTE PÅVIRKER PROFIT MED LEVERAGE:")
print("=" * 80)

print(f"\nMed {leverage}x leverage:")
print(f"   Margin: {margin:.2f} BNFCR")
print(f"   Notional: ${notional:.2f} (Margin × {leverage})")

print(f"\n[CHART_UP] Ved TP (+{tp_pct*100:.0f}% pris-endring):")
price_change_tp = notional * tp_pct
profit_on_margin_tp = (price_change_tp / margin) * 100
print(f"   Price Change Value: ${price_change_tp:.2f}")
print(f"   Profit on Margin:   {profit_on_margin_tp:.0f}% (${price_change_tp:.2f} på {margin:.2f})")
print(f"   ")
print(f"   [TARGET] Med {leverage}x leverage:")
print(f"      → {tp_pct*100:.0f}% pris-bevegelse = ~{leverage * tp_pct * 100:.0f}% ROI på margin")

print(f"\n📉 Ved SL (-{sl_pct*100:.0f}% pris-endring):")
price_change_sl = notional * sl_pct
loss_on_margin_sl = (price_change_sl / margin) * 100
print(f"   Price Change Value: ${price_change_sl:.2f}")
print(f"   Loss on Margin:     {loss_on_margin_sl:.0f}% (${price_change_sl:.2f} på {margin:.2f})")
print(f"   ")
print(f"   🛑 Med {leverage}x leverage:")
print(f"      → {sl_pct*100:.0f}% pris-bevegelse = ~{leverage * sl_pct * 100:.0f}% tap på margin")

print("\n" + "=" * 80)
print("[CHART] SAMMENLIGNING:")
print("=" * 80)

print(f"\n{'Metric':<30} {'TP (+3%)':<20} {'SL (-2%)':<20}")
print("-" * 70)
print(f"{'Pris-endring:':<30} {f'{tp_pct*100:.1f}%':<20} {f'{sl_pct*100:.1f}%':<20}")
print(f"{'Dollar verdi endring:':<30} {f'${price_change_tp:.2f}':<20} {f'${price_change_sl:.2f}':<20}")
print(f"{'ROI på margin (20x):':<30} {f'+{profit_on_margin_tp:.0f}%':<20} {f'-{loss_on_margin_sl:.0f}%':<20}")
print(f"{'Profit/Loss beløp:':<30} {f'+{price_change_tp:.2f} BNFCR':<20} {f'-{price_change_sl:.2f} BNFCR':<20}")

print("\n" + "=" * 80)
print("💡 KONKLUSJON:")
print("=" * 80)
print(f"""
1. TP/SL er basert på PRIS-ENDRING (ikke profit beløp)
   → 3% TP = Prisen beveger seg 3% i din favør
   → 2% SL = Prisen beveger seg 2% mot deg

2. Med {leverage}x leverage blir effekten multiplisert:
   → 3% pris-endring ≈ {leverage * tp_pct * 100:.0f}% ROI på margin
   → 2% pris-endring ≈ {leverage * sl_pct * 100:.0f}% tap på margin

3. Dette gir god Risk/Reward ratio:
   → Risikerer {sl_pct*100:.0f}% pris-bevegelse ({leverage * sl_pct * 100:.0f}% på margin)
   → For å vinne {tp_pct*100:.0f}% pris-bevegelse ({leverage * tp_pct * 100:.0f}% på margin)
   → Ratio: 3:2 (1.5:1 reward:risk)
""")

print("=" * 80)
