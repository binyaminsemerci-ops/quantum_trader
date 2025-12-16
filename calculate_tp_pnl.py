"""Calculate PnL/ROI for TP levels."""

# SOLUSDT Position
entry = 135.9394
current = 137.04
tp0 = 137.53
tp1 = 138.59
tp2 = 140.18

print("=" * 60)
print("SOLUSDT LONG Position - PnL/ROI Analysis")
print("=" * 60)
print(f"\nEntry Price:   ${entry:.4f}")
print(f"Current Price: ${current:.4f}")

# Current PnL
current_pnl = ((current - entry) / entry) * 100
print(f"\n📊 Current PnL: +{current_pnl:.2f}%")

print("\n" + "=" * 60)
print("TP Levels - PnL från Entry:")
print("=" * 60)

# TP0
tp0_pnl = ((tp0 - entry) / entry) * 100
tp0_from_current = ((tp0 - current) / current) * 100
print(f"\nTP0: ${tp0:.2f} (40% av position)")
print(f"  📈 PnL från entry: +{tp0_pnl:.2f}%")
print(f"  ⏳ Behöver gå opp: +{tp0_from_current:.2f}% från current")

# TP1
tp1_pnl = ((tp1 - entry) / entry) * 100
tp1_from_current = ((tp1 - current) / current) * 100
print(f"\nTP1: ${tp1:.2f} (35% av position)")
print(f"  📈 PnL från entry: +{tp1_pnl:.2f}%")
print(f"  ⏳ Behöver gå opp: +{tp1_from_current:.2f}% från current")

# TP2
tp2_pnl = ((tp2 - entry) / entry) * 100
tp2_from_current = ((tp2 - current) / current) * 100
print(f"\nTP2: ${tp2:.2f} (25% av position)")
print(f"  📈 PnL från entry: +{tp2_pnl:.2f}%")
print(f"  ⏳ Behöver gå opp: +{tp2_from_current:.2f}% från current")

# Weighted average TP
weighted_tp_pnl = (tp0_pnl * 0.40) + (tp1_pnl * 0.35) + (tp2_pnl * 0.25)
print("\n" + "=" * 60)
print(f"📊 Gjennomsnittlig TP PnL (weighted): +{weighted_tp_pnl:.2f}%")
print("=" * 60)
