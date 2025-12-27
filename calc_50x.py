margin = 8652
leverage = 50
notional = margin * leverage
profit = notional * 0.016
loss = notional * 0.008

print(f"50x Leverage Test:")
print(f"Margin: ${margin:,}")
print(f"Notional: ${notional:,}")
print(f"Profit/win (1.6% TP): ${profit:,.0f}")
print(f"Loss/loss (0.8% SL): ${loss:,.0f}")
print()
print("Daily (75 trades @ 60% WR):")
wins = 45
losses = 30
daily = wins * profit - losses * loss
print(f"Wins: {wins} x ${profit:,.0f} = ${wins*profit:,.0f}")
print(f"Losses: {losses} x ${loss:,.0f} = ${losses*loss:,.0f}")
print(f"NET: ${daily:,.0f}/day")
