balance = 10000  # $5000 USDT + $5000 USDC
risk_pct = 0.80  # 80%
margin = balance * risk_pct

print(f"💰 Total Balance: ${balance:,}")
print(f"📊 80% Capital: ${margin:,.0f} per trade")
print()

for lev in [5, 10, 15, 20, 25]:
    notional = margin * lev
    profit = notional * 0.016  # 1.6% TP
    loss = notional * 0.008    # 0.8% SL
    
    print(f"{lev}x Leverage:")
    print(f"  Notional: ${notional:,.0f}")
    print(f"  Profit/win: ${profit:,.0f}")
    print(f"  Loss/loss: -${loss:,.0f}")
    print()
