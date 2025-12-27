#!/usr/bin/env python3
"""
Calculate 40x leverage potential - MAX for testnet stability
"""

# Configuration
balance = 10816  # $10,816 total balance
margin_pct = 0.80  # 80% of balance as margin
leverage = 40  # 40x leverage

# Calculate
margin = balance * margin_pct
notional = margin * leverage
tp_pct = 0.016  # 1.6% TP
sl_pct = 0.008  # 0.8% SL

profit_per_win = notional * tp_pct
loss_per_loss = notional * sl_pct

# Daily projection
trades_per_day = 75
win_rate = 0.60
wins = int(trades_per_day * win_rate)
losses = trades_per_day - wins

daily_wins = wins * profit_per_win
daily_losses = losses * loss_per_loss
daily_net = daily_wins - daily_losses

print("\n40x Leverage Test (MAX for testnet):")
print(f"Margin: ${margin:,.0f}")
print(f"Notional: ${notional:,.0f}")
print(f"Profit/win (1.6% TP): ${profit_per_win:,.0f}")
print(f"Loss/loss (0.8% SL): ${loss_per_loss:,.0f}")
print(f"\nDaily ({trades_per_day} trades @ {win_rate:.0%} WR):")
print(f"Wins: {wins} x ${profit_per_win:,.0f} = ${daily_wins:,.0f}")
print(f"Losses: {losses} x ${loss_per_loss:,.0f} = ${daily_losses:,.0f}")
print(f"NET: ${daily_net:,.0f}/day")
print(f"\nComparison:")
print(f"Original: $180/win")
print(f"10x: $1,384/win (7.7x improvement)")
print(f"40x: ${profit_per_win:,.0f}/win ({profit_per_win/180:.1f}x improvement)")
