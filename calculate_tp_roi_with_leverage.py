"""Calculate PnL/ROI with actual leverage from Binance API."""
import os
import sys
from binance.client import Client

# Binance client
client = Client(
    os.getenv('BINANCE_API_KEY'),
    os.getenv('BINANCE_API_SECRET'),
    testnet=True
)

# Hent Exit Brain state for TP levels
def get_exit_brain_tps():
    """Hent TP levels fra Exit Brain state."""
    try:
        from backend.domains.exits.exit_brain_v3.brain import ExitBrainV3
        brain = ExitBrainV3()
        state = brain._state
        
        tp_data = {}
        for symbol, pos_state in state.positions.items():
            if pos_state.tp_legs:
                tp_data[symbol] = {
                    'tp0': pos_state.tp_legs[0].price if len(pos_state.tp_legs) > 0 else None,
                    'tp1': pos_state.tp_legs[1].price if len(pos_state.tp_legs) > 1 else None,
                    'tp2': pos_state.tp_legs[2].price if len(pos_state.tp_legs) > 2 else None,
                }
        return tp_data
    except Exception as e:
        print(f"⚠️  Could not load Exit Brain state: {e}")
        return {}

# Hent alle åpne posisjoner fra Binance
print("🔄 Henter posisjoner fra Binance...")
binance_positions = client.futures_position_information()

# Hent TP levels fra Exit Brain
tp_data = get_exit_brain_tps()

# Fallback TP levels hvis Exit Brain state ikke er tilgjengelig
if not tp_data:
    print("💡 Bruker fallback TP levels fra logs...")
    tp_data = {
        'SOLUSDT': {'tp0': 137.53, 'tp1': 138.59, 'tp2': 140.18},
        'ETHUSDT': {'tp0': 3277.68, 'tp1': 3302.95, 'tp2': 3340.85},
        'XRPUSDT': {'tp0': 2.0549, 'tp1': 2.0707, 'tp2': 2.0945},
        'BTCUSDT': {'tp0': 93320.12, 'tp1': 94039.60, 'tp2': 95118.82}
    }

# Hent account info for å se isolated margin per symbol
account_info = client.futures_account()

# Filtrér kun åpne posisjoner
positions = {}

for pos in binance_positions:
    symbol = pos['symbol']
    position_amt = float(pos['positionAmt'])
    
    # Skip hvis ingen posisjon
    if position_amt == 0:
        continue
    
    # Skip SHORT posisjoner
    if position_amt < 0:
        continue
    
    entry_price = float(pos['entryPrice'])
    mark_price = float(pos['markPrice'])
    unrealized_pnl = float(pos['unRealizedProfit'])
    notional = float(pos['notional'])  # Position value i USDT
    
    # Beregn leverage: notional / margin
    # Finn margin fra account_info
    margin = None
    for asset_pos in account_info['positions']:
        if asset_pos['symbol'] == symbol:
            margin = abs(float(asset_pos['initialMargin']))
            break
    
    if margin and margin > 0:
        leverage = int(abs(notional) / margin)
    else:
        # Fallback: beregn leverage basert på isolatedWallet
        isolated_wallet = float(pos.get('isolatedWallet', 0))
        if isolated_wallet > 0:
            leverage = int(abs(notional) / isolated_wallet)
        else:
            leverage = 1
    
    # Beregn position value og margin
    position_value = abs(position_amt) * mark_price
    margin = position_value / leverage
    
    # Beregn ROI
    price_move_pct = ((mark_price - entry_price) / entry_price) * 100
    roi = price_move_pct * leverage
    
    positions[symbol] = {
        'leverage': leverage,
        'size': abs(position_amt),
        'entry_price': entry_price,
        'current_price': mark_price,
        'position_value': position_value,
        'margin': margin,
        'unrealized_pnl': unrealized_pnl,
        'roi': roi,
        'price_move_pct': price_move_pct
    }
    
    # Legg til TP levels hvis tilgjengelig
    if symbol in tp_data:
        positions[symbol].update(tp_data[symbol])

if not positions:
    print("\n❌ Ingen åpne LONG posisjoner funnet!")
    sys.exit(0)

print("\n" + "="*80)
print(f"EXIT BRAIN V3 - TP PROFIT CALCULATOR (DYNAMISK)")
print(f"Fant {len(positions)} åpne LONG posisjoner")
print("="*80)

total_capital = 0
total_unrealized = 0
total_tp0_profit = 0
total_tp1_profit = 0
total_tp2_profit = 0

for symbol, pos in positions.items():
    leverage = pos['leverage']
    entry_price = pos['entry_price']
    current_price = pos['current_price']
    size = pos['size']
    margin = pos['margin']
    unrealized_pnl = pos['unrealized_pnl']
    roi = pos['roi']
    price_move_pct = pos['price_move_pct']
    
    print(f"\n{'─'*80}")
    print(f"💰 {symbol} LONG - {leverage}x Leverage")
    print(f"{'─'*80}")
    print(f"Position: {size} @ ${entry_price:.4f}")
    print(f"Current:  ${current_price:.4f} ({price_move_pct:+.2f}% price move)")
    print(f"Margin:   ${margin:.2f} USDT")
    print(f"\n📊 Current: ${unrealized_pnl:+.2f} USDT ({roi:+.2f}% ROI)")
    
    total_capital += margin
    total_unrealized += unrealized_pnl
    
    # Check om vi har TP data
    if 'tp0' not in pos or pos['tp0'] is None:
        print(f"\n⚠️  Ingen TP levels funnet for {symbol}")
        continue
    
    print(f"\n{'─'*80}")
    print("TP LEVELS - Projected ROI:")
    print(f"{'─'*80}")
    
    # TP0 (40% exit)
    tp0_price_pnl = ((pos['tp0'] - entry_price) / entry_price) * 100
    tp0_roi = tp0_price_pnl * leverage
    tp0_distance = ((pos['tp0'] - current_price) / current_price) * 100
    tp0_profit = (pos['tp0'] - entry_price) * size * 0.40  # 40% exit
    
    print(f"\n  TP0: ${pos['tp0']:.4f} (40% exit)")
    print(f"    Price move: +{tp0_price_pnl:.2f}%")
    print(f"    💵 ROI:      +{tp0_roi:.2f}%")
    print(f"    💰 Profit:   +${tp0_profit:.2f} USDT (40% position)")
    print(f"    ⏳ Distance: {tp0_distance:+.2f}% from current")
    
    total_tp0_profit += tp0_profit
    
    # TP1 (35% exit) - hvis tilgjengelig
    if 'tp1' in pos and pos['tp1']:
        tp1_price_pnl = ((pos['tp1'] - entry_price) / entry_price) * 100
        tp1_roi = tp1_price_pnl * leverage
        tp1_distance = ((pos['tp1'] - current_price) / current_price) * 100
        tp1_profit = (pos['tp1'] - entry_price) * size * 0.35  # 35% exit
        
        print(f"\n  TP1: ${pos['tp1']:.4f} (35% exit)")
        print(f"    Price move: +{tp1_price_pnl:.2f}%")
        print(f"    💵 ROI:      +{tp1_roi:.2f}%")
        print(f"    💰 Profit:   +${tp1_profit:.2f} USDT (35% position)")
        print(f"    ⏳ Distance: {tp1_distance:+.2f}% from current")
        
        total_tp1_profit += tp1_profit
    
    # TP2 (25% exit) - hvis tilgjengelig
    if 'tp2' in pos and pos['tp2']:
        tp2_price_pnl = ((pos['tp2'] - entry_price) / entry_price) * 100
        tp2_roi = tp2_price_pnl * leverage
        tp2_distance = ((pos['tp2'] - current_price) / current_price) * 100
        tp2_profit = (pos['tp2'] - entry_price) * size * 0.25  # 25% exit
        
        print(f"\n  TP2: ${pos['tp2']:.4f} (25% exit)")
        print(f"    Price move: +{tp2_price_pnl:.2f}%")
        print(f"    💵 ROI:      +{tp2_roi:.2f}%")
        print(f"    💰 Profit:   +${tp2_profit:.2f} USDT (25% position)")
        print(f"    ⏳ Distance: {tp2_distance:+.2f}% from current")
        
        total_tp2_profit += tp2_profit
    
    # Total profit for denne posisjonen
    if 'tp0' in pos and pos['tp0']:
        total_profit = tp0_profit
        if 'tp1' in pos and pos['tp1']:
            total_profit += tp1_profit
        if 'tp2' in pos and pos['tp2']:
            total_profit += tp2_profit
        
        print(f"\n  📊 Total TP Profit: ${total_profit:.2f} USDT")

# Portfolio summary
print(f"\n{'='*80}")
print("📊 PORTFOLIO SUMMARY")
print(f"{'='*80}")
print(f"Total Margin:          ${total_capital:.2f} USDT")
print(f"Current Unrealized:    ${total_unrealized:.2f} USDT ({(total_unrealized/total_capital)*100:+.2f}% ROI)")

if total_tp0_profit > 0:
    print(f"\nProjected TP Profits:")
    print(f"  TP0 (all hit):       +${total_tp0_profit:.2f} USDT")
    if total_tp1_profit > 0:
        print(f"  TP1 (all hit):       +${total_tp1_profit:.2f} USDT")
    if total_tp2_profit > 0:
        print(f"  TP2 (all hit):       +${total_tp2_profit:.2f} USDT")
    
    total_all_tps = total_tp0_profit + total_tp1_profit + total_tp2_profit
    print(f"  Total (all TPs):     +${total_all_tps:.2f} USDT")
    print(f"  Portfolio ROI:       +{(total_all_tps/total_capital)*100:.2f}%")

print(f"{'='*80}\n")
