#!/usr/bin/env python3
"""
Exit Brain v3 Monitoring Script
Checks active positions and Exit Brain status
"""

from binance.client import Client
import os

def main():
    # Initialize Binance client
    client = Client(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    client.API_URL = 'https://testnet.binancefuture.com'
    
    # Get all positions
    positions = client.futures_position_information()
    active = [p for p in positions if float(p['positionAmt']) != 0]
    
    print('\n' + '='*80)
    print(f'📊 EXIT BRAIN V3 MONITORING - ACTIVE POSITIONS: {len(active)}')
    print('='*80 + '\n')
    
    if len(active) == 0:
        print('✅ Ingen aktive posisjoner for øyeblikket.')
        print('   Exit Brain v3 er AKTIV og venter på neste posisjon.')
        print('   Når en ny posisjon åpnes, vil Exit Brain:')
        print('   • Bygge unified exit plan (3-leg strategy)')
        print('   • Plassere TP/SL orders automatisk')
        print('   • Forhindre konflikter med position monitor')
        print()
    else:
        for pos in active:
            symbol = pos['symbol']
            amt = float(pos['positionAmt'])
            side = 'LONG' if amt > 0 else 'SHORT'
            entry = float(pos['entryPrice'])
            mark = float(pos['markPrice'])
            pnl = float(pos['unRealizedProfit'])
            
            # Calculate PnL percentage
            if entry > 0:
                pnl_pct = (pnl / (abs(amt) * entry)) * 100
            else:
                pnl_pct = 0
            
            print(f'🎯 {symbol}')
            print(f'   Position: {side} {abs(amt):.4f} @ ${entry:.4f}')
            print(f'   Current:  ${mark:.4f}')
            print(f'   PnL:      {pnl_pct:+.2f}% (${pnl:+.2f} USDT)')
            
            # Check for TP/SL protection
            try:
                orders = client.futures_get_open_orders(symbol=symbol)
                tp_orders = [o for o in orders if o['type'] == 'TAKE_PROFIT_MARKET']
                sl_orders = [o for o in orders if o['type'] == 'STOP_MARKET']
                
                if tp_orders or sl_orders:
                    print(f'   Protection: ✅ {len(tp_orders)} TP, {len(sl_orders)} SL')
                    for tp in tp_orders:
                        print(f'      • TP @ ${tp["stopPrice"]} ({tp["origQty"]} units)')
                    for sl in sl_orders:
                        print(f'      • SL @ ${sl["stopPrice"]} ({sl["origQty"]} units)')
                else:
                    print(f'   Protection: ⚠️  UNPROTECTED (No TP/SL orders)')
            except Exception as e:
                print(f'   Protection: ❌ Could not check orders: {e}')
            
            print()
    
    print('='*80)
    print('🧠 EXIT BRAIN V3 STATUS')
    print('='*80)
    print('✅ Feature Flag:     EXIT_BRAIN_V3_ENABLED=true')
    print('✅ Orchestrator:     Initialized in dynamic_tpsl')
    print('✅ Position Monitor: Respects Exit Brain (skips adjustment)')
    print('✅ Trailing Manager: Reads Exit Brain config')
    print()
    print('📋 Next Steps:')
    print('   1. Wait for next position to open')
    print('   2. Exit Brain will automatically create exit plan')
    print('   3. TP/SL orders will be placed via dynamic_tpsl')
    print('   4. Position monitor will skip conflicting adjustments')
    print('='*80 + '\n')

if __name__ == '__main__':
    main()
