import asyncio
from backend.services.bybit_service import BybitService

async def check_sol_position():
    service = BybitService()
    positions = await service.get_positions()
    
    sol_pos = [p for p in positions if p['symbol'] == 'SOLUSDT']
    if sol_pos:
        pos = sol_pos[0]
        print('=' * 60)
        print('SOLUSDT POSITION ANALYSIS')
        print('=' * 60)
        print(f'Side: {pos["side"]}')
        print(f'Size: {pos["size"]} SOL')
        print(f'Entry Price: ${pos["avgPrice"]:.4f}')
        print(f'Mark Price: ${pos["markPrice"]:.4f}')
        print(f'Liquidation: ${pos["liqPrice"]:.4f}' if pos.get('liqPrice') else 'Liquidation: N/A')
        print(f'Leverage: {pos["leverage"]}x')
        print(f'Unrealized PnL: ${pos["unrealisedPnl"]:.2f}')
        print(f'PnL %: {float(pos["unrealisedPnl"]) / float(pos["positionValue"]) * 100:.2f}%')
        print(f'Position Value: ${pos["positionValue"]:.2f}')
        print('')
        print('RISK ASSESSMENT:')
        
        entry = float(pos['avgPrice'])
        mark = float(pos['markPrice'])
        liq = float(pos.get('liqPrice', 0))
        
        # Calculate distance to liquidation
        if liq > 0:
            dist_to_liq = ((liq - mark) / mark) * 100
            print(f'Distance to Liquidation: {dist_to_liq:.2f}%')
            
            if dist_to_liq < 5:
                print('⚠️ CRITICAL: Very close to liquidation!')
            elif dist_to_liq < 10:
                print('⚠️ WARNING: Approaching liquidation zone')
            else:
                print('✓ Liquidation risk: Moderate')
        
        # Calculate unrealized loss percentage
        pnl_pct = float(pos['unrealisedPnl']) / float(pos['positionValue']) * 100
        if pnl_pct < -15:
            print('⚠️ CRITICAL: Loss exceeds -15%')
        elif pnl_pct < -10:
            print('⚠️ WARNING: Loss exceeds -10%')
        
        print('')
        print('RECOMMENDATIONS:')
        if pnl_pct < -15 and liq > 0 and dist_to_liq < 10:
            print('1. Consider closing position to prevent liquidation')
            print('2. Or add margin to increase liquidation buffer')
        elif pnl_pct < -10:
            print('1. Monitor closely for stop-loss exit')
            print('2. Check if RL model recommends position adjustment')
        else:
            print('Position within acceptable risk parameters')
            
        # Check stop loss status
        print('')
        print('STOP LOSS CHECK:')
        if pos.get('stopLoss'):
            print(f'Stop Loss Set: ${float(pos["stopLoss"]):.4f}')
            sl_dist = ((float(pos['stopLoss']) - mark) / mark) * 100
            print(f'Distance to SL: {sl_dist:.2f}%')
        else:
            print('⚠️ No stop loss set!')
            
    else:
        print('No SOLUSDT position found')

asyncio.run(check_sol_position())
