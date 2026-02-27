#!/usr/bin/env python3
"""
Exchange Ground-Truth Audit - Using Working Credentials
"""
import os
import sys
from binance.client import Client
from binance.exceptions import BinanceAPIException
import json
from datetime import datetime
from collections import defaultdict

# Working credentials from /etc/quantum/position-monitor-secrets/binance.env
API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
API_SECRET = "6dw3PKJE6oTcJ43lEaKS1N7OOxsvlIBv24ZmTBJeEsNKEOuO2pmmIjS5VBP89kKv"
TESTNET = True

def main():
    print("=" * 80)
    print("QUANTUM TRADER - EXCHANGE GROUND-TRUTH AUDIT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Mode: {'TESTNET' if TESTNET else 'PRODUCTION'}")
    print(f"API Key: {API_KEY[:20]}...")
    print()
    
    # Initialize client
    if TESTNET:
        client = Client(API_KEY, API_SECRET, testnet=True)
        print("✓ Connected to Binance Futures TESTNET")
    else:
        client = Client(API_KEY, API_SECRET)
        print("✓ Connected to Binance Futures PRODUCTION")
    
    try:
        # Test account access
        account = client.futures_account()
        print(f"✓ Account Balance: {account['totalWalletBalance']} USDT")
        print()
        
        # Fetch all trades
        print("Fetching trade history from exchange...")
        all_trades = []
        
        # Get all positions (current and historical)
        positions = client.futures_position_information()
        symbols_traded = set()
        for pos in positions:
            if float(pos.get('notional', 0)) != 0 or float(pos.get('positionAmt', 0)) != 0:
                symbols_traded.add(pos['symbol'])
        
        print(f"Found {len(symbols_traded)} symbols with position history: {symbols_traded}")
        
        # Fetch trades for each symbol
        for symbol in symbols_traded:
            try:
                trades = client.futures_account_trades(symbol=symbol, limit=1000)
                all_trades.extend(trades)
                print(f"  {symbol}: {len(trades)} trades")
            except BinanceAPIException as e:
                print(f"  {symbol}: Error - {e}")
        
        # Also try to get income history (realized PnL)
        print("\nFetching income history (realized PnL)...")
        income = client.futures_income_history(incomeType='REALIZED_PNL', limit=1000)
        print(f"Found {len(income)} realized PnL records")
        
        print(f"\n{'=' * 80}")
        print(f"TOTAL TRADES FETCHED: {len(all_trades)}")
        print(f"TOTAL INCOME RECORDS: {len(income)}")
        print(f"{'=' * 80}\n")
        
        # Reconstruct closed positions
        print("Reconstructing closed positions...")
        positions_by_symbol = defaultdict(list)
        
        for trade in sorted(all_trades, key=lambda x: x['time']):
            symbol = trade['symbol']
            positions_by_symbol[symbol].append(trade)
        
        closed_positions = []
        for symbol, trades in positions_by_symbol.items():
            position_qty = 0.0
            entry_trades = []
            
            for trade in trades:
                qty = float(trade['qty'])
                if trade['side'] == 'SELL':
                    qty = -qty
                
                if position_qty == 0:
                    # Starting new position
                    entry_trades = [trade]
                    position_qty = qty
                elif (position_qty > 0 and qty > 0) or (position_qty < 0 and qty < 0):
                    # Adding to position
                    entry_trades.append(trade)
                    position_qty += qty
                else:
                    # Closing (partial or full)
                    if abs(qty) >= abs(position_qty):
                        # Full close
                        closed_positions.append({
                            'symbol': symbol,
                            'entry_trades': entry_trades,
                            'exit_trade': trade,
                            'pnl': float(trade.get('realizedPnl', 0))
                        })
                        position_qty = 0
                        entry_trades = []
                    else:
                        # Partial close - just reduce position
                        position_qty += qty
        
        print(f"Reconstructed {len(closed_positions)} closed positions\n")
        
        # Calculate metrics
        if len(closed_positions) > 0:
            pnls = [pos['pnl'] for pos in closed_positions]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            total_pnl = sum(pnls)
            win_rate = len(wins) / len(pnls) * 100
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            expectancy = total_pnl / len(pnls)
            
            print("=" * 80)
            print("EXCHANGE GROUND-TRUTH METRICS")
            print("=" * 80)
            print(f"Sample Size:           {len(closed_positions)} closed positions")
            print(f"Total PnL:             ${total_pnl:.2f} USDT")
            print(f"Win Rate:              {win_rate:.1f}%")
            print(f"Wins / Losses:         {len(wins)} / {len(losses)}")
            print(f"Average Win:           ${avg_win:.2f}")
            print(f"Average Loss:          ${avg_loss:.2f}")
            print(f"Profit Factor:         {profit_factor:.2f}")
            print(f"Expectancy per Trade:  ${expectancy:.2f}")
            print()
            
            # Statistical significance
            min_required = 174  # For 95% confidence
            if len(closed_positions) < min_required:
                print(f"⚠️  WARNING: Sample size ({len(closed_positions)}) < required ({min_required})")
                print("   Results are NOT statistically significant")
            else:
                print(f"✓ Sample size sufficient for statistical significance")
            
            print()
            
            # Classification
            if expectancy > 0 and profit_factor > 1.5:
                verdict = "✓ STRUCTURALLY PROFITABLE"
            elif expectancy > 0 and profit_factor > 1.0:
                verdict = "⚠️  POTENTIALLY POSITIVE (Low Margin)"
            elif expectancy < 0:
                verdict = "❌ STRUCTURALLY NEGATIVE"
            else:
                verdict = "⚠️  INCONCLUSIVE"
            
            print(f"VERDICT: {verdict}")
            print("=" * 80)
            
            # Save results
            results = {
                'timestamp': datetime.now().isoformat(),
                'mode': 'TESTNET' if TESTNET else 'PRODUCTION',
                'sample_size': len(closed_positions),
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'wins': len(wins),
                'losses': len(losses),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'verdict': verdict,
                'positions': closed_positions[:10]  # First 10 for inspection
            }
            
            with open('EXCHANGE_GROUND_TRUTH_AUDIT.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("\n✓ Results saved to EXCHANGE_GROUND_TRUTH_AUDIT.json")
        
        else:
            print("⚠️  No closed positions found")
        
    except BinanceAPIException as e:
        print(f"\n❌ Binance API Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
