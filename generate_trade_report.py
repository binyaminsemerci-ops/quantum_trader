#!/usr/bin/env python3
"""
Generate Trade Report - Last 24 Hours
Shows all trades with win/loss values
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

def generate_trade_report():
    # Connect to database
    db_path = "/app/backend/data/trades.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get trades from last 24 hours
        yesterday = datetime.now() - timedelta(hours=24)
        yesterday_str = yesterday.strftime("%Y-%m-%d %H:%M:%S")
        
        print("=" * 80)
        print("🎯 QUANTUM TRADER - TRADE RAPPORT (Siste 24 timer)")
        print("=" * 80)
        print(f"Periode: {yesterday.strftime('%Y-%m-%d %H:%M')} til nå")
        print("=" * 80)
        print()
        
        # Query for opened trades
        cursor.execute("""
            SELECT 
                id,
                symbol,
                side,
                entry_price,
                qty,
                exit_price,
                pnl,
                timestamp
            FROM trades
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (yesterday_str,))
        
        trades = cursor.fetchall()
        
        if not trades:
            print("❌ Ingen trades funnet i databasen for de siste 24 timene.")
            print("\nℹ️  Sjekker trade_logs for aktivitet...")
            
            # Try trade_logs table
            cursor.execute("""
                SELECT 
                    id,
                    symbol,
                    side,
                    qty,
                    price,
                    status,
                    reason,
                    timestamp
                FROM trade_logs
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (yesterday_str,))
            
            trade_logs = cursor.fetchall()
            
            if trade_logs:
                print(f"\n📊 Funnet {len(trade_logs)} trade aktiviteter i logger:\n")
                for log in trade_logs:
                    log_id, symbol, side, qty, price, status, reason, ts = log
                    print(f"🔹 Log #{log_id}")
                    print(f"   Symbol: {symbol} {side}")
                    print(f"   Qty: {qty} @ ${price:.4f}")
                    print(f"   Status: {status}")
                    print(f"   Reason: {reason}")
                    print(f"   Tid: {ts}")
                    print()
            else:
                print("❌ Ingen trade aktivitet funnet i logger heller.")
        else:
            print(f"📊 Totalt {len(trades)} trades funnet:\n")
            
            total_opened = 0
            total_closed = 0
            total_pnl = 0.0
            win_count = 0
            loss_count = 0
            
            for trade in trades:
                trade_id, symbol, side, entry, qty, exit_price, pnl_value, ts = trade
                
                print(f"🔹 Trade #{trade_id}")
                print(f"   Symbol: {symbol} {side}")
                print(f"   Entry: ${entry:.4f} | Qty: {qty}")
                print(f"   Tid: {ts}")
                
                # Check if trade is closed
                if exit_price and exit_price > 0:
                    total_closed += 1
                    print(f"   Exit: ${exit_price:.4f}")
                    
                    if pnl_value:
                        if pnl_value > 0:
                            pnl_emoji = "💚"
                            win_count += 1
                        else:
                            pnl_emoji = "❌"
                            loss_count += 1
                        print(f"   {pnl_emoji} Realized PnL: ${pnl_value:.2f}")
                        total_pnl += pnl_value
                    else:
                        # Calculate PnL manually
                        if side.upper() in ["BUY", "LONG"]:
                            calc_pnl = (exit_price - entry) * qty
                        else:  # SHORT/SELL
                            calc_pnl = (entry - exit_price) * qty
                        
                        if calc_pnl > 0:
                            pnl_emoji = "💚"
                            win_count += 1
                        else:
                            pnl_emoji = "❌"
                            loss_count += 1
                        print(f"   {pnl_emoji} Calculated PnL: ${calc_pnl:.2f}")
                        total_pnl += calc_pnl
                else:
                    total_opened += 1
                    print(f"   ⏳ Status: ÅPEN (fortsatt aktiv)")
                
                print()
            
            print("=" * 80)
            print("📈 OPPSUMMERING:")
            print("=" * 80)
            print(f"✅ Åpne trades: {total_opened}")
            print(f"🔒 Lukkede trades: {total_closed}")
            
            if total_closed > 0:
                win_rate = (win_count / total_closed) * 100
                pnl_emoji = "💚" if total_pnl > 0 else "❌"
                print(f"\n{pnl_emoji} Total Realized PnL: ${total_pnl:.2f}")
                print(f"   🎯 Vinner: {win_count} trades (💚)")
                print(f"   📉 Taper: {loss_count} trades (❌)")
                print(f"   📊 Win Rate: {win_rate:.1f}%")
                
                if total_pnl > 0:
                    print(f"\n   🎉 TOTAL PROFIT: ${total_pnl:.2f}")
                else:
                    print(f"\n   ⚠️  TOTAL LOSS: ${total_pnl:.2f}")
            else:
                print("\nℹ️  Ingen lukkede trades å rapportere på ennå.")
            print("=" * 80)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        print(f"\nℹ️  Prøver å hente fra logger i stedet...")
        return False
    
    return True

if __name__ == "__main__":
    generate_trade_report()
