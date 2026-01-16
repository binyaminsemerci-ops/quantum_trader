#!/usr/bin/env python3
"""
Generate Trade Report from Logs - Last 24 Hours
Parses backend logs to extract trade information
"""

import re
from datetime import datetime, timedelta
import subprocess
import json

def parse_trade_logs():
    print("=" * 80)
    print("🎯 QUANTUM TRADER - TRADE RAPPORT (Siste 24 timer)")
    print("=" * 80)
    print(f"Periode: {(datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%d %H:%M')} til nå")
    print("=" * 80)
    print()
    
    # Get logs from Docker
    result = subprocess.run(
        ["journalctl", "-u", "quantum-backend.service", "--since", "24h"],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='ignore'
    )
    
    logs = (result.stdout or "") + (result.stderr or "")
    
    # Parse opened trades
    opened_pattern = r'Trade OPENED: (\d+)\s+(\w+) (LONG|SHORT)\s+Entry: \$([0-9.]+)\s+Quantity: ([0-9.]+)\s+SL: \$([0-9.]+)\s+TP: \$([0-9.]+)'
    
    opened_trades = []
    for match in re.finditer(opened_pattern, logs):
        trade_id, symbol, side, entry, qty, sl, tp = match.groups()
        opened_trades.append({
            'id': trade_id,
            'symbol': symbol,
            'side': side,
            'entry': float(entry),
            'qty': float(qty),
            'sl': float(sl),
            'tp': float(tp)
        })
    
    # Parse JSON formatted trade logs
    json_pattern = r'"message": "\[ROCKET\]\s+Trade OPENED: (\d+).*?(\w+USDT) (LONG|SHORT).*?Entry: \$([0-9.]+).*?Quantity: ([0-9.]+).*?SL: \$([0-9.]+).*?TP: \$([0-9.]+)'
    
    for match in re.finditer(json_pattern, logs):
        trade_id, symbol, side, entry, qty, sl, tp = match.groups()
        # Avoid duplicates
        if not any(t['id'] == trade_id for t in opened_trades):
            opened_trades.append({
                'id': trade_id,
                'symbol': symbol,
                'side': side,
                'entry': float(entry),
                'qty': float(qty),
                'sl': float(sl),
                'tp': float(tp)
            })
    
    if not opened_trades:
        print("❌ Ingen trades funnet i loggene for de siste 24 timene.")
        return
    
    print(f"📊 Totalt {len(opened_trades)} trades åpnet:\n")
    
    total_risk = 0.0
    total_potential_profit = 0.0
    
    for i, trade in enumerate(opened_trades, 1):
        print(f"🔹 Trade #{trade['id']}")
        print(f"   Symbol: {trade['symbol']} {trade['side']}")
        print(f"   Entry: ${trade['entry']:.4f}")
        print(f"   Quantity: {trade['qty']:.4f}")
        print(f"   Stop Loss: ${trade['sl']:.4f}")
        print(f"   Take Profit: ${trade['tp']:.4f}")
        
        # Calculate potential PnL
        if trade['side'] == 'LONG':
            sl_loss = (trade['sl'] - trade['entry']) * trade['qty']
            tp_profit = (trade['tp'] - trade['entry']) * trade['qty']
            sl_pct = ((trade['sl'] - trade['entry']) / trade['entry']) * 100
            tp_pct = ((trade['tp'] - trade['entry']) / trade['entry']) * 100
        else:  # SHORT
            sl_loss = (trade['entry'] - trade['sl']) * trade['qty']
            tp_profit = (trade['entry'] - trade['tp']) * trade['qty']
            sl_pct = ((trade['entry'] - trade['sl']) / trade['entry']) * 100
            tp_pct = ((trade['entry'] - trade['tp']) / trade['entry']) * 100
        
        risk_reward = abs(tp_profit / sl_loss) if sl_loss != 0 else 0
        
        print(f"   💸 Risiko ved SL: ${sl_loss:.2f} ({sl_pct:+.2f}%)")
        print(f"   💰 Potensial ved TP: ${tp_profit:.2f} ({tp_pct:+.2f}%)")
        print(f"   📊 Risk:Reward Ratio: 1:{risk_reward:.2f}")
        print()
        
        total_risk += abs(sl_loss)
        total_potential_profit += tp_profit
    
    print("=" * 80)
    print("📈 OPPSUMMERING:")
    print("=" * 80)
    print(f"✅ Antall trades åpnet: {len(opened_trades)}")
    print(f"💸 Total risiko (alle SL): ${total_risk:.2f}")
    print(f"💰 Total potensial profit (alle TP): ${total_potential_profit:.2f}")
    
    if total_risk > 0:
        overall_rr = total_potential_profit / total_risk
        print(f"📊 Samlet Risk:Reward: 1:{overall_rr:.2f}")
    
    # Count by direction
    long_count = sum(1 for t in opened_trades if t['side'] == 'LONG')
    short_count = sum(1 for t in opened_trades if t['side'] == 'SHORT')
    print(f"\n📊 Retning:")
    print(f"   📈 LONG: {long_count} trades ({long_count/len(opened_trades)*100:.1f}%)")
    print(f"   📉 SHORT: {short_count} trades ({short_count/len(opened_trades)*100:.1f}%)")
    
    print("\nℹ️  Merk: Dette er åpne trades. PnL vil bli realisert når trades lukkes.")
    print("=" * 80)

if __name__ == "__main__":
    parse_trade_logs()

