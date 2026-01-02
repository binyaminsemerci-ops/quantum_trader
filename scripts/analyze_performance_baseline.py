#!/usr/bin/env python3
"""
Performance Baseline Analyzer
Extracts performance metrics from trade database for P2 optimization
Uses only Python stdlib (no pandas/numpy dependencies)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import math

# Database paths
DB_PATHS = [
    "/home/qt/quantum_trader/data/trades.db",
    "/home/qt/quantum_trader/data/quantum_trader.db"
]

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if len(returns) == 0:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
    std_dev = math.sqrt(variance)
    
    if std_dev == 0:
        return 0.0
    
    excess_returns = mean_return - risk_free_rate
    return (excess_returns / std_dev) * math.sqrt(252)  # Annualized

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate Sortino ratio (downside deviation only)"""
    if len(returns) == 0:
        return 0.0
    
    mean_return = sum(returns) / len(returns)
    downside_returns = [r for r in returns if r < 0]
    
    if len(downside_returns) == 0:
        return 0.0
    
    downside_mean = sum(downside_returns) / len(downside_returns)
    downside_variance = sum((r - downside_mean) ** 2 for r in downside_returns) / len(downside_returns)
    downside_std = math.sqrt(downside_variance)
    
    if downside_std == 0:
        return 0.0
    
    excess_returns = mean_return - risk_free_rate
    return (excess_returns / downside_std) * math.sqrt(252)

def calculate_max_drawdown(cumulative_returns):
    """Calculate maximum drawdown"""
    if len(cumulative_returns) == 0:
        return 0.0
    
    max_dd = 0.0
    running_max = cumulative_returns[0]
    
    for value in cumulative_returns:
        running_max = max(running_max, value)
        drawdown = (value - running_max) / running_max if running_max != 0 else 0
        max_dd = min(max_dd, drawdown)
    
    return max_dd

def analyze_trades():
    """Analyze trade data and generate baseline report"""
    
    # Try to find and load trade data
    trades = []
    db_source = None
    
    for db_path in DB_PATHS:
        if not Path(db_path).exists():
            continue
            
        print(f"Found database: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Try different table names
            for table in ['trades', 'trade_history', 'executed_trades', 'positions']:
                try:
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if cursor.fetchone():
                        cursor.execute(f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 1000")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        trades = [dict(zip(columns, row)) for row in rows]
                        
                        if len(trades) > 0:
                            print(f"Loaded {len(trades)} trades from table '{table}'")
                            db_source = db_path
                            break
                except Exception as e:
                    print(f"Table '{table}' error: {e}")
                    continue
            
            conn.close()
            
            if len(trades) > 0:
                break
        except Exception as e:
            print(f"Error loading {db_path}: {e}")
    
    if len(trades) == 0:
        return generate_empty_baseline()
    
    # Extract PnL values (try different column names)
    pnl_values = []
    for trade in trades:
        pnl = trade.get('pnl') or trade.get('realized_pnl') or trade.get('profit') or 0
        try:
            pnl_values.append(float(pnl))
        except:
            pnl_values.append(0.0)
    
    # Calculate metrics
    total_pnl = sum(pnl_values)
    num_trades = len(trades)
    
    # Winrate
    winning_trades = [p for p in pnl_values if p > 0]
    losing_trades = [p for p in pnl_values if p < 0]
    winrate = len(winning_trades) / num_trades if num_trades > 0 else 0
    
    # Profit Factor
    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
    
    # Average R per trade
    avg_pnl_per_trade = total_pnl / num_trades if num_trades > 0 else 0
    
    # Sharpe/Sortino
    sharpe = calculate_sharpe_ratio(pnl_values)
    sortino = calculate_sortino_ratio(pnl_values)
    
    # Max Drawdown
    cumulative_pnl = []
    running_sum = 0
    for pnl in pnl_values:
        running_sum += pnl
        cumulative_pnl.append(running_sum)
    max_dd = calculate_max_drawdown(cumulative_pnl)
    
    # PnL per symbol
    pnl_by_symbol = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'values': []})
    for trade in trades:
        symbol = trade.get('symbol', 'UNKNOWN')
        pnl = float(trade.get('pnl') or trade.get('realized_pnl') or 0)
        pnl_by_symbol[symbol]['sum'] += pnl
        pnl_by_symbol[symbol]['count'] += 1
        pnl_by_symbol[symbol]['values'].append(pnl)
    
    # Calculate mean for each symbol
    for symbol, data in pnl_by_symbol.items():
        data['mean'] = data['sum'] / data['count'] if data['count'] > 0 else 0
    
    # PnL per confidence bucket
    pnl_by_confidence = defaultdict(lambda: {'sum': 0.0, 'count': 0, 'values': []})
    for trade in trades:
        conf = trade.get('confidence')
        if conf is not None:
            try:
                conf_val = float(conf)
                if conf_val < 0.6:
                    bucket = '0.50-0.60'
                elif conf_val < 0.7:
                    bucket = '0.60-0.70'
                elif conf_val < 0.8:
                    bucket = '0.70-0.80'
                else:
                    bucket = '0.80+'
                
                pnl = float(trade.get('pnl') or trade.get('realized_pnl') or 0)
                pnl_by_confidence[bucket]['sum'] += pnl
                pnl_by_confidence[bucket]['count'] += 1
                pnl_by_confidence[bucket]['values'].append(pnl)
            except:
                pass
    
    # Calculate mean for each bucket
    for bucket, data in pnl_by_confidence.items():
        data['mean'] = data['sum'] / data['count'] if data['count'] > 0 else 0
    
    # Generate report
    report = f"""# PERFORMANCE BASELINE

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Data Source**: {db_source}  
**Period**: Last {num_trades} trades

---

## 📊 CORE METRICS

### Overall Performance
| Metric | Value |
|--------|-------|
| **Total PnL** | ${total_pnl:.2f} |
| **Total Trades** | {num_trades} |
| **Sharpe Ratio** | {sharpe:.2f} |
| **Sortino Ratio** | {sortino:.2f} |
| **Max Drawdown** | {max_dd:.2%} |
| **Win Rate** | {winrate:.2%} |
| **Profit Factor** | {profit_factor:.2f} |
| **Avg R/Trade** | ${avg_pnl_per_trade:.2f} |

### Risk Metrics
- **Winning Trades**: {len(winning_trades)} ({len(winning_trades)/num_trades*100:.1f}%)
- **Losing Trades**: {len(losing_trades)} ({len(losing_trades)/num_trades*100:.1f}%)
- **Gross Profit**: ${gross_profit:.2f}
- **Gross Loss**: ${gross_loss:.2f}

---

## 💰 PNL ATTRIBUTION

### By Symbol
"""
    
    if pnl_by_symbol:
        report += "\n| Symbol | Total PnL | Trades | Avg PnL/Trade |\n|--------|-----------|--------|---------------|\n"
        # Sort by total PnL descending
        sorted_symbols = sorted(pnl_by_symbol.items(), key=lambda x: x[1]['sum'], reverse=True)[:10]
        for symbol, data in sorted_symbols:
            report += f"| {symbol} | ${data['sum']:.2f} | {data['count']} | ${data['mean']:.2f} |\n"
    else:
        report += "\n*No symbol attribution data available*\n"
    
    report += "\n### By Confidence Bucket\n"
    if pnl_by_confidence:
        report += "\n| Confidence | Total PnL | Trades | Avg PnL/Trade | Expectancy |\n|------------|-----------|--------|---------------|------------|\n"
        # Sort by bucket order
        bucket_order = ['0.50-0.60', '0.60-0.70', '0.70-0.80', '0.80+']
        for bucket in bucket_order:
            if bucket in pnl_by_confidence:
                data = pnl_by_confidence[bucket]
                expectancy = data['sum'] / data['count'] if data['count'] > 0 else 0
                report += f"| {bucket} | ${data['sum']:.2f} | {data['count']} | ${data['mean']:.2f} | ${expectancy:.2f} |\n"
    else:
        report += "\n*No confidence attribution data available*\n"
    
    report += f"""

---

## 🎯 P2 OPTIMIZATION TARGETS

Based on baseline analysis:

### Priority 1: Sharpe Ratio Improvement
- **Current**: {sharpe:.2f}
- **Target**: {sharpe * 1.2:.2f} (+20%)
- **Action**: Confidence filtering analysis

### Priority 2: Drawdown Reduction
- **Current**: {max_dd:.2%}
- **Target**: {max_dd * 0.8:.2%} (-20%)
- **Action**: Exit optimization + position correlation

### Priority 3: Win Rate Enhancement
- **Current**: {winrate:.2%}
- **Target**: {min(winrate * 1.1, 0.65):.2%} (+10% or 65% max)
- **Action**: Regime-aware filtering

---

## 📝 NOTES

This baseline represents the **STARTING POINT** for P2 optimization.

**Do NOT optimize without this reference.**

All P2 changes will be measured against these metrics.

---

## ⏭️ NEXT STEPS

1. **Trade Attribution Layer**: Tag all trades with regime, strategy, exit reason
2. **Confidence Analysis**: Analyze PnL by confidence bucket (detailed)
3. **Regime Analysis**: Identify losing regimes
4. **Exit Analysis**: MFE vs MAE, trailing effectiveness
5. **Correlation Analysis**: Multi-position drawdown clustering

**Start with**: Step 1 (Attribution) or Step 2 (Confidence) depending on data availability.
"""
    
    return report

def generate_empty_baseline():
    """Generate baseline when no trade data available"""
    return f"""# PERFORMANCE BASELINE

**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Data Source**: No trade data found  
**Status**: ⚠️ WAITING FOR TRADES

---

## ⚠️ NO BASELINE DATA

No trade history found in:
- `/home/qt/quantum_trader/data/trades.db`
- `/home/qt/quantum_trader/data/quantum_trader.db`

**System Status**: Likely in preflight or shadow mode (no real trades executed yet).

---

## 🎯 WHEN TRADES ARE AVAILABLE

Run this script again after:
- Phase C (Live Small) has executed 1-3 trades
- Or after shadow mode with logged "WOULD_SUBMIT" entries

---

## 📝 PLACEHOLDER METRICS

Until real data is available, use these targets:

| Metric | Target |
|--------|--------|
| Sharpe Ratio | >1.5 |
| Max Drawdown | <20% |
| Win Rate | >55% |
| Profit Factor | >1.5 |

---

## ⏭️ NEXT STEPS

1. Complete Phase C (Live Small) to generate trade data
2. Re-run: `python3 scripts/analyze_performance_baseline.py`
3. Proceed with P2 optimization once baseline established
"""

if __name__ == "__main__":
    print("=" * 60)
    print("QUANTUM TRADER - PERFORMANCE BASELINE ANALYZER")
    print("=" * 60)
    print()
    
    report = analyze_trades()
    
    # Write report
    output_path = "PERFORMANCE_BASELINE.md"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Baseline report generated: {output_path}")
    print()
    print("Preview:")
    print("-" * 60)
    print(report[:800])
    print("...")
    print("-" * 60)
