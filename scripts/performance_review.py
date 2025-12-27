"""
Weekly Performance Review for TFT Model v1.1
Compares actual vs predicted returns, calculates metrics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List
import numpy as np


def get_closed_positions(db_path: str = "data/execution_journal.db", days: int = 7) -> List[Dict]:
    """Get closed positions from last N days"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cutoff = datetime.now() - timedelta(days=days)
    
    query = """
    SELECT * FROM execution_journal 
    WHERE timestamp > ? 
    AND agent = 'TFTAgent'
    AND action IN ('BUY', 'SELL')
    ORDER BY timestamp ASC
    """
    
    cursor.execute(query, (cutoff.isoformat(),))
    rows = cursor.fetchall()
    
    positions = []
    for row in rows:
        pos = dict(row)
        if pos.get('metadata'):
            try:
                pos['metadata'] = json.loads(pos['metadata'])
            except:
                pass
        positions.append(pos)
    
    conn.close()
    return positions


def calculate_performance_metrics(positions: List[Dict]) -> Dict:
    """Calculate win rate, avg return, Sharpe ratio, etc."""
    
    if not positions:
        return {}
    
    returns = []
    wins = 0
    losses = 0
    
    for pos in positions:
        # Extract actual return (if available in metadata)
        metadata = pos.get('metadata', {})
        if isinstance(metadata, dict):
            actual_return = metadata.get('actual_return')
            if actual_return is not None:
                returns.append(actual_return)
                if actual_return > 0:
                    wins += 1
                else:
                    losses += 1
    
    if not returns:
        return {
            'total_positions': len(positions),
            'positions_with_returns': 0,
            'note': 'No actual returns tracked yet'
        }
    
    returns_array = np.array(returns)
    
    metrics = {
        'total_positions': len(positions),
        'positions_with_returns': len(returns),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(returns) * 100 if returns else 0,
        'avg_return': np.mean(returns_array) * 100,
        'median_return': np.median(returns_array) * 100,
        'total_return': np.sum(returns_array) * 100,
        'best_trade': np.max(returns_array) * 100,
        'worst_trade': np.min(returns_array) * 100,
        'volatility': np.std(returns_array) * 100,
        'sharpe_ratio': (np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)) if np.std(returns_array) > 0 else 0,
    }
    
    return metrics


def check_quantile_calibration(positions: List[Dict]) -> Dict:
    """Check if actual returns fall within predicted quantile ranges"""
    
    within_p10_p90 = 0
    below_p10 = 0
    above_p90 = 0
    
    for pos in positions:
        metadata = pos.get('metadata', {})
        if not isinstance(metadata, dict):
            continue
        
        actual_return = metadata.get('actual_return')
        q10 = metadata.get('q10')
        q90 = metadata.get('q90')
        
        if actual_return is not None and q10 is not None and q90 is not None:
            if actual_return < q10:
                below_p10 += 1
            elif actual_return > q90:
                above_p90 += 1
            else:
                within_p10_p90 += 1
    
    total_checked = below_p10 + within_p10_p90 + above_p90
    
    return {
        'total_checked': total_checked,
        'below_p10': below_p10,
        'within_range': within_p10_p90,
        'above_p90': above_p90,
        'below_p10_pct': (below_p10 / total_checked * 100) if total_checked > 0 else 0,
        'above_p90_pct': (above_p90 / total_checked * 100) if total_checked > 0 else 0,
        'calibration_ok': (8 <= below_p10 / total_checked * 100 <= 12) if total_checked > 0 else False
    }


def main():
    """Weekly performance review"""
    
    print("\n" + "="*70)
    print("[CHART] TFT MODEL v1.1 - WEEKLY PERFORMANCE REVIEW")
    print("="*70 + "\n")
    
    print(f"Review Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Review Period: Last 7 days\n")
    
    db_path = Path("data/execution_journal.db")
    
    if not db_path.exists():
        print(f"❌ Execution journal not found: {db_path}")
        return
    
    # Get closed positions
    print("[SEARCH] Fetching closed positions from last 7 days...")
    positions = get_closed_positions(str(db_path), days=7)
    
    if not positions:
        print("[WARNING]  No TFT positions found in last 7 days")
        print("💡 Model may be new or in staging mode")
        return
    
    print(f"[OK] Found {len(positions)} positions\n")
    
    # Performance metrics
    print("="*70)
    print("[CHART_UP] PERFORMANCE METRICS")
    print("="*70 + "\n")
    
    metrics = calculate_performance_metrics(positions)
    
    if 'note' in metrics:
        print(f"[WARNING]  {metrics['note']}")
        print(f"   Total positions: {metrics['total_positions']}")
        print("\n💡 Return tracking needs to be implemented in backend")
    else:
        print(f"[CHART] Trading Statistics:")
        print(f"   Total positions: {metrics['total_positions']}")
        print(f"   Tracked returns: {metrics['positions_with_returns']}")
        print(f"   Wins: {metrics['wins']}")
        print(f"   Losses: {metrics['losses']}")
        print(f"   Win rate: {metrics['win_rate']:.1f}%")
        
        print(f"\n[MONEY] Returns:")
        print(f"   Average: {metrics['avg_return']:+.2f}%")
        print(f"   Median: {metrics['median_return']:+.2f}%")
        print(f"   Total: {metrics['total_return']:+.2f}%")
        print(f"   Best trade: {metrics['best_trade']:+.2f}%")
        print(f"   Worst trade: {metrics['worst_trade']:+.2f}%")
        
        print(f"\n[CHART] Risk Metrics:")
        print(f"   Volatility: {metrics['volatility']:.2f}%")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Quantile calibration
    print(f"\n" + "="*70)
    print("[TARGET] QUANTILE CALIBRATION CHECK")
    print("="*70 + "\n")
    
    calibration = check_quantile_calibration(positions)
    
    if calibration['total_checked'] == 0:
        print("[WARNING]  No quantile predictions with actual returns yet")
        print("💡 Need to track actual returns for calibration check")
    else:
        print(f"Positions checked: {calibration['total_checked']}")
        print(f"Below P10: {calibration['below_p10']} ({calibration['below_p10_pct']:.1f}%)")
        print(f"Within P10-P90: {calibration['within_range']}")
        print(f"Above P90: {calibration['above_p90']} ({calibration['above_p90_pct']:.1f}%)")
        
        print(f"\n[CHART] Calibration Status:")
        if calibration['calibration_ok']:
            print("   [OK] GOOD: P10 coverage within target range (8-12%)")
        else:
            print(f"   [WARNING]  NEEDS ATTENTION: P10 coverage at {calibration['below_p10_pct']:.1f}% (target: 10%)")
    
    # Recommendations
    print(f"\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70 + "\n")
    
    if 'win_rate' in metrics:
        win_rate = metrics['win_rate']
        sharpe = metrics['sharpe_ratio']
        
        print("[TARGET] Model Performance:")
        if win_rate >= 50 and sharpe >= 1.0:
            print("   [OK] EXCELLENT: Meeting success criteria")
            print(f"      Win rate: {win_rate:.1f}% (≥50% ✓)")
            print(f"      Sharpe: {sharpe:.2f} (≥1.0 ✓)")
        elif win_rate >= 50:
            print("   [WARNING]  GOOD: Win rate acceptable but Sharpe could improve")
            print(f"      Win rate: {win_rate:.1f}% (≥50% ✓)")
            print(f"      Sharpe: {sharpe:.2f} (target: ≥1.0)")
        elif win_rate >= 40:
            print("   [WARNING]  MARGINAL: Performance below target")
            print(f"      Win rate: {win_rate:.1f}% (target: ≥50%)")
            print(f"      Sharpe: {sharpe:.2f} (target: ≥1.0)")
        else:
            print("   ❌ POOR: Performance requires attention")
            print(f"      Win rate: {win_rate:.1f}% (target: ≥50%)")
            print(f"      Sharpe: {sharpe:.2f} (target: ≥1.0)")
            print("\n   🔧 Action Items:")
            print("      1. Review model predictions vs outcomes")
            print("      2. Consider retraining with more data")
            print("      3. Increase quantile_weight to 0.7-0.8")
    
    if 'calibration_ok' in calibration and not calibration['calibration_ok']:
        print("\n[TARGET] Quantile Calibration:")
        print("   [WARNING]  Predictions not well-calibrated")
        print("   🔧 Action Items:")
        print("      1. Increase quantile_weight from 0.5 to 0.7")
        print("      2. Retrain for more epochs (50 instead of 21)")
        print("      3. Use higher quantile loss coefficient")
    
    print("\n📅 Next Actions:")
    print("   • Monitor for another week if performance acceptable")
    print("   • Retrain if win rate < 40% or Sharpe < 0.5")
    print("   • Adjust quantile_weight if calibration poor")
    print("   • Update review date: 2025-11-26 → 2025-12-03")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
