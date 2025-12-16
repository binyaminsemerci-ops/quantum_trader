"""
Monitor TFT signals from execution_journal
Track R/R ratios, confidence levels, and prediction accuracy
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict


def get_recent_signals(db_path: str = "data/execution_journal.db", hours: int = 24) -> List[Dict]:
    """Get recent TFT signals from execution journal"""
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get signals from last N hours
    cutoff = datetime.now() - timedelta(hours=hours)
    
    query = """
    SELECT * FROM execution_journal 
    WHERE timestamp > ? 
    AND agent = 'TFTAgent'
    ORDER BY timestamp DESC
    """
    
    cursor.execute(query, (cutoff.isoformat(),))
    rows = cursor.fetchall()
    
    signals = []
    for row in rows:
        signal = dict(row)
        # Parse metadata JSON
        if signal.get('metadata'):
            try:
                signal['metadata'] = json.loads(signal['metadata'])
            except:
                pass
        signals.append(signal)
    
    conn.close()
    return signals


def analyze_rr_ratios(signals: List[Dict]) -> Dict:
    """Analyze risk/reward ratios from signal metadata"""
    
    rr_ratios = []
    upside_values = []
    downside_values = []
    confidence_values = []
    
    by_action = defaultdict(list)
    by_symbol = defaultdict(list)
    
    for signal in signals:
        metadata = signal.get('metadata', {})
        if not isinstance(metadata, dict):
            continue
        
        rr = metadata.get('risk_reward_ratio')
        upside = metadata.get('upside')
        downside = metadata.get('downside')
        confidence = signal.get('confidence', 0)
        action = signal.get('action', 'HOLD')
        symbol = signal.get('symbol', 'UNKNOWN')
        
        if rr is not None:
            rr_ratios.append(rr)
            by_action[action].append(rr)
            by_symbol[symbol].append(rr)
        
        if upside is not None:
            upside_values.append(upside)
        
        if downside is not None:
            downside_values.append(downside)
        
        if confidence > 0:
            confidence_values.append(confidence)
    
    # Calculate statistics
    stats = {
        'total_signals': len(signals),
        'signals_with_rr': len(rr_ratios),
        'avg_rr_ratio': sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0,
        'min_rr_ratio': min(rr_ratios) if rr_ratios else 0,
        'max_rr_ratio': max(rr_ratios) if rr_ratios else 0,
        'avg_upside': sum(upside_values) / len(upside_values) if upside_values else 0,
        'avg_downside': sum(downside_values) / len(downside_values) if downside_values else 0,
        'avg_confidence': sum(confidence_values) / len(confidence_values) if confidence_values else 0,
        'by_action': {},
        'by_symbol': {}
    }
    
    # Per-action stats
    for action, ratios in by_action.items():
        stats['by_action'][action] = {
            'count': len(ratios),
            'avg_rr': sum(ratios) / len(ratios)
        }
    
    # Per-symbol stats
    for symbol, ratios in by_symbol.items():
        stats['by_symbol'][symbol] = {
            'count': len(ratios),
            'avg_rr': sum(ratios) / len(ratios)
        }
    
    return stats


def check_confidence_adjustments(signals: List[Dict]) -> Dict:
    """Check how many signals had confidence adjusted by R/R logic"""
    
    excellent_rr = 0  # R/R > 2.0 (boosted)
    good_rr = 0       # R/R > 1.5
    poor_rr = 0       # R/R 0.7-1.3 (reduced)
    bearish_rr = 0    # R/R < 0.5
    
    for signal in signals:
        metadata = signal.get('metadata', {})
        if not isinstance(metadata, dict):
            continue
        
        rr = metadata.get('risk_reward_ratio', 0)
        
        if rr > 2.0:
            excellent_rr += 1
        elif rr > 1.5:
            good_rr += 1
        elif 0.7 <= rr <= 1.3:
            poor_rr += 1
        elif rr < 0.5:
            bearish_rr += 1
    
    return {
        'excellent_rr_count': excellent_rr,
        'good_rr_count': good_rr,
        'poor_rr_count': poor_rr,
        'bearish_rr_count': bearish_rr,
        'excellent_pct': (excellent_rr / len(signals) * 100) if signals else 0,
        'poor_pct': (poor_rr / len(signals) * 100) if signals else 0
    }


def main():
    """Monitor TFT signals and R/R ratios"""
    
    print("\n" + "="*70)
    print("[CHART] TFT SIGNAL MONITORING - LIVE PERFORMANCE")
    print("="*70 + "\n")
    
    db_path = Path("data/execution_journal.db")
    
    if not db_path.exists():
        print(f"❌ Execution journal not found: {db_path}")
        print("💡 Signals will appear after backend starts generating predictions")
        return
    
    # Get signals from last 24 hours
    print("[SEARCH] Fetching signals from last 24 hours...")
    signals = get_recent_signals(str(db_path), hours=24)
    
    if not signals:
        print("[WARNING]  No TFT signals found in last 24 hours")
        print("💡 Try:\n   - Checking if backend is running")
        print("   - Waiting for market activity")
        print("   - Checking different time window")
        return
    
    print(f"[OK] Found {len(signals)} TFT signals\n")
    
    # Analyze R/R ratios
    print("="*70)
    print("[CHART_UP] RISK/REWARD ANALYSIS")
    print("="*70 + "\n")
    
    stats = analyze_rr_ratios(signals)
    
    print(f"[CHART] Overall Statistics:")
    print(f"   Total signals: {stats['total_signals']}")
    print(f"   Signals with R/R: {stats['signals_with_rr']}")
    print(f"   Average R/R ratio: {stats['avg_rr_ratio']:.2f}:1")
    print(f"   Min R/R ratio: {stats['min_rr_ratio']:.2f}:1")
    print(f"   Max R/R ratio: {stats['max_rr_ratio']:.2f}:1")
    print(f"   Average upside: {stats['avg_upside']*100:.2f}%")
    print(f"   Average downside: {stats['avg_downside']*100:.2f}%")
    print(f"   Average confidence: {stats['avg_confidence']:.2f}")
    
    # R/R distribution
    print(f"\n[TARGET] Confidence Adjustments:")
    adjustments = check_confidence_adjustments(signals)
    print(f"   Excellent R/R (>2.0): {adjustments['excellent_rr_count']} ({adjustments['excellent_pct']:.1f}%) [×1.15 boost]")
    print(f"   Good R/R (>1.5): {adjustments['good_rr_count']}")
    print(f"   Poor R/R (0.7-1.3): {adjustments['poor_rr_count']} ({adjustments['poor_pct']:.1f}%) [×0.85 penalty]")
    print(f"   Bearish R/R (<0.5): {adjustments['bearish_rr_count']}")
    
    # Per-action breakdown
    if stats['by_action']:
        print(f"\n[CHART] By Action:")
        for action, data in stats['by_action'].items():
            print(f"   {action:6s}: {data['count']:3d} signals, avg R/R {data['avg_rr']:.2f}:1")
    
    # Top/bottom symbols by R/R
    if stats['by_symbol']:
        print(f"\n🏆 Top 5 Symbols by R/R:")
        sorted_symbols = sorted(stats['by_symbol'].items(), 
                                key=lambda x: x[1]['avg_rr'], 
                                reverse=True)[:5]
        for symbol, data in sorted_symbols:
            print(f"   {symbol:12s}: {data['avg_rr']:.2f}:1 ({data['count']} signals)")
    
    # Recent signals
    print(f"\n[CLIPBOARD] Most Recent Signals (last 10):")
    print("-"*70)
    for i, signal in enumerate(signals[:10]):
        timestamp = signal.get('timestamp', 'N/A')
        symbol = signal.get('symbol', 'N/A')
        action = signal.get('action', 'N/A')
        confidence = signal.get('confidence', 0)
        
        metadata = signal.get('metadata', {})
        if isinstance(metadata, dict):
            rr = metadata.get('risk_reward_ratio', 0)
            upside = metadata.get('upside', 0)
            downside = metadata.get('downside', 0)
        else:
            rr = upside = downside = 0
        
        print(f"{i+1:2d}. {timestamp[:19]} | {symbol:12s} | {action:4s} | "
              f"Conf: {confidence:.2f} | R/R: {rr:.2f} | "
              f"↑{upside*100:5.1f}% ↓{downside*100:5.1f}%")
    
    # Recommendations
    print(f"\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70 + "\n")
    
    avg_rr = stats['avg_rr_ratio']
    excellent_pct = adjustments['excellent_pct']
    poor_pct = adjustments['poor_pct']
    
    if avg_rr >= 1.5:
        print("[OK] GOOD: Average R/R ratio is healthy (≥1.5:1)")
    elif avg_rr >= 1.2:
        print("[WARNING]  OK: Average R/R ratio acceptable but could improve (1.2-1.5:1)")
    else:
        print(f"❌ CONCERN: Average R/R ratio low (<1.2:1)")
        print("   → Consider retraining with higher quantile_weight (0.7-0.8)")
    
    if excellent_pct >= 20:
        print(f"[OK] GOOD: {excellent_pct:.1f}% of signals have excellent R/R (>2.0)")
    elif excellent_pct >= 10:
        print(f"[WARNING]  OK: {excellent_pct:.1f}% of signals have excellent R/R")
    else:
        print(f"❌ CONCERN: Only {excellent_pct:.1f}% of signals have excellent R/R")
        print("   → Model may need more asymmetric upside focus")
    
    if poor_pct >= 30:
        print(f"[WARNING]  CONCERN: {poor_pct:.1f}% of signals have poor R/R (0.7-1.3)")
        print("   → Many symmetric/uncertain predictions")
    
    print("\n📅 Next Review: 2025-11-26 (1 week)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
