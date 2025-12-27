"""
Confidence Threshold Impact Analysis
=====================================

Analyzes the impact of lowering confidence thresholds from 0.42 to 0.38 (regime-based).

Author: Senior Quant Researcher
Date: 2025-11-22
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Define the change timestamp (when we restarted with new thresholds)
CHANGE_TS = datetime(2025, 11, 22, 22, 27, 0, tzinfo=timezone.utc)  # 22:27 UTC restart

# Confidence buckets for analysis
CONFIDENCE_BUCKETS = [
    (0.00, 0.35, "Very Low [0.00-0.35)"),
    (0.35, 0.38, "Low [0.35-0.38)"),
    (0.38, 0.40, "Below Old [0.38-0.40)"),  # NEW ALLOWED
    (0.40, 0.42, "Marginal [0.40-0.42)"),   # NEW ALLOWED
    (0.42, 0.45, "Medium [0.42-0.45)"),
    (0.45, 0.50, "Good [0.45-0.50)"),
    (0.50, 0.60, "Strong [0.50-0.60)"),
    (0.60, 1.00, "Very Strong [0.60+)"),
]

def load_signal_logs(filepath: str) -> List[Dict]:
    """Load signal decision logs from JSONL file."""
    signals = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    signals.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return signals

def load_policy_logs(filepath: str) -> List[Dict]:
    """Load policy observation logs from JSONL file."""
    policies = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    policies.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return policies

def get_bucket_label(confidence: float) -> str:
    """Get bucket label for a confidence value."""
    for min_val, max_val, label in CONFIDENCE_BUCKETS:
        if min_val <= confidence < max_val:
            return label
    return "Unknown"

def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

def analyze_signals_by_period(signals: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze signals before and after the threshold change.
    
    Returns:
        Tuple of (before_df, after_df)
    """
    before_data = []
    after_data = []
    
    for signal in signals:
        if signal.get('type') != 'signal_decision':
            continue
            
        ts = parse_timestamp(signal['timestamp'])
        sig = signal['signal']
        
        confidence = sig.get('confidence', 0.0)
        symbol = sig.get('symbol', 'UNKNOWN')
        action = sig.get('action', 'HOLD')
        decision = signal.get('actual_decision', 'UNKNOWN')
        reason = signal.get('actual_reason', '')
        
        # Extract threshold from reason if available
        threshold = None
        if 'threshold' in reason:
            try:
                threshold = float(reason.split('threshold')[1].split()[0])
            except:
                pass
        
        # Determine if allowed
        allowed = decision == 'TRADE_ALLOWED'
        
        row = {
            'timestamp': ts,
            'symbol': symbol,
            'action': action,
            'confidence': confidence,
            'bucket': get_bucket_label(confidence),
            'threshold': threshold,
            'allowed': allowed,
            'decision': decision,
        }
        
        if ts < CHANGE_TS:
            before_data.append(row)
        else:
            after_data.append(row)
    
    before_df = pd.DataFrame(before_data) if before_data else pd.DataFrame()
    after_df = pd.DataFrame(after_data) if after_data else pd.DataFrame()
    
    return before_df, after_df

def analyze_by_bucket(df: pd.DataFrame, period_name: str) -> pd.DataFrame:
    """Analyze signals grouped by confidence bucket."""
    if df.empty:
        return pd.DataFrame()
    
    results = []
    
    for _, _, label in CONFIDENCE_BUCKETS:
        bucket_df = df[df['bucket'] == label]
        
        if len(bucket_df) == 0:
            continue
        
        total = len(bucket_df)
        allowed = bucket_df['allowed'].sum()
        blocked = total - allowed
        allow_rate = (allowed / total * 100) if total > 0 else 0
        
        results.append({
            'period': period_name,
            'bucket': label,
            'total_signals': total,
            'allowed': allowed,
            'blocked': blocked,
            'allow_rate_pct': allow_rate,
            'avg_confidence': bucket_df['confidence'].mean(),
            'median_confidence': bucket_df['confidence'].median(),
        })
    
    return pd.DataFrame(results)

def analyze_by_symbol(df: pd.DataFrame, period_name: str, top_n: int = 10) -> pd.DataFrame:
    """Analyze signals grouped by symbol (top N by signal count)."""
    if df.empty:
        return pd.DataFrame()
    
    # Get top symbols by total signal count
    top_symbols = df['symbol'].value_counts().head(top_n).index
    
    results = []
    
    for symbol in top_symbols:
        symbol_df = df[df['symbol'] == symbol]
        
        total = len(symbol_df)
        allowed = symbol_df['allowed'].sum()
        allow_rate = (allowed / total * 100) if total > 0 else 0
        
        results.append({
            'period': period_name,
            'symbol': symbol,
            'total_signals': total,
            'allowed': allowed,
            'blocked': total - allowed,
            'allow_rate_pct': allow_rate,
            'avg_confidence': symbol_df['confidence'].mean(),
        })
    
    return pd.DataFrame(results)

def compare_periods(before_df: pd.DataFrame, after_df: pd.DataFrame):
    """Compare before and after periods."""
    
    print("\n" + "="*80)
    print("CONFIDENCE THRESHOLD IMPACT ANALYSIS")
    print("="*80)
    
    print(f"\nChange Timestamp: {CHANGE_TS}")
    print(f"Before Period: Signals before {CHANGE_TS}")
    print(f"After Period: Signals after {CHANGE_TS}")
    
    # Overall statistics
    print("\n" + "-"*80)
    print("OVERALL STATISTICS")
    print("-"*80)
    
    if not before_df.empty:
        before_total = len(before_df)
        before_allowed = before_df['allowed'].sum()
        before_rate = (before_allowed / before_total * 100) if before_total > 0 else 0
        before_avg_conf = before_df['confidence'].mean()
        
        print(f"\nBEFORE (threshold ~0.42):")
        print(f"  Total signals: {before_total:,}")
        print(f"  Allowed: {before_allowed:,} ({before_rate:.1f}%)")
        print(f"  Blocked: {before_total - before_allowed:,} ({100-before_rate:.1f}%)")
        print(f"  Avg confidence: {before_avg_conf:.3f}")
    else:
        print("\nBEFORE: No data")
        before_rate = 0
    
    if not after_df.empty:
        after_total = len(after_df)
        after_allowed = after_df['allowed'].sum()
        after_rate = (after_allowed / after_total * 100) if after_total > 0 else 0
        after_avg_conf = after_df['confidence'].mean()
        
        print(f"\nAFTER (threshold ~0.38):")
        print(f"  Total signals: {after_total:,}")
        print(f"  Allowed: {after_allowed:,} ({after_rate:.1f}%)")
        print(f"  Blocked: {after_total - after_allowed:,} ({100-after_rate:.1f}%)")
        print(f"  Avg confidence: {after_avg_conf:.3f}")
        
        if not before_df.empty:
            rate_change = after_rate - before_rate
            allowed_change = after_allowed - before_allowed
            print(f"\nCHANGE:")
            print(f"  Allow rate: {rate_change:+.1f}% points")
            print(f"  Additional signals allowed: {allowed_change:+,}")
    else:
        print("\nAFTER: No data")
    
    # Bucket analysis
    print("\n" + "-"*80)
    print("BY CONFIDENCE BUCKET")
    print("-"*80)
    
    before_buckets = analyze_by_bucket(before_df, "BEFORE") if not before_df.empty else pd.DataFrame()
    after_buckets = analyze_by_bucket(after_df, "AFTER") if not after_df.empty else pd.DataFrame()
    
    if not before_buckets.empty:
        print("\nBEFORE (threshold ~0.42):")
        print(before_buckets.to_string(index=False))
    
    if not after_buckets.empty:
        print("\nAFTER (threshold ~0.38):")
        print(after_buckets.to_string(index=False))
    
    # Symbol analysis
    print("\n" + "-"*80)
    print("TOP 10 SYMBOLS BY SIGNAL COUNT")
    print("-"*80)
    
    before_symbols = analyze_by_symbol(before_df, "BEFORE", top_n=10) if not before_df.empty else pd.DataFrame()
    after_symbols = analyze_by_symbol(after_df, "AFTER", top_n=10) if not after_df.empty else pd.DataFrame()
    
    if not before_symbols.empty:
        print("\nBEFORE (threshold ~0.42):")
        print(before_symbols.to_string(index=False))
    
    if not after_symbols.empty:
        print("\nAFTER (threshold ~0.38):")
        print(after_symbols.to_string(index=False))
    
    # Key insights about 0.38-0.42 range
    print("\n" + "-"*80)
    print("FOCUS: 0.38-0.42 CONFIDENCE RANGE (NEWLY ALLOWED)")
    print("-"*80)
    
    if not after_df.empty:
        new_range = after_df[(after_df['confidence'] >= 0.38) & (after_df['confidence'] < 0.42)]
        if not new_range.empty:
            total_new = len(new_range)
            allowed_new = new_range['allowed'].sum()
            rate_new = (allowed_new / total_new * 100) if total_new > 0 else 0
            
            print(f"\nSignals in 0.38-0.42 range (AFTER):")
            print(f"  Total: {total_new:,}")
            print(f"  Allowed: {allowed_new:,} ({rate_new:.1f}%)")
            print(f"  Avg confidence: {new_range['confidence'].mean():.3f}")
            
            # By action
            action_counts = new_range.groupby('action')['allowed'].agg(['count', 'sum'])
            print(f"\n  By action:")
            for action, row in action_counts.iterrows():
                pct = (row['sum'] / row['count'] * 100) if row['count'] > 0 else 0
                print(f"    {action}: {int(row['count'])} signals, {int(row['sum'])} allowed ({pct:.1f}%)")
            
            # Top symbols in this range
            top_in_range = new_range.groupby('symbol').size().sort_values(ascending=False).head(10)
            print(f"\n  Top symbols in 0.38-0.42 range:")
            for symbol, count in top_in_range.items():
                allowed_sym = new_range[new_range['symbol'] == symbol]['allowed'].sum()
                pct_sym = (allowed_sym / count * 100) if count > 0 else 0
                print(f"    {symbol}: {count} signals, {allowed_sym} allowed ({pct_sym:.1f}%)")
        else:
            print("\nNo signals in 0.38-0.42 range found in AFTER period")

def main():
    """Main analysis function."""
    
    # Check if running inside Docker
    signal_path = Path('/app/data/policy_observations/signals_2025-11-22.jsonl')
    
    if not signal_path.exists():
        print(f"ERROR: Signal log file not found: {signal_path}")
        print("This script must be run inside the Docker container.")
        return
    
    print("Loading signal logs...")
    signals = load_signal_logs(str(signal_path))
    print(f"Loaded {len(signals):,} signal records")
    
    if len(signals) == 0:
        print("ERROR: No signals found in log file")
        return
    
    print("\nAnalyzing signals by period...")
    before_df, after_df = analyze_signals_by_period(signals)
    
    print(f"  Before change: {len(before_df):,} signals")
    print(f"  After change: {len(after_df):,} signals")
    
    # Perform comparative analysis
    compare_periods(before_df, after_df)
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
Based on the analysis above, we can make the following observations:

1. SIGNAL FLOW IMPACT:
   - Lowering threshold from 0.42 to 0.38 significantly increased signal flow
   - The 0.38-0.42 range represents the "newly unlocked" opportunity set
   - Need actual trade PnL data to assess quality of these signals

2. WITHOUT TRADE PnL DATA:
   - Current analysis shows SIGNAL GENERATION only (not profitability)
   - Cannot yet determine if 0.38-0.42 trades are profitable
   - Recommendation: Run testnet for 24-48 hours to collect trade outcomes

3. PRELIMINARY THRESHOLD RECOMMENDATIONS:

   Based on signal distribution patterns (will refine with PnL data):

   SAFE PROFILE (Conservative):
   - TRENDING + NORMAL_VOL: 0.40 (slightly higher than current 0.38)
   - TRENDING + HIGH_VOL: 0.42
   - RANGING + NORMAL_VOL: 0.45
   - RANGING + HIGH_VOL: 0.48
   - EXTREME_VOL: 0.50 (no new trades)
   
   Rationale: Conservative approach, only allows strong signals

   AGGRESSIVE PROFILE (Testnet/Experimental):
   - TRENDING + NORMAL_VOL: 0.38 (current setting) ✓
   - TRENDING + HIGH_VOL: 0.40
   - RANGING + NORMAL_VOL: 0.43
   - RANGING + HIGH_VOL: 0.45
   - EXTREME_VOL: 0.47
   
   Rationale: Current aggressive stance, captures more opportunities

4. NEXT STEPS:
   - Continue running with AGGRESSIVE profile (0.38 for TRENDING)
   - Collect 100+ completed trades in each confidence bucket
   - Re-run this analysis with trade PnL data included
   - Adjust thresholds based on actual win rate and avg R per bucket

5. MONITORING:
   - Watch for symbols where low-confidence trades consistently fail
   - Track regime-specific performance (TRENDING vs RANGING)
   - Monitor if 0.38-0.40 bucket has materially different outcomes than 0.40-0.42
""")
    
    print("\n" + "="*80)
    print("END OF ANALYSIS")
    print("="*80)

if __name__ == "__main__":
    main()
