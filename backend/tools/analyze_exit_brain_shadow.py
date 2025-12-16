"""
Analyze Exit Brain V3 Shadow Logs

This script analyzes shadow mode logs to evaluate AI decision quality
before enabling LIVE mode.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any

import pandas as pd


SHADOW_LOG_FILE = Path("backend/data/exit_brain_shadow.jsonl")


def load_shadow_logs() -> List[Dict[str, Any]]:
    """Load shadow logs from JSONL file."""
    if not SHADOW_LOG_FILE.exists():
        print(f"❌ Shadow log file not found: {SHADOW_LOG_FILE}")
        print(f"   Make sure system is running in SHADOW mode")
        sys.exit(1)
    
    logs = []
    with open(SHADOW_LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping malformed line: {e}")
    
    return logs


def analyze_decision_distribution(df: pd.DataFrame) -> None:
    """Analyze distribution of decision types."""
    print("\n" + "="*60)
    print("📊 DECISION TYPE DISTRIBUTION")
    print("="*60)
    
    decision_col = 'decision.decision_type' if 'decision.decision_type' in df.columns else 'decision_type'
    
    if decision_col not in df.columns:
        print("⚠️  No decision type data available")
        return
    
    distribution = df[decision_col].value_counts()
    total = len(df)
    
    for decision_type, count in distribution.items():
        pct = (count / total) * 100
        print(f"  {decision_type:20s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\n  Total decisions: {total}")


def analyze_confidence_scores(df: pd.DataFrame) -> None:
    """Analyze confidence scores by decision type."""
    print("\n" + "="*60)
    print("📈 CONFIDENCE SCORES BY DECISION TYPE")
    print("="*60)
    
    decision_col = 'decision.decision_type' if 'decision.decision_type' in df.columns else 'decision_type'
    confidence_col = 'decision.confidence' if 'decision.confidence' in df.columns else 'confidence'
    
    if decision_col not in df.columns or confidence_col not in df.columns:
        print("⚠️  Confidence data not available")
        return
    
    grouped = df.groupby(decision_col)[confidence_col].agg(['mean', 'std', 'min', 'max', 'count'])
    
    print(f"\n{'Decision Type':<20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Count':>8s}")
    print("-" * 75)
    
    for decision_type, row in grouped.iterrows():
        print(f"{decision_type:<20s} {row['mean']:>8.3f} {row['std']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f} {int(row['count']):>8d}")


def analyze_emergency_exits(df: pd.DataFrame) -> None:
    """Analyze emergency exit suggestions."""
    print("\n" + "="*60)
    print("🚨 EMERGENCY EXIT ANALYSIS")
    print("="*60)
    
    decision_col = 'decision.decision_type' if 'decision.decision_type' in df.columns else 'decision_type'
    
    if decision_col not in df.columns:
        print("⚠️  No decision type data available")
        return
    
    emergency_exits = df[df[decision_col] == 'full_exit_now']
    
    if len(emergency_exits) == 0:
        print("  ✅ No emergency exits suggested (good sign)")
        return
    
    print(f"  ⚠️  {len(emergency_exits)} emergency exits suggested:")
    
    symbol_col = 'symbol' if 'symbol' in emergency_exits.columns else 'position.symbol'
    if symbol_col in emergency_exits.columns:
        symbols = emergency_exits[symbol_col].value_counts()
        for symbol, count in symbols.items():
            print(f"     {symbol}: {count} times")
        
        # Show some examples
        print("\n  Recent examples:")
        for idx, row in emergency_exits.head(5).iterrows():
            reason_col = 'decision.reasoning' if 'decision.reasoning' in row.index else 'reasoning'
            reasoning = row.get(reason_col, 'No reasoning provided')
            print(f"     - {row.get(symbol_col, 'Unknown')}: {reasoning[:80]}...")


def analyze_symbol_patterns(df: pd.DataFrame) -> None:
    """Analyze patterns by symbol."""
    print("\n" + "="*60)
    print("💱 SYMBOL-SPECIFIC PATTERNS")
    print("="*60)
    
    symbol_col = 'symbol' if 'symbol' in df.columns else 'position.symbol'
    decision_col = 'decision.decision_type' if 'decision.decision_type' in df.columns else 'decision_type'
    
    if symbol_col not in df.columns or decision_col not in df.columns:
        print("⚠️  Symbol/decision data not available")
        return
    
    # Get top symbols by activity
    top_symbols = df[symbol_col].value_counts().head(10)
    
    print(f"\n  Top 10 Most Active Symbols:")
    for symbol, count in top_symbols.items():
        symbol_df = df[df[symbol_col] == symbol]
        decisions = symbol_df[decision_col].value_counts()
        decision_str = ", ".join([f"{dt}: {ct}" for dt, ct in decisions.head(3).items()])
        print(f"     {symbol:12s}: {count:4d} decisions ({decision_str})")


def analyze_temporal_patterns(df: pd.DataFrame) -> None:
    """Analyze temporal patterns."""
    print("\n" + "="*60)
    print("⏰ TEMPORAL ANALYSIS")
    print("="*60)
    
    timestamp_col = 'timestamp' if 'timestamp' in df.columns else None
    
    if timestamp_col is None or timestamp_col not in df.columns:
        print("⚠️  Timestamp data not available")
        return
    
    try:
        df['datetime'] = pd.to_datetime(df[timestamp_col])
        df['hour'] = df['datetime'].dt.hour
        
        print("\n  Decisions by Hour of Day:")
        hourly = df['hour'].value_counts().sort_index()
        for hour, count in hourly.items():
            bar = "█" * (count // 5)
            print(f"     {hour:02d}:00 - {hour:02d}:59: {count:4d} {bar}")
        
        # Data collection period
        print(f"\n  Data Collection Period:")
        print(f"     Start: {df['datetime'].min()}")
        print(f"     End:   {df['datetime'].max()}")
        print(f"     Duration: {df['datetime'].max() - df['datetime'].min()}")
    except Exception as e:
        print(f"⚠️  Error analyzing temporal patterns: {e}")


def generate_recommendations(df: pd.DataFrame) -> None:
    """Generate recommendations for LIVE mode activation."""
    print("\n" + "="*60)
    print("✅ RECOMMENDATIONS FOR LIVE MODE")
    print("="*60)
    
    decision_col = 'decision.decision_type' if 'decision.decision_type' in df.columns else 'decision_type'
    confidence_col = 'decision.confidence' if 'decision.confidence' in df.columns else 'confidence'
    
    issues = []
    good_signs = []
    
    # Check data volume
    if len(df) < 50:
        issues.append(f"Limited data: Only {len(df)} decisions. Run shadow mode longer (24-48h recommended).")
    elif len(df) < 200:
        good_signs.append(f"Adequate data: {len(df)} decisions collected.")
    else:
        good_signs.append(f"Excellent data: {len(df)} decisions collected.")
    
    # Check decision diversity
    if decision_col in df.columns:
        unique_decisions = df[decision_col].nunique()
        if unique_decisions < 2:
            issues.append(f"Low decision diversity: Only {unique_decisions} decision type(s) observed.")
        else:
            good_signs.append(f"Good decision diversity: {unique_decisions} different decision types.")
    
    # Check confidence levels
    if confidence_col in df.columns:
        avg_confidence = df[confidence_col].mean()
        if avg_confidence < 0.5:
            issues.append(f"Low average confidence: {avg_confidence:.3f}. AI may not be ready.")
        elif avg_confidence < 0.7:
            good_signs.append(f"Moderate confidence: {avg_confidence:.3f}. Monitor closely in LIVE mode.")
        else:
            good_signs.append(f"High confidence: {avg_confidence:.3f}. AI appears confident.")
    
    # Check emergency exits
    if decision_col in df.columns:
        emergency_count = len(df[df[decision_col] == 'full_exit_now'])
        emergency_pct = (emergency_count / len(df)) * 100 if len(df) > 0 else 0
        if emergency_pct > 30:
            issues.append(f"High emergency exit rate: {emergency_pct:.1f}%. Review market conditions.")
        elif emergency_pct < 5:
            good_signs.append(f"Low emergency exit rate: {emergency_pct:.1f}%. System appears stable.")
    
    # Display results
    print("\n✅ Good Signs:")
    if good_signs:
        for sign in good_signs:
            print(f"   • {sign}")
    else:
        print("   None identified")
    
    print("\n⚠️  Concerns:")
    if issues:
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("   None identified")
    
    # Final recommendation
    print("\n🎯 Final Recommendation:")
    if len(issues) == 0 and len(good_signs) >= 3:
        print("   ✅ System appears READY for LIVE mode activation")
        print("   ➡️  Next step: Run pre-LIVE validation")
        print("   ➡️  Command: .\\scripts\\deploy_exit_brain_v3_live.ps1 -Mode prelive")
    elif len(issues) <= 1 and len(good_signs) >= 2:
        print("   ⚠️  System MOSTLY ready, but monitor closely")
        print("   ➡️  Consider extending shadow mode collection")
        print("   ➡️  Or proceed with caution and close monitoring")
    else:
        print("   ❌ System NOT ready for LIVE mode")
        print("   ➡️  Address concerns above")
        print("   ➡️  Continue shadow mode collection")


def main():
    """Main analysis function."""
    print("\n" + "="*60)
    print("🔍 EXIT BRAIN V3 SHADOW LOG ANALYSIS")
    print("="*60)
    print(f"Analyzing: {SHADOW_LOG_FILE}")
    
    # Load logs
    logs = load_shadow_logs()
    
    if not logs:
        print("\n❌ No shadow logs found or file is empty")
        print("\nMake sure:")
        print("  1. Backend is running in SHADOW mode")
        print("  2. System has open positions to monitor")
        print("  3. Shadow log file is being written to")
        sys.exit(1)
    
    print(f"✅ Loaded {len(logs)} shadow log entries")
    
    # Convert to DataFrame
    df = pd.json_normalize(logs)
    
    # Run analyses
    analyze_decision_distribution(df)
    analyze_confidence_scores(df)
    analyze_emergency_exits(df)
    analyze_symbol_patterns(df)
    analyze_temporal_patterns(df)
    generate_recommendations(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
