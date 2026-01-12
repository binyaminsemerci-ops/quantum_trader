#!/usr/bin/env python3
"""
Quality Gate (Simplified) - Check recent predictions from trade_intents

DETECTS degenerate behavior from production predictions (no model loading needed)
"""

import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def get_recent_predictions(db_path, model_name, hours=24, limit=500):
    """Get recent predictions from trade_intents"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    cursor.execute("""
        SELECT action, confidence, metadata
        FROM trade_intents
        WHERE model_source LIKE ?
        AND created_at > ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (f'%{model_name}%', cutoff.strftime('%Y-%m-%d %H:%M:%S'), limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows


def analyze_predictions(predictions):
    """Analyze prediction distribution"""
    if not predictions:
        return None
    
    actions = [row[0] for row in predictions]
    confidences = [row[1] for row in predictions]
    
    action_counts = {
        'BUY': actions.count('BUY'),
        'SELL': actions.count('SELL'),
        'HOLD': actions.count('HOLD')
    }
    total = len(actions)
    action_pcts = {k: v/total*100 for k, v in action_counts.items()}
    
    conf_stats = {
        'mean': np.mean(confidences),
        'std': np.std(confidences),
        'p10': np.percentile(confidences, 10),
        'p50': np.percentile(confidences, 50),
        'p90': np.percentile(confidences, 90),
        'p10_p90_range': np.percentile(confidences, 90) - np.percentile(confidences, 10),
        'unique_count': len(np.unique(confidences))
    }
    
    return {
        'action_counts': action_counts,
        'action_pcts': action_pcts,
        'confidence_stats': conf_stats,
        'sample_count': total
    }


def check_quality_gate(analysis):
    """HARD CHECKS - FAIL IF ANY VIOLATION"""
    failures = []
    
    # Check 1: No class >70%
    for action, pct in analysis['action_pcts'].items():
        if pct > 70:
            failures.append(f"Action {action} = {pct:.1f}% > 70% (MAJORITY BIAS)")
    
    # Check 2: Confidence spread
    if analysis['confidence_stats']['std'] < 0.05:
        failures.append(f"Confidence std = {analysis['confidence_stats']['std']:.4f} < 0.05 (COLLAPSED)")
    
    if analysis['confidence_stats']['p10_p90_range'] < 0.12:
        failures.append(f"Confidence P10-P90 = {analysis['confidence_stats']['p10_p90_range']:.4f} < 0.12 (NARROW RANGE)")
    
    # Check 3: HOLD dead-zone collapse
    hold_pct = analysis['action_pcts']['HOLD']
    if hold_pct > 85:
        failures.append(f"HOLD = {hold_pct:.1f}% > 85% (DEAD-ZONE COLLAPSE)")
    
    return failures


def generate_report(model_name, analysis, failures, report_path):
    """Generate markdown report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    status = "❌ FAIL (BLOCKER)" if failures else "✅ PASS"
    
    with open(report_path, 'w') as f:
        f.write(f"# Quality Gate Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Status:** {status}\n")
        f.write(f"**Sample Count:** {analysis['sample_count']}\n\n")
        
        f.write(f"## Action Distribution\n\n")
        for action, count in analysis['action_counts'].items():
            pct = analysis['action_pcts'][action]
            f.write(f"- **{action}**: {count} ({pct:.1f}%)\n")
        
        f.write(f"\n## Confidence Statistics\n\n")
        cs = analysis['confidence_stats']
        f.write(f"- Mean: {cs['mean']:.4f}\n")
        f.write(f"- Stddev: {cs['std']:.4f}\n")
        f.write(f"- P10: {cs['p10']:.4f}\n")
        f.write(f"- P50: {cs['p50']:.4f}\n")
        f.write(f"- P90: {cs['p90']:.4f}\n")
        f.write(f"- P10-P90 Range: {cs['p10_p90_range']:.4f}\n")
        f.write(f"- Unique values: {cs['unique_count']}\n")
        
        f.write(f"\n## Quality Gate Checks\n\n")
        if failures:
            f.write(f"**FAILED ({len(failures)} violations):**\n\n")
            for fail in failures:
                f.write(f"- ❌ {fail}\n")
        else:
            f.write(f"**ALL CHECKS PASSED**\n\n")
            f.write(f"- ✅ No class >70%\n")
            f.write(f"- ✅ Confidence std ≥0.05\n")
            f.write(f"- ✅ Confidence P10-P90 ≥0.12\n")
            f.write(f"- ✅ No HOLD dead-zone collapse (HOLD ≤85%)\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 ops/model_safety/quality_gate_simple.py <model_name>")
        print("Example: python3 ops/model_safety/quality_gate_simple.py patchtst")
        return 1
    
    model_name = sys.argv[1]
    db_path = Path('/opt/quantum/data/quantum_trader.db')
    report_dir = Path('reports/safety')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'quality_gate_{model_name}_{timestamp_str}.md'
    
    print(f"{'='*70}")
    print(f"QUALITY GATE - {model_name.upper()}")
    print(f"{'='*70}\n")
    
    # Get recent predictions
    print(f"Loading recent predictions (last 24h)...")
    predictions = get_recent_predictions(db_path, model_name, hours=24, limit=500)
    
    if not predictions:
        print(f"❌ No recent predictions found for model: {model_name}")
        return 2
    
    print(f"Loaded {len(predictions)} predictions\n")
    
    # Analyze
    print(f"Analyzing predictions...")
    analysis = analyze_predictions(predictions)
    
    # Quality gate checks
    print(f"\n{'='*70}")
    print(f"QUALITY GATE CHECKS")
    print(f"{'='*70}\n")
    
    failures = check_quality_gate(analysis)
    
    # Display action distribution
    print(f"Action Distribution:")
    for action, pct in analysis['action_pcts'].items():
        print(f"  {action:5s}: {analysis['action_counts'][action]:4d} ({pct:5.1f}%)")
    
    print(f"\nConfidence Stats:")
    cs = analysis['confidence_stats']
    print(f"  Mean:       {cs['mean']:.4f}")
    print(f"  Stddev:     {cs['std']:.4f}")
    print(f"  P10-P90:    {cs['p10_p90_range']:.4f}")
    print(f"  Unique:     {cs['unique_count']}")
    
    # Result
    print(f"\n{'='*70}")
    if failures:
        print(f"❌ QUALITY GATE: FAIL (BLOCKER)")
        print(f"{'='*70}\n")
        for fail in failures:
            print(f"  - {fail}")
        exit_code = 2
    else:
        print(f"✅ QUALITY GATE: PASS")
        print(f"{'='*70}")
        exit_code = 0
    
    # Generate report
    generate_report(model_name, analysis, failures, report_path)
    print(f"\nReport: {report_path}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
