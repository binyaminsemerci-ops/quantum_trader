#!/usr/bin/env python3
"""
PatchTST Shadow Mode Metrics Analyzer
Extracts and analyzes shadow predictions vs ensemble decisions

Usage:
    python3 ops/analysis/analyze_shadow_metrics.py [--hours 2]
    
Metrics Computed:
- Action distribution (BUY/SELL/HOLD %)
- Confidence statistics (mean, stddev, histogram)
- Shadow vs Ensemble correlation
- Disagreement rate
- Calibration analysis
"""
import sys
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import numpy as np
import argparse

def parse_trade_intent_logs(log_lines):
    """Extract trade.intent payloads from logs"""
    events = []
    
    for line in log_lines:
        # Look for DEBUG lines with trade.intent payload
        if 'trade.intent' in line and 'model_breakdown' in line:
            try:
                # Extract JSON payload after "About to publish trade.intent:"
                match = re.search(r"'symbol':\s*'([^']+)'.*'side':\s*'([^']+)'.*'confidence':\s*([0-9.]+).*'model_breakdown':\s*({.*?})\s*,\s*'atr", line)
                if match:
                    symbol = match.group(1)
                    ensemble_action = match.group(2)
                    ensemble_conf = float(match.group(3))
                    
                    # Parse model_breakdown
                    breakdown_str = match.group(4)
                    # Fix single quotes to double quotes for JSON
                    breakdown_str = breakdown_str.replace("'", '"')
                    breakdown = json.loads(breakdown_str)
                    
                    event = {
                        'symbol': symbol,
                        'ensemble_action': ensemble_action,
                        'ensemble_conf': ensemble_conf,
                        'models': breakdown,
                        'timestamp': line.split()[0:3]  # Extract timestamp
                    }
                    events.append(event)
            except Exception as e:
                # Skip malformed lines
                continue
    
    return events

def analyze_shadow_predictions(events):
    """Analyze PatchTST shadow predictions vs ensemble"""
    
    print("=" * 70)
    print("PATCHTST SHADOW MODE ANALYSIS")
    print("=" * 70)
    print(f"\nTotal Events: {len(events)}")
    
    # Extract PatchTST predictions
    shadow_actions = []
    shadow_confs = []
    ensemble_actions = []
    ensemble_confs = []
    agreements = []
    shadow_flags = []
    
    for event in events:
        if 'patchtst' in event['models']:
            pt = event['models']['patchtst']
            
            shadow_actions.append(pt['action'])
            shadow_confs.append(pt['confidence'])
            ensemble_actions.append(event['ensemble_action'])
            ensemble_confs.append(event['ensemble_conf'])
            
            # Check agreement
            agrees = pt['action'] == event['ensemble_action']
            agreements.append(agrees)
            
            # Check shadow flag
            is_shadow = pt.get('shadow', False)
            shadow_flags.append(is_shadow)
    
    if not shadow_actions:
        print("\n⚠️  No PatchTST predictions found in logs")
        return
    
    print(f"PatchTST Predictions: {len(shadow_actions)}")
    print(f"Shadow Mode Active: {sum(shadow_flags)} / {len(shadow_flags)} ({sum(shadow_flags)/len(shadow_flags)*100:.1f}%)")
    
    # Action Distribution
    print(f"\n{'─'*70}")
    print("PATCHTST ACTION DISTRIBUTION")
    print(f"{'─'*70}")
    
    action_counts = Counter(shadow_actions)
    for action, count in action_counts.most_common():
        pct = (count / len(shadow_actions)) * 100
        print(f"  {action:5s}: {count:4d} ({pct:5.1f}%)")
    
    # ✅ EVAL GATE 1: Action Diversity (no class > 70%)
    max_action_pct = max(action_counts.values()) / len(shadow_actions)
    if max_action_pct > 0.70:
        print(f"\n  ❌ GATE FAIL: Action diversity ({max_action_pct:.1%} > 70% threshold)")
    else:
        print(f"\n  ✅ GATE PASS: Action diversity ({max_action_pct:.1%} ≤ 70%)")
    
    # Confidence Statistics
    print(f"\n{'─'*70}")
    print("CONFIDENCE STATISTICS")
    print(f"{'─'*70}")
    
    shadow_confs_arr = np.array(shadow_confs)
    print(f"  Min:    {shadow_confs_arr.min():.4f}")
    print(f"  Max:    {shadow_confs_arr.max():.4f}")
    print(f"  Mean:   {shadow_confs_arr.mean():.4f}")
    print(f"  Median: {np.median(shadow_confs_arr):.4f}")
    print(f"  Stddev: {shadow_confs_arr.std():.4f}")
    
    # ✅ EVAL GATE 2: Confidence Spread (stddev ≥ 0.05)
    if shadow_confs_arr.std() < 0.05:
        print(f"\n  ❌ GATE FAIL: Confidence spread (σ={shadow_confs_arr.std():.4f} < 0.05)")
    else:
        print(f"\n  ✅ GATE PASS: Confidence spread (σ={shadow_confs_arr.std():.4f} ≥ 0.05)")
    
    # Confidence Histogram
    print(f"\n  Confidence Histogram:")
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    hist, _ = np.histogram(shadow_confs_arr, bins=bins)
    for i in range(len(bins)-1):
        pct = (hist[i] / len(shadow_confs)) * 100
        bar = '█' * int(pct / 2)
        print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    # Shadow vs Ensemble Correlation
    print(f"\n{'─'*70}")
    print("SHADOW VS ENSEMBLE")
    print(f"{'─'*70}")
    
    agreement_rate = sum(agreements) / len(agreements)
    print(f"  Agreement Rate: {agreement_rate:.1%} ({sum(agreements)} / {len(agreements)})")
    
    # ✅ EVAL GATE 3: Shadow Correlation (≥ 55% agreement)
    if agreement_rate < 0.55:
        print(f"\n  ❌ GATE FAIL: Correlation ({agreement_rate:.1%} < 55%)")
    else:
        print(f"\n  ✅ GATE PASS: Correlation ({agreement_rate:.1%} ≥ 55%)")
    
    # Disagreement breakdown
    print(f"\n  Disagreement Breakdown:")
    disagreements = defaultdict(int)
    for i, agrees in enumerate(agreements):
        if not agrees:
            key = f"{shadow_actions[i]} vs {ensemble_actions[i]}"
            disagreements[key] += 1
    
    for key, count in sorted(disagreements.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(agreements)) * 100
        print(f"    {key:15s}: {count:4d} ({pct:5.1f}%)")
    
    # Calibration Analysis (simplified)
    print(f"\n{'─'*70}")
    print("CALIBRATION ANALYSIS")
    print(f"{'─'*70}")
    
    # Group by confidence buckets and check accuracy
    conf_buckets = [0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"  Confidence Bucket | Count | Agrees | Accuracy")
    print(f"  ─────────────────┼───────┼────────┼─────────")
    
    monotonic = True
    prev_acc = 0.0
    
    for i in range(len(conf_buckets)-1):
        low, high = conf_buckets[i], conf_buckets[i+1]
        
        # Find predictions in this bucket
        bucket_indices = [j for j, c in enumerate(shadow_confs) if low <= c < high]
        
        if bucket_indices:
            bucket_agrees = [agreements[j] for j in bucket_indices]
            bucket_acc = sum(bucket_agrees) / len(bucket_agrees)
            
            print(f"  [{low:.1f}, {high:.1f})      | {len(bucket_indices):5d} | {sum(bucket_agrees):6d} | {bucket_acc:6.1%}")
            
            # Check monotonicity
            if bucket_acc < prev_acc:
                monotonic = False
            prev_acc = bucket_acc
        else:
            print(f"  [{low:.1f}, {high:.1f})      |     0 |      - |       -")
    
    # ✅ EVAL GATE 4: Calibration (monotonic accuracy)
    if not monotonic:
        print(f"\n  ⚠️  GATE WARNING: Calibration not strictly monotonic")
    else:
        print(f"\n  ✅ GATE PASS: Calibration monotonic")
    
    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION GATES SUMMARY")
    print(f"{'='*70}")
    
    gates_passed = 0
    if max_action_pct <= 0.70:
        print("  ✅ Action Diversity")
        gates_passed += 1
    else:
        print("  ❌ Action Diversity")
    
    if shadow_confs_arr.std() >= 0.05:
        print("  ✅ Confidence Spread")
        gates_passed += 1
    else:
        print("  ❌ Confidence Spread")
    
    if agreement_rate >= 0.55:
        print("  ✅ Shadow Correlation")
        gates_passed += 1
    else:
        print("  ❌ Shadow Correlation")
    
    if monotonic:
        print("  ✅ Calibration")
        gates_passed += 1
    else:
        print("  ⚠️  Calibration")
    
    print(f"\n  TOTAL: {gates_passed} / 4 gates passed")
    
    if gates_passed >= 3:
        print(f"\n  ✅ READY FOR ACTIVATION (≥3/4 gates passed)")
    else:
        print(f"\n  ❌ NOT READY (need ≥3/4 gates)")
    
    print(f"{'='*70}")

def main():
    parser = argparse.ArgumentParser(description='Analyze PatchTST shadow mode metrics')
    parser.add_argument('--hours', type=int, default=2, help='Hours of logs to analyze (default: 2)')
    parser.add_argument('--input', type=str, help='Input log file (if not using journalctl)')
    args = parser.parse_args()
    
    print(f"\nFetching logs from last {args.hours} hours...")
    
    if args.input:
        # Read from file
        with open(args.input, 'r') as f:
            log_lines = f.readlines()
    else:
        # Fetch from journalctl via stdin or direct call
        import subprocess
        try:
            result = subprocess.run(
                ['journalctl', '-u', 'quantum-ai-engine.service', '--since', f'{args.hours} hours ago'],
                capture_output=True,
                text=True,
                timeout=30
            )
            log_lines = result.stdout.split('\n')
        except Exception as e:
            print(f"Error fetching logs: {e}")
            print("Usage: python3 analyze_shadow_metrics.py --input <logfile>")
            sys.exit(1)
    
    print(f"Processing {len(log_lines)} log lines...\n")
    
    # Parse and analyze
    events = parse_trade_intent_logs(log_lines)
    analyze_shadow_predictions(events)

if __name__ == '__main__':
    main()
