#!/usr/bin/env python3
"""
XGBoost Overconfidence Forensic Analysis
Analyzes model_breakdown from recent trade.intent events
"""

import json
import subprocess
from collections import defaultdict, Counter
from typing import Dict, List

def fetch_redis_events(count: int = 200) -> List[Dict]:
    """Fetch recent trade.intent events from Redis."""
    cmd = [
        'redis-cli', 'XREVRANGE', 
        'quantum:stream:trade.intent', '+', '-', 
        'COUNT', str(count)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to fetch Redis events: {result.stderr}")
        return []
    
    # Parse Redis stream output
    lines = result.stdout.strip().split('\n')
    events = []
    
    i = 0
    while i < len(lines):
        # Event ID (skip)
        if i + 1 >= len(lines):
            break
        i += 1
        
        # Find payload field
        while i < len(lines) and lines[i].strip() != 'payload':
            i += 1
        
        if i + 1 >= len(lines):
            break
        i += 1
        
        # Parse JSON payload
        payload_line = lines[i].strip()
        try:
            payload = json.loads(payload_line)
            if 'model_breakdown' in payload:
                events.append(payload)
        except json.JSONDecodeError:
            pass
        
        i += 1
    
    return events

def analyze_model_distribution(events: List[Dict]) -> Dict:
    """Analyze action and confidence distribution per model."""
    models = ['xgb', 'lgbm', 'nhits', 'patchtst']
    
    analysis = {}
    for model in models:
        actions = []
        confidences = []
        model_types = []
        
        for event in events:
            breakdown = event.get('model_breakdown', {})
            if model in breakdown:
                model_data = breakdown[model]
                actions.append(model_data.get('action', 'UNKNOWN'))
                conf = model_data.get('confidence', 0.5)
                confidences.append(conf)
                model_types.append(model_data.get('model', 'unknown'))
        
        if confidences:
            analysis[model] = {
                'count': len(confidences),
                'actions': Counter(actions),
                'conf_min': min(confidences),
                'conf_max': max(confidences),
                'conf_mean': sum(confidences) / len(confidences),
                'conf_bins': {
                    '0.50-0.60': sum(1 for c in confidences if 0.50 <= c < 0.60),
                    '0.60-0.70': sum(1 for c in confidences if 0.60 <= c < 0.70),
                    '0.70-0.80': sum(1 for c in confidences if 0.70 <= c < 0.80),
                    '0.80-0.90': sum(1 for c in confidences if 0.80 <= c < 0.90),
                    '0.90-1.00': sum(1 for c in confidences if 0.90 <= c <= 1.00),
                },
                'model_types': Counter(model_types),
            }
        else:
            analysis[model] = None
    
    return analysis

def print_report(analysis: Dict, total_events: int):
    """Print formatted analysis report."""
    print("=" * 80)
    print("XGBoost OVERCONFIDENCE FORENSIC ANALYSIS")
    print("=" * 80)
    print(f"\n📊 Analyzed {total_events} recent trade.intent events\n")
    
    models = ['xgb', 'lgbm', 'nhits', 'patchtst']
    model_names = {
        'xgb': 'XGBoost',
        'lgbm': 'LightGBM',
        'nhits': 'N-HiTS',
        'patchtst': 'PatchTST'
    }
    
    for model in models:
        data = analysis.get(model)
        if data is None:
            print(f"⚠️  {model_names[model]}: NO DATA")
            continue
        
        print(f"\n{'='*80}")
        print(f"📈 {model_names[model].upper()}")
        print(f"{'='*80}")
        print(f"Total Predictions: {data['count']}")
        
        # Action distribution
        print(f"\n🎯 Action Distribution:")
        for action, count in data['actions'].most_common():
            pct = (count / data['count']) * 100
            bar = '█' * int(pct / 2)
            print(f"  {action:5s}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Confidence stats
        print(f"\n📊 Confidence Statistics:")
        print(f"  Min:  {data['conf_min']:.4f}")
        print(f"  Mean: {data['conf_mean']:.4f}")
        print(f"  Max:  {data['conf_max']:.4f}")
        
        # Confidence distribution
        print(f"\n📈 Confidence Distribution:")
        for bin_range, count in sorted(data['conf_bins'].items()):
            if count > 0:
                pct = (count / data['count']) * 100
                bar = '█' * int(pct / 2)
                print(f"  {bin_range}: {count:4d} ({pct:5.1f}%) {bar}")
        
        # Model type (ML vs fallback)
        print(f"\n🔧 Model Type:")
        for mtype, count in data['model_types'].most_common():
            pct = (count / data['count']) * 100
            is_fallback = 'fallback' in mtype.lower()
            icon = '⚠️ ' if is_fallback else '✅ '
            print(f"  {icon}{mtype}: {count} ({pct:.1f}%)")
    
    # Comparison summary
    print(f"\n{'='*80}")
    print("🔍 DIAGNOSTIC SUMMARY")
    print(f"{'='*80}")
    
    xgb_data = analysis.get('xgb')
    if xgb_data:
        xgb_buy_pct = (xgb_data['actions'].get('BUY', 0) / xgb_data['count']) * 100
        xgb_high_conf = data['conf_bins'].get('0.90-1.00', 0)
        xgb_high_conf_pct = (xgb_high_conf / xgb_data['count']) * 100
        
        print(f"\n📌 XGBoost Metrics:")
        print(f"  • BUY rate: {xgb_buy_pct:.1f}%")
        print(f"  • High confidence (>0.90): {xgb_high_conf_pct:.1f}%")
        print(f"  • Mean confidence: {xgb_data['conf_mean']:.4f}")
        
        # Diagnosis
        print(f"\n🔬 Diagnosis:")
        if xgb_buy_pct > 80:
            print(f"  ⚠️  ISSUE: XGB heavily biased toward BUY ({xgb_buy_pct:.1f}%)")
            print(f"      → Possible causes: (A) training data imbalance, (B) feature drift")
        
        if xgb_high_conf_pct > 70:
            print(f"  ⚠️  ISSUE: XGB overconfident (>90% conf in {xgb_high_conf_pct:.1f}% of cases)")
            print(f"      → Possible causes: (A) confidence mapping bug, (B) calibration needed")
        
        if xgb_data['conf_min'] > 0.95:
            print(f"  🚨 CRITICAL: XGB NEVER predicts below 0.95 confidence")
            print(f"      → Likely cause: confidence = max(proba) always near 1.0")
            print(f"      → Check: Is predict_proba returning one-hot vectors?")
        
        # Compare with other models
        lgbm_data = analysis.get('lgbm')
        if lgbm_data:
            lgbm_buy_pct = (lgbm_data['actions'].get('BUY', 0) / lgbm_data['count']) * 100
            print(f"\n📊 Comparison with LightGBM:")
            print(f"  • XGB BUY rate:  {xgb_buy_pct:.1f}%")
            print(f"  • LGBM BUY rate: {lgbm_buy_pct:.1f}%")
            if abs(xgb_buy_pct - lgbm_buy_pct) > 30:
                print(f"  ⚠️  DIVERGENCE: {abs(xgb_buy_pct - lgbm_buy_pct):.1f}% difference")
                print(f"      → Suggests XGB-specific issue (not feature drift)")

def main():
    print("Fetching recent trade.intent events from Redis...")
    events = fetch_redis_events(200)
    
    if not events:
        print("❌ No events found. Check Redis connection.")
        return
    
    print(f"✅ Fetched {len(events)} events\n")
    
    analysis = analyze_model_distribution(events)
    print_report(analysis, len(events))
    
    # Save raw data for further analysis
    output_file = '/tmp/xgb_confidence_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"\n💾 Raw analysis saved to: {output_file}")

if __name__ == '__main__':
    main()
