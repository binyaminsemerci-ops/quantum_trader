#!/usr/bin/env python3
"""
Quality Gate - TELEMETRY-ONLY (FAIL-CLOSED)

NEVER loads model files. Uses Redis stream telemetry only.

PRIMARY SOURCE: Redis stream quantum:stream:trade.intent
METHOD: redis-cli XREVRANGE (last N events, default 2000)

FAIL-CLOSED RULES:
- <200 events → FAIL (insufficient data = no activation)
- Missing model_breakdown → FAIL
- Any hard blocker → FAIL

DETECTS:
- Constant output (std<0.01 OR p10==p90)
- HOLD collapse (HOLD>85% AND confidence flat)
- Majority bias (any class >70%)
- Confidence collapse (std<0.05 OR p10-p90<0.12)

EXIT CODES:
  0 = PASS (all gates passed)
  2 = FAIL (BLOCKER - do NOT activate)
"""

import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def read_redis_stream(stream_key='quantum:stream:trade.intent', count=2000):
    """
    Read last N events from Redis stream using redis-cli
    
    SYSTEMD-ONLY: Assumes Redis on localhost:6379
    """
    try:
        # XREVRANGE returns events in reverse order (newest first)
        cmd = [
            'redis-cli',
            'XREVRANGE',
            stream_key,
            '+',  # max ID (newest)
            '-',  # min ID (oldest)
            'COUNT',
            str(count)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            raise RuntimeError(f"redis-cli failed: {result.stderr}")
        
        return result.stdout
    except Exception as e:
        raise RuntimeError(f"Failed to read Redis stream: {e}")


def parse_redis_output(output):
    """
    Parse redis-cli XREVRANGE output into list of events
    
    Format (nested array):
    1) "1736478123456-0"
    2) 1) "action"
       2) "BUY"
       3) "confidence"
       4) "0.75"
       5) "model_breakdown"
       6) "{...json...}"
    """
    lines = output.strip().split('\n')
    events = []
    
    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i].strip():
            i += 1
            continue
        
        # Event ID line (e.g., "1736478123456-0")
        if lines[i].startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
            # Extract event ID
            event_id_line = lines[i]
            event_id = event_id_line.split(')')[1].strip().strip('"')
            
            i += 1
            
            # Parse fields (key-value pairs)
            fields = {}
            while i < len(lines) and not lines[i].startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
                line = lines[i].strip()
                
                # Skip field number lines
                if line and not line.startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
                    # This is a value
                    if i > 0:
                        prev_line = lines[i-1].strip()
                        # Find the key (previous non-number line)
                        if prev_line and prev_line.split(')')[-1].strip().strip('"'):
                            key = prev_line.split(')')[-1].strip().strip('"')
                            value = line.strip('"')
                            fields[key] = value
                
                i += 1
            
            if fields:
                events.append({
                    'id': event_id,
                    'fields': fields
                })
        else:
            i += 1
    
    return events


def extract_model_predictions(events):
    """
    Extract per-model predictions from events
    
    Returns: {model_name: [predictions]}
    """
    model_data = defaultdict(list)
    
    for event in events:
        fields = event.get('fields', {})
        
        # Parse model_breakdown JSON
        breakdown_json = fields.get('model_breakdown', '{}')
        try:
            breakdown = json.loads(breakdown_json)
        except:
            continue
        
        # Extract per-model predictions
        for model_name, model_info in breakdown.items():
            if isinstance(model_info, dict):
                action = model_info.get('action')
                confidence = model_info.get('confidence')
                
                if action and confidence is not None:
                    model_data[model_name].append({
                        'action': action,
                        'confidence': float(confidence)
                    })
    
    return model_data


def analyze_predictions(predictions):
    """Analyze prediction distribution for one model"""
    if not predictions:
        return None
    
    actions = [p['action'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
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
    
    # Check 3: Constant output
    if analysis['confidence_stats']['std'] < 0.01:
        failures.append(f"Confidence std = {analysis['confidence_stats']['std']:.6f} < 0.01 (CONSTANT OUTPUT)")
    
    if analysis['confidence_stats']['p10'] == analysis['confidence_stats']['p90']:
        failures.append(f"Confidence P10 == P90 = {analysis['confidence_stats']['p10']:.4f} (FLATLINED)")
    
    # Check 4: HOLD collapse (>85% AND confidence flat/clustered)
    hold_pct = analysis['action_pcts']['HOLD']
    if hold_pct > 85:
        failures.append(f"HOLD = {hold_pct:.1f}% > 85% (HOLD COLLAPSE)")
    
    return failures


def generate_report(model_results, telemetry_info, report_path):
    """Generate markdown report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    # Overall status
    all_passed = all(not r['failures'] for r in model_results.values())
    status = "✅ PASS" if all_passed else "❌ FAIL (BLOCKER)"
    
    with open(report_path, 'w') as f:
        f.write(f"# Quality Gate Report (Telemetry-Only)\n\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Status:** {status}\n")
        f.write(f"**Source:** Redis stream `{telemetry_info['stream_key']}`\n")
        f.write(f"**Events analyzed:** {telemetry_info['events_count']}\n")
        f.write(f"**Events requested:** {telemetry_info['events_requested']}\n\n")
        
        if telemetry_info.get('insufficient_data'):
            f.write(f"## ⚠️ INSUFFICIENT DATA (FAIL-CLOSED)\n\n")
            f.write(f"Minimum required: 200 events\n")
            f.write(f"Found: {telemetry_info['events_count']}\n\n")
            f.write(f"**BLOCKER:** Cannot validate model safety without sufficient telemetry.\n\n")
        
        f.write(f"## Model Breakdown\n\n")
        
        for model_name, result in model_results.items():
            analysis = result['analysis']
            failures = result['failures']
            
            model_status = "✅ PASS" if not failures else "❌ FAIL"
            
            f.write(f"### {model_name} - {model_status}\n\n")
            
            if not analysis:
                f.write(f"**No predictions found in telemetry**\n\n")
                continue
            
            f.write(f"**Sample count:** {analysis['sample_count']}\n\n")
            
            f.write(f"#### Action Distribution\n\n")
            for action, count in analysis['action_counts'].items():
                pct = analysis['action_pcts'][action]
                f.write(f"- **{action}**: {count} ({pct:.1f}%)\n")
            
            f.write(f"\n#### Confidence Statistics\n\n")
            cs = analysis['confidence_stats']
            f.write(f"- Mean: {cs['mean']:.4f}\n")
            f.write(f"- Stddev: {cs['std']:.4f}\n")
            f.write(f"- P10: {cs['p10']:.4f}\n")
            f.write(f"- P50: {cs['p50']:.4f}\n")
            f.write(f"- P90: {cs['p90']:.4f}\n")
            f.write(f"- P10-P90 Range: {cs['p10_p90_range']:.4f}\n")
            f.write(f"- Unique values: {cs['unique_count']}\n")
            
            f.write(f"\n#### Quality Gate Checks\n\n")
            if failures:
                f.write(f"**FAILED ({len(failures)} violations):**\n\n")
                for fail in failures:
                    f.write(f"- ❌ {fail}\n")
            else:
                f.write(f"**ALL CHECKS PASSED**\n\n")
                f.write(f"- ✅ No class >70%\n")
                f.write(f"- ✅ Confidence std ≥0.05\n")
                f.write(f"- ✅ Confidence P10-P90 ≥0.12\n")
                f.write(f"- ✅ No constant output\n")
                f.write(f"- ✅ HOLD ≤85%\n")
            
            f.write(f"\n")


def main():
    STREAM_KEY = 'quantum:stream:trade.intent'
    EVENT_COUNT = 2000
    MIN_EVENTS = 200  # FAIL-CLOSED: Require minimum events
    
    report_dir = Path('reports/safety')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'quality_gate_{timestamp_str}.md'
    
    print(f"{'='*70}")
    print(f"QUALITY GATE - TELEMETRY-ONLY (FAIL-CLOSED)")
    print(f"{'='*70}\n")
    
    # Read Redis stream
    print(f"Reading Redis stream: {STREAM_KEY}")
    print(f"Requested events: {EVENT_COUNT}")
    
    try:
        redis_output = read_redis_stream(STREAM_KEY, EVENT_COUNT)
    except Exception as e:
        print(f"❌ FAILED TO READ REDIS STREAM: {e}")
        return 2
    
    # Parse events
    print(f"Parsing events...")
    events = parse_redis_output(redis_output)
    print(f"Found {len(events)} events\n")
    
    telemetry_info = {
        'stream_key': STREAM_KEY,
        'events_requested': EVENT_COUNT,
        'events_count': len(events),
        'insufficient_data': len(events) < MIN_EVENTS
    }
    
    # FAIL-CLOSED: Insufficient data
    if len(events) < MIN_EVENTS:
        print(f"❌ INSUFFICIENT DATA (FAIL-CLOSED)")
        print(f"   Minimum required: {MIN_EVENTS}")
        print(f"   Found: {len(events)}\n")
        
        # Generate report showing failure
        generate_report({}, telemetry_info, report_path)
        print(f"Report: {report_path}\n")
        
        print(f"{'='*70}")
        print(f"❌ QUALITY GATE: FAIL (BLOCKER)")
        print(f"{'='*70}")
        print(f"\nMissing data = NO ACTIVATION\n")
        
        return 2
    
    # Extract model predictions
    print(f"Extracting model predictions...")
    model_data = extract_model_predictions(events)
    
    if not model_data:
        print(f"❌ NO MODEL DATA FOUND IN TELEMETRY")
        generate_report({}, telemetry_info, report_path)
        print(f"Report: {report_path}\n")
        return 2
    
    print(f"Found predictions for {len(model_data)} models\n")
    
    # Analyze each model
    model_results = {}
    
    for model_name, predictions in model_data.items():
        print(f"Analyzing {model_name}... ({len(predictions)} predictions)")
        
        analysis = analyze_predictions(predictions)
        
        if not analysis:
            model_results[model_name] = {
                'analysis': None,
                'failures': ['No predictions found']
            }
            continue
        
        failures = check_quality_gate(analysis)
        
        model_results[model_name] = {
            'analysis': analysis,
            'failures': failures
        }
        
        if failures:
            print(f"  ❌ FAIL: {len(failures)} violations")
        else:
            print(f"  ✅ PASS")
    
    # Generate report
    generate_report(model_results, telemetry_info, report_path)
    
    # Overall result
    print(f"\n{'='*70}")
    print(f"QUALITY GATE RESULTS")
    print(f"{'='*70}\n")
    
    any_failures = any(r['failures'] for r in model_results.values())
    
    for model_name, result in model_results.items():
        failures = result['failures']
        analysis = result['analysis']
        
        if not analysis:
            print(f"{model_name}: ❌ FAIL (no data)")
        elif failures:
            print(f"{model_name}: ❌ FAIL ({len(failures)} violations)")
            for fail in failures:
                print(f"  - {fail}")
        else:
            print(f"{model_name}: ✅ PASS")
    
    print(f"\nReport: {report_path}")
    
    print(f"\n{'='*70}")
    if any_failures:
        print(f"❌ QUALITY GATE: FAIL (BLOCKER)")
        print(f"{'='*70}")
        return 2
    else:
        print(f"✅ QUALITY GATE: PASS")
        print(f"{'='*70}")
        return 0


if __name__ == '__main__':
    sys.exit(main())
