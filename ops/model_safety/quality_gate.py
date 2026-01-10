#!/usr/bin/env python3
"""
Quality Gate - Telemetry-Only Model Safety Check (FAIL-CLOSED)

Uses Redis stream telemetry ONLY (NEVER loads model files).

HARD RULES:
- Source: Redis Python client (localhost:6379)
- Stream: quantum:stream:trade.intent
- Parse payload JSON → extract model_breakdown
- <200 events → FAIL (insufficient data)
- Majority class >70% → FAIL (collapse)
- Confidence std <0.05 → FAIL (flat)
- P10-P90 range <0.12 → FAIL (narrow)
- HOLD >85% → FAIL (dead zone)
- Constant output (std<0.01 OR p10==p90) → FAIL

Exit codes:
  0 = PASS (safe to proceed)
  2 = FAIL (BLOCKER - missing data or quality violation)

"Manglende bevis = ingen aktivering" (Missing data = no activation)
"""

import sys
import json
import redis
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def read_redis_stream(stream_key='quantum:stream:trade.intent', count=2000):
    """
    Read last N events from Redis stream using Python redis client
    
    SYSTEMD-ONLY: Assumes Redis on localhost:6379
    """
    try:
        # Connect to Redis (localhost only)
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        # Test connection
        r.ping()
        
        # XREVRANGE returns events in reverse order (newest first)
        # Returns: [(event_id, {field: value, ...}), ...]
        raw_events = r.xrevrange(stream_key, count=count)
        
        # Convert bytes to strings and parse
        events = []
        for event_id, fields in raw_events:
            event_id_str = event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id
            
            # Decode fields
            decoded_fields = {}
            for key, value in fields.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                value_str = value.decode('utf-8') if isinstance(value, bytes) else value
                decoded_fields[key_str] = value_str
            
            events.append({
                'id': event_id_str,
                'fields': decoded_fields
            })
        
        return events
    except Exception as e:
        raise RuntimeError(f"Failed to read Redis stream: {e}")


def extract_model_predictions(events):
    """
    Extract per-model predictions from events
    
    Stream format:
    - fields['event_type'] = 'trade.intent'
    - fields['payload'] = JSON string with model_breakdown
    """
    model_data = defaultdict(list)
    
    for event in events:
        fields = event.get('fields', {})
        
        # Check event type
        if fields.get('event_type') != 'trade.intent':
            continue
        
        # Parse payload JSON
        payload_json = fields.get('payload', '{}')
        try:
            payload = json.loads(payload_json)
        except:
            continue
        
        # Extract model_breakdown
        breakdown = payload.get('model_breakdown', {})
        
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
    """Calculate action distribution and confidence stats"""
    if not predictions:
        return None
    
    # Extract actions and confidences
    actions = [p['action'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    # Action distribution
    action_counts = {
        'BUY': actions.count('BUY'),
        'SELL': actions.count('SELL'),
        'HOLD': actions.count('HOLD')
    }
    total = len(actions)
    action_pcts = {k: v/total*100 for k, v in action_counts.items()}
    
    # Confidence stats
    conf_array = np.array(confidences)
    conf_mean = float(np.mean(conf_array))
    conf_std = float(np.std(conf_array))
    conf_p10 = float(np.percentile(conf_array, 10))
    conf_p90 = float(np.percentile(conf_array, 90))
    p10_p90_range = conf_p90 - conf_p10
    
    return {
        'action_counts': action_counts,
        'action_pcts': action_pcts,
        'confidence': {
            'mean': conf_mean,
            'std': conf_std,
            'p10': conf_p10,
            'p90': conf_p90,
            'p10_p90_range': p10_p90_range
        },
        'sample_count': total
    }


def check_quality_gate(analysis):
    """
    HARD CHECKS - FAIL IF ANY VIOLATION
    
    Returns: list of failure reasons (empty = PASS)
    """
    if not analysis:
        return ['No data to analyze']
    
    failures = []
    
    # Check 1: Majority bias (any class >70%)
    for action, pct in analysis['action_pcts'].items():
        if pct > 70:
            failures.append(f"{action} majority {pct:.1f}% (>70% threshold)")
    
    # Check 2: Confidence spread
    conf_stats = analysis['confidence']
    
    if conf_stats['std'] < 0.05:
        failures.append(f"Confidence std {conf_stats['std']:.4f} (<0.05 threshold - flat)")
    
    if conf_stats['p10_p90_range'] < 0.12:
        failures.append(f"P10-P90 range {conf_stats['p10_p90_range']:.4f} (<0.12 threshold - narrow)")
    
    # Check 3: Constant output
    if conf_stats['std'] < 0.01:
        failures.append(f"Constant output detected (std={conf_stats['std']:.4f})")
    
    if abs(conf_stats['p10'] - conf_stats['p90']) < 0.001:
        failures.append(f"Constant output detected (P10={conf_stats['p10']:.4f}, P90={conf_stats['p90']:.4f})")
    
    # Check 4: HOLD collapse
    if analysis['action_pcts']['HOLD'] > 85:
        failures.append(f"HOLD collapse {analysis['action_pcts']['HOLD']:.1f}% (>85% threshold)")
    
    return failures


def generate_report(model_results, telemetry_info, report_path):
    """Generate markdown report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    lines = [
        "# Quality Gate Report (Telemetry-Only)",
        "",
        f"**Timestamp:** {timestamp}",
        "",
        "## Telemetry Info",
        f"- Redis stream: {telemetry_info['stream_key']}",
        f"- Events analyzed: {telemetry_info['event_count']}",
        f"- Events requested: {telemetry_info['event_requested']}",
        ""
    ]
    
    # Check if insufficient data
    if telemetry_info['event_count'] < telemetry_info['min_events']:
        lines.append("## ⚠️ INSUFFICIENT DATA (FAIL-CLOSED)")
        lines.append("")
        lines.append(f"Minimum required: {telemetry_info['min_events']} events")
        lines.append(f"Found: {telemetry_info['event_count']}")
        lines.append("")
        lines.append("**BLOCKER:** Cannot validate model safety without sufficient telemetry.")
        lines.append("")
    
    # Per-model breakdown
    if model_results:
        lines.append("## Model Breakdown")
        lines.append("")
        
        for model_name, result in sorted(model_results.items()):
            analysis = result['analysis']
            failures = result['failures']
            
            status = "✅ PASS" if not failures else "❌ FAIL"
            
            lines.append(f"### {model_name} - {status}")
            lines.append("")
            
            if analysis:
                lines.append("**Action Distribution:**")
                for action in ['BUY', 'SELL', 'HOLD']:
                    pct = analysis['action_pcts'][action]
                    count = analysis['action_counts'][action]
                    lines.append(f"- {action}: {pct:.1f}% ({count}/{analysis['sample_count']})")
                
                lines.append("")
                lines.append("**Confidence Stats:**")
                conf = analysis['confidence']
                lines.append(f"- Mean: {conf['mean']:.4f}")
                lines.append(f"- Std: {conf['std']:.4f}")
                lines.append(f"- P10: {conf['p10']:.4f}")
                lines.append(f"- P90: {conf['p90']:.4f}")
                lines.append(f"- P10-P90 Range: {conf['p10_p90_range']:.4f}")
                
                lines.append("")
                lines.append("**Quality Checks:**")
                if failures:
                    for failure in failures:
                        lines.append(f"- ❌ {failure}")
                else:
                    lines.append("- ✅ All checks passed")
            else:
                lines.append("**No data**")
            
            lines.append("")
    else:
        lines.append("## Model Breakdown")
        lines.append("")
        lines.append("**No models found in telemetry**")
        lines.append("")
    
    # Write report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines))


def main():
    """Main quality gate entry point"""
    
    STREAM_KEY = 'quantum:stream:trade.intent'
    EVENT_COUNT = 2000
    MIN_EVENTS = 200
    
    # Setup paths
    reports_dir = Path('reports/safety')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f'quality_gate_{timestamp_str}.md'
    
    print("="*80)
    print("QUALITY GATE - TELEMETRY-ONLY (FAIL-CLOSED)")
    print("="*80)
    print()
    print(f"Reading Redis stream: {STREAM_KEY}")
    print(f"Requested events: {EVENT_COUNT}")
    
    try:
        # Read Redis stream
        events = read_redis_stream(STREAM_KEY, EVENT_COUNT)
        
        telemetry_info = {
            'stream_key': STREAM_KEY,
            'event_count': len(events),
            'event_requested': EVENT_COUNT,
            'min_events': MIN_EVENTS
        }
        
        print(f"Parsing events...")
        print(f"Found {len(events)} events")
        print()
        
        # FAIL-CLOSED: Check minimum data
        if len(events) < MIN_EVENTS:
            print(f"❌ INSUFFICIENT DATA (FAIL-CLOSED)")
            print(f"   Minimum required: {MIN_EVENTS}")
            print(f"   Found: {len(events)}")
            print()
            generate_report({}, telemetry_info, report_path)
            print(f"Report: {report_path}")
            print()
            print("="*80)
            print("❌ QUALITY GATE: FAIL (BLOCKER)")
            print("="*80)
            print()
            print("Missing data = NO ACTIVATION")
            return 2
        
        # Extract model predictions
        print(f"Extracting model predictions...")
        model_data = extract_model_predictions(events)
        
        if not model_data:
            print(f"❌ No model_breakdown found in events")
            generate_report({}, telemetry_info, report_path)
            print(f"Report: {report_path}")
            print()
            print("="*80)
            print("❌ QUALITY GATE: FAIL (BLOCKER)")
            print("="*80)
            return 2
        
        print(f"Found {len(model_data)} models")
        print()
        
        # Analyze per-model
        model_results = {}
        any_failures = False
        
        for model_name, predictions in model_data.items():
            print(f"Analyzing {model_name}...")
            analysis = analyze_predictions(predictions)
            failures = check_quality_gate(analysis)
            
            model_results[model_name] = {
                'analysis': analysis,
                'failures': failures
            }
            
            if failures:
                any_failures = True
                print(f"  ❌ FAIL: {len(failures)} violations")
                for failure in failures:
                    print(f"     - {failure}")
            else:
                print(f"  ✅ PASS")
            print()
        
        # Generate report
        generate_report(model_results, telemetry_info, report_path)
        print(f"Report: {report_path}")
        print()
        print("="*80)
        
        if any_failures:
            print("❌ QUALITY GATE: FAIL (BLOCKER)")
            print("="*80)
            print()
            print("Quality violations detected = NO ACTIVATION")
            return 2
        else:
            print("✅ QUALITY GATE: PASS")
            print("="*80)
            print()
            print("All models passed quality checks")
            print("Safe to proceed with canary activation (manual only)")
            return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == '__main__':
    sys.exit(main())
