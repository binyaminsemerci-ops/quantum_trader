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
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# PATCH CUTOVER: AI engine restart after hardcoded confidence removal
PATCH_CUTOVER_TS = "2026-01-10T05:43:15Z"  # 1736486595 Unix timestamp


def sigmoid(x):
    """Compute sigmoid for logit normalization"""
    return 1.0 / (1.0 + np.exp(-x))


def normalize_confidence(raw_value, model_name="unknown"):
    """
    Normalize confidence to [0, 1] range per QSC telemetry contract.
    
    Rules:
    - If value in [0, 1]: Return as-is (already probability)
    - If value > 1: Treat as logit/score, apply sigmoid
    - If value < 0 or NaN: BLOCKER (invalid confidence)
    
    Returns:
        dict: {
            'normalized_prob': float in [0, 1],
            'raw_value': original value,
            'normalization_applied': bool,
            'violation': str or None (BLOCKER if out-of-range)
        }
    """
    try:
        val = float(raw_value)
        
        # Check for invalid values (BLOCKER)
        if np.isnan(val) or np.isinf(val):
            return {
                'normalized_prob': 0.5,  # Fallback for reporting
                'raw_value': raw_value,
                'normalization_applied': False,
                'violation': f"BLOCKER: Invalid confidence (NaN/Inf) from {model_name}"
            }
        
        if val < 0:
            return {
                'normalized_prob': 0.0,
                'raw_value': val,
                'normalization_applied': False,
                'violation': f"BLOCKER: Negative confidence {val} from {model_name}"
            }
        
        # Valid probability range [0, 1]
        if 0 <= val <= 1:
            return {
                'normalized_prob': val,
                'raw_value': val,
                'normalization_applied': False,
                'violation': None
            }
        
        # Logit/score (>1): Apply sigmoid
        normalized = sigmoid(val)
        return {
            'normalized_prob': normalized,
            'raw_value': val,
            'normalization_applied': True,
            'violation': None  # Not a blocker, just needs normalization
        }
    
    except (TypeError, ValueError) as e:
        return {
            'normalized_prob': 0.5,
            'raw_value': raw_value,
            'normalization_applied': False,
            'violation': f"BLOCKER: Cannot parse confidence '{raw_value}' from {model_name}: {e}"
        }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Quality Gate - Telemetry-Only Model Safety Check')
    parser.add_argument(
        '--after',
        type=str,
        default=None,
        help=f'Analyze only events after timestamp (ISO 8601 format). Use {PATCH_CUTOVER_TS} for post-patch analysis'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['collection', 'canary'],
        default='canary',
        help='Quality gate mode: collection (min_events=100, exit 3) or canary (min_events=200, can exit 0)'
    )
    return parser.parse_args()


def timestamp_to_stream_id(ts_str):
    """Convert ISO 8601 timestamp to Redis stream ID (milliseconds-sequenceNumber)"""
    from datetime import datetime
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    unix_ms = int(dt.timestamp() * 1000)
    return f"{unix_ms}-0"  # Use sequence 0 as minimum


def read_redis_stream(stream_key='quantum:stream:trade.intent', count=2000, after_ts=None):
    """
    Read last N events from Redis stream using Python redis client
    
    SYSTEMD-ONLY: Assumes Redis on localhost:6379
    
    Args:
        stream_key: Redis stream name
        count: Maximum events to read
        after_ts: If provided (ISO 8601), filter events after this timestamp
    
    Returns:
        list of events (filtered by timestamp if after_ts provided)
    """
    try:
        # Connect to Redis (localhost only)
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        
        # Test connection
        r.ping()
        
        if after_ts:
            # Read all events after cutover timestamp
            # NOTE: Python redis client has a bug where xrange with large count returns incomplete results
            # Workaround: Use redis-cli XRANGE directly
            import subprocess
            min_stream_id = timestamp_to_stream_id(after_ts)
            
            # Use redis-cli to get ALL events after cutover
            cmd = f"redis-cli XRANGE {stream_key} {min_stream_id} +"
            result = subprocess.check_output(cmd, shell=True, text=True)
            
            # Parse redis-cli output: ID, field1, value1, field2, value2, ...
            # Fields can have empty values (e.g., trace_id is often empty)
            lines = result.strip().split('\n')
            raw_events = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                # Check if line is an event ID (format: 1234567890-0)
                if '-' in line and line.replace('-', '').isdigit():
                    event_id = line
                    fields = {}
                    i += 1
                    # Read field-value pairs until next ID or EOF
                    while i < len(lines):
                        next_line = lines[i].strip()
                        # Stop if next line is an ID
                        if '-' in next_line and next_line.replace('-', '').isdigit():
                            break
                        # Read key
                        key = lines[i]
                        i += 1
                        # Read value (may be empty string)
                        value = lines[i] if i < len(lines) else ""
                        i += 1
                        fields[key] = value
                    raw_events.append((event_id, fields))
                else:
                    i += 1
            
            # Limit to requested count and reverse to newest-first
            raw_events = list(reversed(raw_events[:count]))
        else:
            # XREVRANGE returns events in reverse order (newest first)
            raw_events = r.xrevrange(stream_key, count=count)
        
        # Convert bytes to strings and parse
        events = []
        for event_id, fields in raw_events:
            # Handle both string (redis-cli) and bytes (xrevrange) formats
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
    Extract per-model predictions from events.
    Normalizes confidence values to [0, 1] per QSC contract.
    
    Stream format:
    - fields['event_type'] = 'trade.intent'
    - fields['payload'] = JSON string with model_breakdown
    
    Returns: dict {model_name: {
        'predictions': [...],
        'normalization_stats': {...}
    }}
    """
    model_data = defaultdict(lambda: {'predictions': [], 'normalization_stats': {'count': 0, 'normalized': 0, 'violations': []}})
    
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
        
        # Extract per-model predictions with normalization
        for model_name, model_info in breakdown.items():
            if isinstance(model_info, dict):
                action = model_info.get('action')
                raw_confidence = model_info.get('confidence')
                
                if action and raw_confidence is not None:
                    # Normalize confidence to [0, 1]
                    norm_result = normalize_confidence(raw_confidence, model_name)
                    
                    # Track stats
                    model_data[model_name]['normalization_stats']['count'] += 1
                    if norm_result['normalization_applied']:
                        model_data[model_name]['normalization_stats']['normalized'] += 1
                    if norm_result['violation']:
                        model_data[model_name]['normalization_stats']['violations'].append(norm_result['violation'])
                    
                    model_data[model_name]['predictions'].append({
                        'action': action,
                        'confidence': norm_result['normalized_prob'],  # Use normalized [0, 1]
                        'raw_confidence': norm_result['raw_value'],
                        'normalization_applied': norm_result['normalization_applied'],
                        'violation': norm_result['violation']
                    })
    
    return dict(model_data)


def analyze_predictions(predictions):
    """
    Calculate action distribution and confidence stats.
    Collects confidence violations for BLOCKER reporting.
    """
    if not predictions:
        return None
    
    # Extract actions and confidences (already normalized)
    actions = [p['action'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]  # Already normalized [0, 1]
    
    # Collect violations
    violations = [p['violation'] for p in predictions if p.get('violation')]
    normalized_count = sum(1 for p in predictions if p.get('normalization_applied'))
    
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
        'sample_count': total,
        'confidence_violations': violations,
        'normalized_count': normalized_count
    }


def check_quality_gate(analysis):
    """
    HARD CHECKS - FAIL IF ANY VIOLATION
    
    Returns: list of failure reasons (empty = PASS)
    """
    if not analysis:
        return ['No data to analyze']
    
    failures = []
    
    # Check 0: Confidence violations (BLOCKER)
    violations = analysis.get('confidence_violations', [])
    if violations:
        failures.append(f"CONFIDENCE VIOLATIONS ({len(violations)} found)")
        for violation in violations[:5]:  # Show first 5
            failures.append(f"  - {violation}")
        if len(violations) > 5:
            failures.append(f"  - ... and {len(violations) - 5} more violations")
        return failures  # BLOCKER: Don't check other thresholds
    
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


def generate_report(model_results, telemetry_info, report_path, pre_results=None):
    """Generate markdown report with optional pre/post comparison"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    lines = [
        "# Quality Gate Report (Telemetry-Only)",
        "",
        f"**Timestamp:** {timestamp}",        "",
        f"**MODE:** {'🔒 COLLECTION (DATA GATHERING ONLY)' if telemetry_info.get('mode') == 'collection' else '🚀 CANARY (DEPLOYMENT ELIGIBLE)'}",        ""
    ]
    
    # Cutover info
    if telemetry_info.get('cutover_ts'):
        lines.append("## Cutover Analysis")
        lines.append("")
        lines.append(f"**Cutover Timestamp:** {telemetry_info['cutover_ts']}")
        lines.append(f"**Mode:** Post-cutover analysis (events after patch deployment)")
        lines.append("")
    
    # Confidence normalization audit
    if telemetry_info.get('normalization_summary'):
        lines.append("## Confidence Normalization Audit")
        lines.append("")
        summary = telemetry_info['normalization_summary']
        lines.append(f"**Total predictions:** {summary['total_predictions']}")
        lines.append(f"**Normalized (logit → prob):** {summary['normalized_count']} ({summary['normalized_pct']:.1f}%)")
        lines.append(f"**Violations (BLOCKER):** {summary['violation_count']}")
        lines.append("")
        
        if summary['violation_count'] > 0:
            lines.append("### ⚠️ CONFIDENCE VIOLATIONS")
            lines.append("")
            for violation in summary['violations'][:10]:  # Show first 10
                lines.append(f"- {violation}")
            if summary['violation_count'] > 10:
                lines.append(f"- ... and {summary['violation_count'] - 10} more violations")
            lines.append("")
        
        if summary['normalized_count'] > 0:
            lines.append("**Normalization applied:**")
            lines.append("- Values >1.0 treated as logits")
            lines.append("- Sigmoid applied: prob = 1 / (1 + exp(-logit))")
            lines.append("- Quality gate uses normalized [0, 1] range only")
            lines.append("")
    
    lines.extend([
        "## Telemetry Info",
        f"- Redis stream: {telemetry_info['stream_key']}",
        f"- Events analyzed: {telemetry_info['event_count']}",
        f"- Events requested: {telemetry_info['event_requested']}",
        ""
    ])
    
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
        
        # Add delta comparison if pre_results provided
        if pre_results:
            lines.append("### Pre/Post Cutover Comparison")
            lines.append("")
            lines.append("| Model | Metric | Before Patch | After Patch | Delta |")
            lines.append("|-------|--------|--------------|-------------|-------|")
            
            for model_name in sorted(model_results.keys()):
                post = model_results[model_name]['analysis']
                pre = pre_results.get(model_name, {}).get('analysis')
                
                if post and pre:
                    # HOLD% delta
                    hold_pre = pre['action_pcts']['HOLD']
                    hold_post = post['action_pcts']['HOLD']
                    hold_delta = hold_post - hold_pre
                    lines.append(f"| {model_name} | HOLD% | {hold_pre:.1f}% | {hold_post:.1f}% | {hold_delta:+.1f}% |")
                    
                    # Confidence std delta
                    std_pre = pre['confidence']['std']
                    std_post = post['confidence']['std']
                    std_delta = std_post - std_pre
                    lines.append(f"| {model_name} | Conf Std | {std_pre:.4f} | {std_post:.4f} | {std_delta:+.4f} |")
                    
                    # P10-P90 range delta
                    range_pre = pre['confidence']['p10_p90_range']
                    range_post = post['confidence']['p10_p90_range']
                    range_delta = range_post - range_pre
                    lines.append(f"| {model_name} | P10-P90 | {range_pre:.4f} | {range_post:.4f} | {range_delta:+.4f} |")
            
            lines.append("")
            lines.append("**Improvement indicators:**")
            lines.append("- HOLD% decrease = Less dead zone trap ✅")
            lines.append("- Conf Std increase = More variance ✅")
            lines.append("- P10-P90 increase = Wider distribution ✅")
            lines.append("")
        
        for model_name, result in sorted(model_results.items()):
            analysis = result['analysis']
            failures = result['failures']
            
            status = "✅ PASS" if not failures else "❌ FAIL"
            
            lines.append(f"### {model_name} - {status}")
            lines.append("")
            
            if analysis:
                # Normalization info
                if analysis.get('normalized_count', 0) > 0:
                    norm_pct = (analysis['normalized_count'] / analysis['sample_count']) * 100
                    lines.append(f"**Normalization:** {analysis['normalized_count']}/{analysis['sample_count']} predictions ({norm_pct:.1f}%) normalized from logits")
                    lines.append("")
                
                # Violations
                if analysis.get('confidence_violations'):
                    lines.append(f"**⚠️ VIOLATIONS:** {len(analysis['confidence_violations'])} confidence errors")
                    for violation in analysis['confidence_violations'][:3]:
                        lines.append(f"  - {violation}")
                    if len(analysis['confidence_violations']) > 3:
                        lines.append(f"  - ... and {len(analysis['confidence_violations']) - 3} more")
                    lines.append("")
                
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
    
    # Parse arguments
    args = parse_args()
    
    STREAM_KEY = 'quantum:stream:trade.intent'
    EVENT_COUNT = 2000
    
    # 🔒 FAIL-CLOSED: Collection vs Canary modes
    # collection: Lower threshold for data gathering, exit 3 (never promotes)
    # canary: Full threshold for deployment, exit 0 only if safe
    if args.mode == 'collection':
        MIN_EVENTS = 100
        print(f"\n{'='*80}")
        print("📦 COLLECTION MODE: min_events={MIN_EVENTS}, will exit 3 (NO PROMOTION)")
        print(f"{'='*80}\n")
    else:  # canary
        MIN_EVENTS = 200
        print(f"\n{'='*80}")
        print("🚀 CANARY MODE: min_events={MIN_EVENTS}, exit 0 enables promotion")
        print(f"{'='*80}\n")
    
    # Setup paths
    reports_dir = Path('reports/safety')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode_suffix = f'_{args.mode}' if args.mode == 'collection' else ''
    report_suffix = '_post_cutover' if args.after else ''
    report_path = reports_dir / f'quality_gate_{timestamp_str}{mode_suffix}{report_suffix}.md'
    
    print("="*80)
    print("QUALITY GATE - TELEMETRY-ONLY (FAIL-CLOSED)")
    print("="*80)
    print()
    print(f"Reading Redis stream: {STREAM_KEY}")
    print(f"Requested events: {EVENT_COUNT}")
    if args.after:
        print(f"Cutover filter: Events after {args.after}")
    print()
    
    try:
        # If cutover analysis, get pre-cutover data first for comparison
        pre_results = None
        if args.after:
            print("Step 1: Analyzing pre-cutover data for comparison...")
            pre_events = read_redis_stream(STREAM_KEY, EVENT_COUNT, after_ts=None)
            pre_model_data = extract_model_predictions(pre_events)
            pre_results = {}
            for model_name, model_info in pre_model_data.items():
                predictions = model_info['predictions']
                analysis = analyze_predictions(predictions)
                failures = check_quality_gate(analysis)
                pre_results[model_name] = {'analysis': analysis, 'failures': failures}
            print(f"   Found {len(pre_events)} pre-cutover events")
            print()
        
        # Read Redis stream (with cutover filter if specified)
        print(f"Step {'2' if args.after else '1'}: Analyzing {'post-cutover' if args.after else 'all'} data...")
        events = read_redis_stream(STREAM_KEY, EVENT_COUNT, after_ts=args.after)
        
        telemetry_info = {
            'stream_key': STREAM_KEY,
            'event_count': len(events),
            'event_requested': EVENT_COUNT,
            'min_events': MIN_EVENTS,
            'cutover_ts': args.after,
            'mode': args.mode  # 🔒 PASS MODE TO REPORT
        }
        
        print(f"Parsing events...")
        print(f"Found {len(events)} events")
        print()
        
        # Extract model predictions (with normalization)
        print("Extracting model predictions...")
        model_data = extract_model_predictions(events)
        
        # Calculate normalization summary
        total_preds = sum(len(model_info['predictions']) for model_info in model_data.values())
        total_normalized = 0
        all_violations = []
        
        for model_name, model_info in model_data.items():
            predictions = model_info['predictions']
            total_normalized += sum(1 for p in predictions if p.get('normalization_applied'))
            all_violations.extend([p['violation'] for p in predictions if p.get('violation')])
        
        telemetry_info['normalization_summary'] = {
            'total_predictions': total_preds,
            'normalized_count': total_normalized,
            'normalized_pct': (total_normalized / total_preds * 100) if total_preds > 0 else 0,
            'violation_count': len(all_violations),
            'violations': all_violations
        }
        
        print(f"✅ Extracted {total_preds} predictions from {len(model_data)} models")
        if total_normalized > 0:
            print(f"   📊 Normalized {total_normalized} logits → probabilities ({telemetry_info['normalization_summary']['normalized_pct']:.1f}%)")
        if len(all_violations) > 0:
            print(f"   ⚠️  Found {len(all_violations)} confidence violations (BLOCKER)")
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
        
        # Check for model data
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
        
        for model_name, model_info in model_data.items():
            predictions = model_info['predictions']
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
        generate_report(model_results, telemetry_info, report_path, pre_results=pre_results)
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
            # 🔒 FAIL-CLOSED: Collection mode NEVER promotes (exit 3)
            if args.mode == 'collection':
                print("📦 QUALITY GATE: COLLECTION COMPLETE")
                print("="*80)
                print()
                print("Collection mode: Data gathered, but CANNOT PROMOTE")
                print("Rerun with --mode canary and >=200 events to enable deployment")
                return 3  # Special exit code: collection only, no promotion
            
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
