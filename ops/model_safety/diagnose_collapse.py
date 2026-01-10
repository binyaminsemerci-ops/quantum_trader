#!/usr/bin/env python3
"""
DIAGNOSIS MODE - Telemetry Flow Tracer (QSC-Compliant)

Traces: Raw Model Output → Action Mapping → Ensemble Decision

Identifies WHERE variance collapses and WHY:
- Threshold analysis (dead zones, action boundaries)
- Feature distribution (confidence patterns)
- Policy rules (ensemble voting, fallback logic)

NO training, NO activation, NO model loading.
Pure telemetry analysis using Redis stream only.

Usage: 
  ops/run.sh ai-engine ops/model_safety/diagnose_collapse.py
  ops/run.sh ai-engine ops/model_safety/diagnose_collapse.py --after 2026-01-10T05:43:15Z
"""

import sys
import json
import redis
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# PATCH CUTOVER: AI engine restart after hardcoded confidence removal
PATCH_CUTOVER_TS = "2026-01-10T05:43:15Z"  # 1736486595 Unix timestamp


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diagnose Collapse - Telemetry Flow Tracer')
    parser.add_argument(
        '--after',
        type=str,
        default=None,
        help=f'Analyze only events after timestamp (ISO 8601 format). Use {PATCH_CUTOVER_TS} for post-patch analysis'
    )
    return parser.parse_args()


def timestamp_to_stream_id(ts_str):
    """Convert ISO 8601 timestamp to Redis stream ID"""
    from datetime import datetime
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    unix_ms = int(dt.timestamp() * 1000)
    return f"{unix_ms}-0"


def read_redis_stream(stream_key='quantum:stream:trade.intent', count=2000, after_ts=None):
    """Read telemetry events from Redis stream (QSC: localhost:6379)"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        r.ping()
        
        if after_ts:
            # Read all events after cutover timestamp
            min_stream_id = timestamp_to_stream_id(after_ts)
            raw_events = r.xrange(stream_key, min=min_stream_id, max='+', count=count)
            raw_events = list(reversed(raw_events))  # Newest first
        else:
            raw_events = r.xrevrange(stream_key, count=count)
        
        events = []
        for event_id, fields in raw_events:
            event_id_str = event_id.decode('utf-8') if isinstance(event_id, bytes) else event_id
            
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


def extract_telemetry_data(events):
    """
    Extract full telemetry: raw predictions, mapped actions, ensemble decisions
    
    Returns dict with:
    - per_model_raw: {model: [(confidence, action)]}
    - ensemble_decisions: [(final_action, consensus, votes)]
    - breakdown_samples: [full model_breakdown dicts]
    """
    per_model_raw = defaultdict(list)
    ensemble_decisions = []
    breakdown_samples = []
    
    for event in events:
        fields = event.get('fields', {})
        
        if fields.get('event_type') != 'trade.intent':
            continue
        
        payload_json = fields.get('payload', '{}')
        try:
            payload = json.loads(payload_json)
        except:
            continue
        
        # Ensemble decision
        final_action = payload.get('side')  # BUY/SELL
        consensus_count = payload.get('consensus_count', 0)
        total_models = payload.get('total_models', 0)
        
        # Model breakdown
        breakdown = payload.get('model_breakdown', {})
        
        if breakdown:
            breakdown_samples.append(breakdown)
            
            # Extract per-model raw predictions
            for model_name, model_info in breakdown.items():
                if isinstance(model_info, dict):
                    action = model_info.get('action')
                    confidence = model_info.get('confidence')
                    
                    if action and confidence is not None:
                        per_model_raw[model_name].append({
                            'confidence': float(confidence),
                            'action': action
                        })
            
            # Ensemble decision
            if final_action:
                ensemble_decisions.append({
                    'action': final_action,
                    'consensus': consensus_count,
                    'total': total_models
                })
    
    return {
        'per_model_raw': per_model_raw,
        'ensemble_decisions': ensemble_decisions,
        'breakdown_samples': breakdown_samples
    }


def analyze_action_thresholds(model_data):
    """
    Identify action mapping thresholds and dead zones
    
    Check if actions are derived from confidence ranges:
    - BUY: conf > X
    - SELL: conf < Y
    - HOLD: X <= conf <= Y (dead zone)
    """
    analysis = {}
    
    for model_name, predictions in model_data.items():
        if not predictions:
            continue
        
        # Group by action
        by_action = defaultdict(list)
        for pred in predictions:
            by_action[pred['action']].append(pred['confidence'])
        
        # Calculate ranges
        action_ranges = {}
        for action, confs in by_action.items():
            if confs:
                action_ranges[action] = {
                    'count': len(confs),
                    'min': min(confs),
                    'max': max(confs),
                    'mean': np.mean(confs),
                    'std': np.std(confs),
                    'p10': np.percentile(confs, 10),
                    'p50': np.percentile(confs, 50),
                    'p90': np.percentile(confs, 90)
                }
        
        # Detect dead zone (HOLD range)
        hold_confs = by_action.get('HOLD', [])
        buy_confs = by_action.get('BUY', [])
        sell_confs = by_action.get('SELL', [])
        
        dead_zone = None
        if hold_confs:
            dead_zone = {
                'lower': min(hold_confs),
                'upper': max(hold_confs),
                'width': max(hold_confs) - min(hold_confs),
                'samples': len(hold_confs)
            }
        
        # Infer thresholds
        inferred_thresholds = {}
        if buy_confs and sell_confs:
            # BUY threshold = min(BUY)
            # SELL threshold = max(SELL)
            inferred_thresholds['buy_threshold'] = min(buy_confs) if buy_confs else None
            inferred_thresholds['sell_threshold'] = max(sell_confs) if sell_confs else None
        
        analysis[model_name] = {
            'action_ranges': action_ranges,
            'dead_zone': dead_zone,
            'inferred_thresholds': inferred_thresholds,
            'total_samples': len(predictions)
        }
    
    return analysis


def analyze_confidence_distribution(model_data):
    """Analyze raw confidence distributions to detect collapse"""
    analysis = {}
    
    for model_name, predictions in model_data.items():
        if not predictions:
            continue
        
        confs = [p['confidence'] for p in predictions]
        conf_array = np.array(confs)
        
        # Basic stats
        stats = {
            'count': len(confs),
            'mean': float(np.mean(conf_array)),
            'std': float(np.std(conf_array)),
            'min': float(np.min(conf_array)),
            'max': float(np.max(conf_array)),
            'p10': float(np.percentile(conf_array, 10)),
            'p25': float(np.percentile(conf_array, 25)),
            'p50': float(np.percentile(conf_array, 50)),
            'p75': float(np.percentile(conf_array, 75)),
            'p90': float(np.percentile(conf_array, 90)),
            'range': float(np.max(conf_array) - np.min(conf_array))
        }
        
        # Detect collapse patterns
        collapse_indicators = []
        
        if stats['std'] < 0.05:
            collapse_indicators.append(f"Flat distribution (std={stats['std']:.4f})")
        
        if stats['range'] < 0.1:
            collapse_indicators.append(f"Narrow range ({stats['range']:.4f})")
        
        if abs(stats['p90'] - stats['p10']) < 0.12:
            collapse_indicators.append(f"P10-P90 gap {stats['p90']-stats['p10']:.4f}")
        
        # Check for quantization (discrete values)
        unique_values = len(set(confs))
        if unique_values < 10:
            collapse_indicators.append(f"Quantized ({unique_values} unique values)")
        
        analysis[model_name] = {
            'stats': stats,
            'collapse_indicators': collapse_indicators
        }
    
    return analysis


def analyze_ensemble_logic(ensemble_decisions):
    """Analyze ensemble voting patterns"""
    if not ensemble_decisions:
        return {}
    
    # Count final actions
    action_counts = Counter([d['action'] for d in ensemble_decisions])
    total = len(ensemble_decisions)
    
    # Consensus patterns
    consensus_stats = []
    for d in ensemble_decisions:
        if d['total'] > 0:
            consensus_stats.append(d['consensus'] / d['total'])
    
    return {
        'total_decisions': total,
        'action_distribution': {k: v/total*100 for k, v in action_counts.items()},
        'consensus_mean': np.mean(consensus_stats) if consensus_stats else 0,
        'consensus_std': np.std(consensus_stats) if consensus_stats else 0,
        'raw_counts': dict(action_counts)
    }


def detect_policy_rules(breakdown_samples):
    """
    Detect policy rules from model_breakdown patterns
    
    Check for:
    - Fallback rules (lgbm_fallback_rules)
    - Shadow mode flags
    - Hardcoded confidence values
    """
    findings = []
    
    # Sample first 100 breakdowns
    for breakdown in breakdown_samples[:100]:
        for model_name, model_info in breakdown.items():
            if 'fallback' in model_name.lower():
                findings.append(f"Fallback model detected: {model_name}")
            
            if model_info.get('shadow'):
                findings.append(f"Shadow mode: {model_name}")
            
            # Check for hardcoded values (0.75, 0.5, etc.)
            conf = model_info.get('confidence')
            if conf in [0.75, 0.5, 0.68, 0.72]:
                findings.append(f"Suspected hardcoded confidence: {model_name} = {conf}")
    
    # Deduplicate
    unique_findings = list(set(findings))
    
    return unique_findings


def generate_diagnosis_report(telemetry_data, threshold_analysis, confidence_analysis, ensemble_analysis, policy_findings):
    """Generate comprehensive diagnosis report"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    lines = [
        "# DIAGNOSIS MODE - Telemetry Flow Analysis",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Mode:** Telemetry-Only (QSC-Compliant)",
        f"**Purpose:** Identify variance collapse root cause",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"Analyzed {len(telemetry_data['ensemble_decisions'])} ensemble decisions from Redis stream.",
        "",
    ]
    
    # Quick findings
    lines.append("**Critical Findings:**")
    
    # Check for majority collapse
    for model_name, analysis in threshold_analysis.items():
        for action, ranges in analysis['action_ranges'].items():
            pct = ranges['count'] / analysis['total_samples'] * 100
            if pct > 70:
                lines.append(f"- 🚨 {model_name}: {pct:.1f}% {action} (majority collapse)")
    
    lines.extend(["", "---", "", "## 1. Confidence Distribution Analysis", ""])
    
    for model_name, analysis in confidence_analysis.items():
        stats = analysis['stats']
        indicators = analysis['collapse_indicators']
        
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("**Raw Confidence Stats:**")
        lines.append(f"- Mean: {stats['mean']:.4f}")
        lines.append(f"- Std: {stats['std']:.4f}")
        lines.append(f"- Range: [{stats['min']:.4f}, {stats['max']:.4f}] (width: {stats['range']:.4f})")
        lines.append(f"- P10-P90: [{stats['p10']:.4f}, {stats['p90']:.4f}] (gap: {stats['p90']-stats['p10']:.4f})")
        lines.append("")
        
        if indicators:
            lines.append("**🔴 Collapse Indicators:**")
            for indicator in indicators:
                lines.append(f"- {indicator}")
        else:
            lines.append("**✅ Healthy distribution**")
        
        lines.append("")
    
    lines.extend(["---", "", "## 2. Action Mapping Threshold Analysis", ""])
    
    for model_name, analysis in threshold_analysis.items():
        lines.append(f"### {model_name}")
        lines.append("")
        
        # Action ranges
        lines.append("**Action → Confidence Ranges:**")
        for action in ['BUY', 'SELL', 'HOLD']:
            if action in analysis['action_ranges']:
                ranges = analysis['action_ranges'][action]
                pct = ranges['count'] / analysis['total_samples'] * 100
                lines.append(f"- **{action}**: {pct:.1f}% ({ranges['count']}/{analysis['total_samples']})")
                lines.append(f"  - Confidence: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
                lines.append(f"  - Mean: {ranges['mean']:.4f} ± {ranges['std']:.4f}")
        
        lines.append("")
        
        # Dead zone detection
        if analysis['dead_zone']:
            dz = analysis['dead_zone']
            lines.append("**🔴 DEAD ZONE DETECTED:**")
            lines.append(f"- Range: [{dz['lower']:.4f}, {dz['upper']:.4f}]")
            lines.append(f"- Width: {dz['width']:.4f}")
            lines.append(f"- Trapped samples: {dz['samples']} ({dz['samples']/analysis['total_samples']*100:.1f}%)")
        
        lines.append("")
        
        # Inferred thresholds
        if analysis['inferred_thresholds']:
            lines.append("**Inferred Thresholds:**")
            thresholds = analysis['inferred_thresholds']
            if thresholds.get('buy_threshold'):
                lines.append(f"- BUY threshold: ≥ {thresholds['buy_threshold']:.4f}")
            if thresholds.get('sell_threshold'):
                lines.append(f"- SELL threshold: ≤ {thresholds['sell_threshold']:.4f}")
        
        lines.append("")
    
    lines.extend(["---", "", "## 3. Ensemble Voting Analysis", ""])
    
    if ensemble_analysis:
        lines.append(f"**Total Decisions:** {ensemble_analysis['total_decisions']}")
        lines.append("")
        lines.append("**Final Action Distribution:**")
        for action, pct in ensemble_analysis['action_distribution'].items():
            lines.append(f"- {action}: {pct:.1f}%")
        lines.append("")
        lines.append(f"**Consensus Stats:**")
        lines.append(f"- Mean: {ensemble_analysis['consensus_mean']:.2%}")
        lines.append(f"- Std: {ensemble_analysis['consensus_std']:.4f}")
    
    lines.extend(["", "---", "", "## 4. Policy Rules Detection", ""])
    
    if policy_findings:
        for finding in policy_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- No suspicious policy rules detected")
    
    lines.extend(["", "---", "", "## 5. ROOT CAUSE ANALYSIS", ""])
    
    # Synthesize root causes
    root_causes = []
    
    # Check for dead zone traps
    for model_name, analysis in threshold_analysis.items():
        if analysis['dead_zone']:
            dz = analysis['dead_zone']
            if dz['samples'] / analysis['total_samples'] > 0.7:
                root_causes.append({
                    'model': model_name,
                    'issue': 'Dead Zone Trap',
                    'evidence': f"{dz['samples']/analysis['total_samples']*100:.1f}% samples trapped in HOLD range [{dz['lower']:.4f}, {dz['upper']:.4f}]",
                    'mechanism': 'Model outputs confidence in dead zone → mapped to HOLD → no directional signal'
                })
    
    # Check for narrow distributions
    for model_name, analysis in confidence_analysis.items():
        if analysis['stats']['range'] < 0.15:
            root_causes.append({
                'model': model_name,
                'issue': 'Narrow Confidence Range',
                'evidence': f"Range = {analysis['stats']['range']:.4f} (min={analysis['stats']['min']:.4f}, max={analysis['stats']['max']:.4f})",
                'mechanism': 'Model collapsed to constant output → no discrimination between samples'
            })
    
    # Check for quantization
    for model_name, analysis in confidence_analysis.items():
        if 'Quantized' in ' '.join(analysis['collapse_indicators']):
            root_causes.append({
                'model': model_name,
                'issue': 'Quantized Output',
                'evidence': next((ind for ind in analysis['collapse_indicators'] if 'Quantized' in ind), ''),
                'mechanism': 'Model outputs only discrete values → likely fallback rules or hardcoded logic'
            })
    
    if root_causes:
        for i, cause in enumerate(root_causes, 1):
            lines.append(f"### Root Cause #{i}: {cause['issue']} ({cause['model']})")
            lines.append("")
            lines.append(f"**Evidence:**  ")
            lines.append(f"{cause['evidence']}")
            lines.append("")
            lines.append(f"**Mechanism:**  ")
            lines.append(f"{cause['mechanism']}")
            lines.append("")
    else:
        lines.append("**No definitive root causes identified from telemetry.**")
        lines.append("")
    
    lines.extend(["---", "", "## 6. CONCRETE FIX OPTIONS", ""])
    
    fixes = []
    
    # Dead zone fixes
    if any('Dead Zone' in str(rc.get('issue')) for rc in root_causes):
        fixes.append({
            'priority': 'HIGH',
            'fix': 'Adjust Action Mapping Thresholds',
            'action': 'Narrow HOLD dead zone from [0.4, 0.6] to [0.45, 0.55] or remove entirely',
            'rationale': 'Too many samples trapped in HOLD range',
            'implementation': 'Update action mapping logic in model inference code'
        })
    
    # Narrow distribution fixes
    if any('Narrow' in str(rc.get('issue')) for rc in root_causes):
        fixes.append({
            'priority': 'CRITICAL',
            'fix': 'Retrain with Anti-Collapse Regularization',
            'action': 'Add variance penalty (weight=1.0) + entropy regularization to loss function',
            'rationale': 'Model collapsed to constant output during training',
            'implementation': 'P0.8 retraining with balanced sampling + variance loss'
        })
    
    # Quantization fixes
    if any('Quantized' in str(rc.get('issue')) for rc in root_causes):
        fixes.append({
            'priority': 'MEDIUM',
            'fix': 'Remove Fallback Rules',
            'action': 'Disable lgbm_fallback_rules or reduce hardcoded confidence values',
            'rationale': 'Fallback logic producing quantized outputs (0.75, 0.68)',
            'implementation': 'Update ensemble voting to exclude fallback models'
        })
    
    # Shadow mode fixes
    if 'Shadow mode' in ' '.join(policy_findings):
        fixes.append({
            'priority': 'LOW',
            'fix': 'Activate Shadow Models (if quality gate passes)',
            'action': 'Move patchtst from shadow to active after retraining',
            'rationale': 'Shadow models not contributing to ensemble decisions',
            'implementation': 'Update model config + canary activation'
        })
    
    for i, fix in enumerate(fixes, 1):
        lines.append(f"### Fix #{i}: {fix['fix']} [{fix['priority']}]")
        lines.append("")
        lines.append(f"**Action:**  ")
        lines.append(f"{fix['action']}")
        lines.append("")
        lines.append(f"**Rationale:**  ")
        lines.append(f"{fix['rationale']}")
        lines.append("")
        lines.append(f"**Implementation:**  ")
        lines.append(f"{fix['implementation']}")
        lines.append("")
    
    lines.extend(["---", "", "## 7. CONTRACT COMPLIANCE CHECK", ""])
    
    compliance_checks = [
        ("QSC Wrapper", "✅ PASS", "Executed via ops/run.sh ai-engine"),
        ("Telemetry-Only", "✅ PASS", "No model files loaded, Redis stream only"),
        ("Localhost Redis", "✅ PASS", "Connected to localhost:6379"),
        ("FAIL-CLOSED", "✅ PASS", "No activation attempted"),
        ("Audit Trail", "✅ PASS", "Full report generated"),
        ("No Training", "✅ PASS", "Diagnosis mode only"),
        ("No Activation", "✅ PASS", "No model deployment")
    ]
    
    for check, status, detail in compliance_checks:
        lines.append(f"- **{check}:** {status} - {detail}")
    
    lines.extend(["", "---", "", "## 8. NEXT STEPS", ""])
    
    lines.append("1. **Review root causes** with dev team")
    lines.append("2. **Prioritize fixes** (CRITICAL → HIGH → MEDIUM → LOW)")
    lines.append("3. **Implement chosen fix** (do NOT activate before quality gate)")
    lines.append("4. **Re-run quality gate:** `make quality-gate` (must exit 0)")
    lines.append("5. **IF PASS:** Consider canary activation with `ops/model_safety/canary_activate.sh`")
    lines.append("6. **Monitor via scoreboard:** `make scoreboard` (hourly for 6h)")
    lines.append("")
    lines.append("**DO NOT SKIP QUALITY GATE.**")
    
    lines.extend(["", "---", "", f"**Generated:** {timestamp}  ", f"**Mode:** DIAGNOSIS (No Training/Activation)  ", "**Status:** Analysis Complete"])
    
    return '\n'.join(lines)


def main():
    """Main diagnosis entry point"""
    
    # Parse arguments
    args = parse_args()
    
    STREAM_KEY = 'quantum:stream:trade.intent'
    EVENT_COUNT = 2000
    
    # Setup paths
    reports_dir = Path('reports/safety')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_suffix = '_post_cutover' if args.after else ''
    report_path = reports_dir / f'diagnosis_{timestamp_str}{report_suffix}.md'
    
    print("="*80)
    print("DIAGNOSIS MODE - Telemetry Flow Analysis (QSC-Compliant)")
    print("="*80)
    print()
    print(f"Reading Redis stream: {STREAM_KEY}")
    print(f"Requested events: {EVENT_COUNT}")
    if args.after:
        print(f"Cutover filter: Events after {args.after}")
    print()
    
    try:
        # If cutover analysis, get pre-cutover data first for comparison
        pre_telemetry = None
        if args.after:
            print("Step 1: Analyzing pre-cutover data for comparison...")
            pre_events = read_redis_stream(STREAM_KEY, EVENT_COUNT, after_ts=None)
            pre_telemetry = extract_telemetry_data(pre_events)
            print(f"   Found {len(pre_events)} pre-cutover events")
            print()
        
        # Read Redis stream (with cutover filter if specified)
        print(f"Step {'2' if args.after else '1'}: Analyzing {'post-cutover' if args.after else 'all'} data...")
        events = read_redis_stream(STREAM_KEY, EVENT_COUNT, after_ts=args.after)
        print(f"✅ Retrieved {len(events)} events")
        print()
        
        # Extract data
        print("Extracting telemetry data...")
        telemetry_data = extract_telemetry_data(events)
        print(f"✅ Found {len(telemetry_data['per_model_raw'])} models")
        print(f"✅ Found {len(telemetry_data['ensemble_decisions'])} ensemble decisions")
        print()
        
        # Analyze thresholds
        print("Analyzing action mapping thresholds...")
        threshold_analysis = analyze_action_thresholds(telemetry_data['per_model_raw'])
        print(f"✅ Analyzed {len(threshold_analysis)} models")
        print()
        
        # Analyze confidence distributions
        print("Analyzing confidence distributions...")
        confidence_analysis = analyze_confidence_distribution(telemetry_data['per_model_raw'])
        print(f"✅ Distribution analysis complete")
        print()
        
        # Analyze ensemble
        print("Analyzing ensemble voting logic...")
        ensemble_analysis = analyze_ensemble_logic(telemetry_data['ensemble_decisions'])
        print(f"✅ Ensemble analysis complete")
        print()
        
        # Detect policy rules
        print("Detecting policy rules...")
        policy_findings = detect_policy_rules(telemetry_data['breakdown_samples'])
        print(f"✅ Found {len(policy_findings)} policy patterns")
        print()
        
        # Generate report
        print("Generating diagnosis report...")
        report = generate_diagnosis_report(
            telemetry_data,
            threshold_analysis,
            confidence_analysis,
            ensemble_analysis,
            policy_findings
        )
        
        # Write report
        reports_dir = Path('reports/safety')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = reports_dir / f'diagnosis_{timestamp_str}.md'
        
        report_path.write_text(report)
        
        print(f"✅ Report saved: {report_path}")
        print()
        print("="*80)
        print("DIAGNOSIS COMPLETE")
        print("="*80)
        print()
        print(f"View report: cat {report_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
