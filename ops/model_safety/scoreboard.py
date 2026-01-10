#!/usr/bin/env python3
"""
Scoreboard - Combined model status overview (TELEMETRY-ONLY)

Uses same Redis stream telemetry as quality_gate.py

STATUS LOGIC:
- GO = passes quality gate + agreement 55-80% + hard_disagree <20%
- WAIT = passes gate but missing agreement data or outside range
- NO-GO = fails quality gate

OUTPUT: reports/safety/scoreboard_latest.md
"""

import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def read_redis_stream(stream_key='quantum:stream:trade.intent', count=2000):
    """Read last N events from Redis stream using redis-cli"""
    try:
        cmd = [
            'redis-cli',
            'XREVRANGE',
            stream_key,
            '+',
            '-',
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
    """Parse redis-cli XREVRANGE output into list of events"""
    lines = output.strip().split('\n')
    events = []
    
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        
        if lines[i].startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
            event_id_line = lines[i]
            event_id = event_id_line.split(')')[1].strip().strip('"')
            
            i += 1
            
            fields = {}
            while i < len(lines) and not lines[i].startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
                line = lines[i].strip()
                
                if line and not line.startswith(('1)', '2)', '3)', '4)', '5)', '6)', '7)', '8)', '9)')):
                    if i > 0:
                        prev_line = lines[i-1].strip()
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
    """Extract per-model predictions from events"""
    model_data = defaultdict(list)
    
    for event in events:
        fields = event.get('fields', {})
        
        breakdown_json = fields.get('model_breakdown', '{}')
        try:
            breakdown = json.loads(breakdown_json)
        except:
            continue
        
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


def calculate_model_stats(predictions):
    """Calculate action%, conf_std, p10-p90 for one model"""
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
    
    conf_std = np.std(confidences) if confidences else 0.0
    conf_p10 = np.percentile(confidences, 10) if confidences else 0.0
    conf_p90 = np.percentile(confidences, 90) if confidences else 0.0
    p10_p90_range = conf_p90 - conf_p10
    
    return {
        'action_pcts': action_pcts,
        'conf_std': conf_std,
        'p10_p90_range': p10_p90_range,
        'sample_count': total
    }


def calculate_ensemble_agreement(events):
    """Calculate agreement between models from ensemble votes"""
    agreements = []
    hard_disagrees = []
    
    for event in events:
        fields = event.get('fields', {})
        
        breakdown_json = fields.get('model_breakdown', '{}')
        try:
            breakdown = json.loads(breakdown_json)
        except:
            continue
        
        # Count votes
        vote_counts = defaultdict(int)
        for model_info in breakdown.values():
            if isinstance(model_info, dict):
                action = model_info.get('action')
                if action:
                    vote_counts[action] += 1
        
        total_votes = sum(vote_counts.values())
        
        if total_votes > 0:
            max_votes = max(vote_counts.values())
            agreement_pct = max_votes / total_votes * 100
            agreements.append(agreement_pct)
            
            # Hard disagree: BUY vs SELL split
            buy_votes = vote_counts.get('BUY', 0)
            sell_votes = vote_counts.get('SELL', 0)
            if buy_votes > 0 and sell_votes > 0:
                hard_disagrees.append(1)
            else:
                hard_disagrees.append(0)
    
    if not agreements:
        return None, None
    
    avg_agreement = np.mean(agreements)
    hard_disagree_pct = np.mean(hard_disagrees) * 100
    
    return avg_agreement, hard_disagree_pct


def check_quality_gate(stats):
    """Quick quality gate check (same logic as quality_gate.py)"""
    if not stats:
        return False
    
    # Check 1: No class >70%
    for pct in stats['action_pcts'].values():
        if pct > 70:
            return False
    
    # Check 2: Confidence spread
    if stats['conf_std'] < 0.05:
        return False
    
    if stats['p10_p90_range'] < 0.12:
        return False
    
    # Check 3: HOLD collapse
    if stats['action_pcts']['HOLD'] > 85:
        return False
    
    return True


def determine_status(stats, agreement, hard_disagree):
    """Determine GO/WAIT/NO-GO status"""
    quality_gate_pass = check_quality_gate(stats)
    
    if not quality_gate_pass:
        return 'NO-GO'
    
    if agreement is None:
        return 'WAIT'
    
    if 55 <= agreement <= 80 and hard_disagree < 20:
        return 'GO'
    else:
        return 'WAIT'


def generate_scoreboard(model_stats, agreement, hard_disagree, telemetry_info):
    """Generate scoreboard markdown report"""
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    lines = [
        "# Model Scoreboard (Telemetry-Only)",
        f"Generated: {timestamp}",
        "",
        "## Telemetry Info",
        f"- Redis stream: {telemetry_info['stream_key']}",
        f"- Events analyzed: {telemetry_info['event_count']}",
        f"- Time window: Last {telemetry_info['event_window']} events",
        "",
        "## Overall Status",
        ""
    ]
    
    # Determine overall status
    all_go = True
    any_nogo = False
    
    for model_name, stats in model_stats.items():
        status = determine_status(stats['stats'], agreement, hard_disagree)
        if status != 'GO':
            all_go = False
        if status == 'NO-GO':
            any_nogo = True
    
    if any_nogo:
        overall = "🔴 NO-GO"
    elif all_go:
        overall = "🟢 ALL-GO"
    else:
        overall = "🟡 WAIT"
    
    lines.append(f"**{overall}**")
    lines.append("")
    
    # Ensemble agreement
    lines.append("## Ensemble Agreement")
    if agreement is not None:
        lines.append(f"- Agreement: {agreement:.1f}%")
        lines.append(f"- Hard Disagree: {hard_disagree:.1f}%")
        
        if 55 <= agreement <= 80:
            lines.append("- ✅ Agreement in target range [55-80%]")
        else:
            lines.append("- ⚠️ Agreement outside target range")
        
        if hard_disagree < 20:
            lines.append("- ✅ Hard disagree below 20%")
        else:
            lines.append("- ⚠️ Hard disagree above 20%")
    else:
        lines.append("- ⚠️ Insufficient data")
    
    lines.append("")
    
    # Per-model status
    lines.append("## Model Status")
    lines.append("")
    
    for model_name, stats in sorted(model_stats.items()):
        status = determine_status(stats['stats'], agreement, hard_disagree)
        
        if status == 'GO':
            icon = "🟢"
        elif status == 'WAIT':
            icon = "🟡"
        else:
            icon = "🔴"
        
        lines.append(f"### {icon} {model_name} - {status}")
        lines.append("")
        
        model_data = stats['stats']
        
        lines.append("**Action Distribution:**")
        for action in ['BUY', 'SELL', 'HOLD']:
            pct = model_data['action_pcts'][action]
            lines.append(f"- {action}: {pct:.1f}%")
        
        lines.append("")
        lines.append("**Confidence Stats:**")
        lines.append(f"- Std: {model_data['conf_std']:.4f}")
        lines.append(f"- P10-P90 Range: {model_data['p10_p90_range']:.4f}")
        
        lines.append("")
        lines.append("**Quality Gate:**")
        
        gate_pass = check_quality_gate(model_data)
        if gate_pass:
            lines.append("- ✅ PASS")
        else:
            lines.append("- ❌ FAIL")
            
            # Show failures
            for pct in model_data['action_pcts'].values():
                if pct > 70:
                    lines.append(f"  - ❌ Action class >70% (collapse)")
            
            if model_data['conf_std'] < 0.05:
                lines.append(f"  - ❌ Confidence std <0.05 (flat)")
            
            if model_data['p10_p90_range'] < 0.12:
                lines.append(f"  - ❌ P10-P90 range <0.12 (narrow)")
            
            if model_data['action_pcts']['HOLD'] > 85:
                lines.append(f"  - ❌ HOLD >85% (dead zone collapse)")
        
        lines.append("")
        lines.append(f"Sample size: {model_data['sample_count']} events")
        lines.append("")
    
    return '\n'.join(lines)


def main():
    """Main scoreboard entry point"""
    
    STREAM_KEY = 'quantum:stream:trade.intent'
    EVENT_COUNT = 2000
    MIN_EVENTS = 200
    
    # Setup paths
    reports_dir = Path('reports/safety')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = reports_dir / 'scoreboard_latest.md'
    
    print(f"📊 Reading telemetry from Redis stream: {STREAM_KEY}")
    
    try:
        # Read Redis stream
        redis_output = read_redis_stream(STREAM_KEY, EVENT_COUNT)
        events = parse_redis_output(redis_output)
        
        telemetry_info = {
            'stream_key': STREAM_KEY,
            'event_count': len(events),
            'event_window': EVENT_COUNT
        }
        
        print(f"📦 Parsed {len(events)} events")
        
        # FAIL-CLOSED: Check minimum data
        if len(events) < MIN_EVENTS:
            print(f"⚠️  WARNING: Only {len(events)} events (need {MIN_EVENTS})")
            print("⚠️  Scoreboard may be inaccurate")
        
        # Extract model predictions
        model_data = extract_model_predictions(events)
        
        if not model_data:
            print("❌ No model_breakdown found in events")
            print("❌ Cannot generate scoreboard")
            return 1
        
        # Calculate per-model stats
        model_stats = {}
        for model_name, predictions in model_data.items():
            stats = calculate_model_stats(predictions)
            if stats:
                model_stats[model_name] = {
                    'stats': stats
                }
        
        # Calculate ensemble agreement
        agreement, hard_disagree = calculate_ensemble_agreement(events)
        
        # Generate report
        report = generate_scoreboard(model_stats, agreement, hard_disagree, telemetry_info)
        
        # Write report
        output_path.write_text(report)
        
        print(f"\n✅ Scoreboard saved: {output_path}")
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

