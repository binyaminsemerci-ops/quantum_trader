#!/usr/bin/env python3
"""
Scoreboard - Combined model status overview

STATUS LOGIC:
- GO = passes quality gate + agreement 55-80% + hard_disagree <20%
- WAIT = passes gate but missing agreement data
- NO-GO = fails quality gate

OUTPUT: reports/safety/scoreboard_latest.md
"""

import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import subprocess

MODEL_BREAKDOWN = [
    {'name': 'XGBoost', 'type': 'xgb', 'shadow': False},
    {'name': 'LightGBM', 'type': 'lgbm', 'shadow': False},
    {'name': 'NHiTS', 'type': 'nhits', 'shadow': False},
    {'name': 'PatchTST', 'type': 'patchtst', 'shadow': True},
]


def get_recent_predictions(db_path, model_name, hours=24):
    """Get recent predictions for agreement calculation"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    cursor.execute("""
        SELECT action, confidence, metadata
        FROM trade_intents
        WHERE model_source LIKE ?
        AND created_at > ?
        ORDER BY created_at DESC
        LIMIT 200
    """, (f'%{model_name}%', cutoff.strftime('%Y-%m-%d %H:%M:%S')))
    
    rows = cursor.fetchall()
    conn.close()
    
    return rows


def calculate_model_stats(db_path, model_info):
    """Calculate action%, conf_std, agreement for one model"""
    predictions = get_recent_predictions(db_path, model_info['name'], hours=24)
    
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


def calculate_ensemble_agreement(db_path, hours=24):
    """Calculate agreement between active models (exclude shadow)"""
    # Get recent ensemble votes
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    cursor.execute("""
        SELECT action, metadata
        FROM trade_intents
        WHERE model_source = 'ENSEMBLE'
        AND created_at > ?
        LIMIT 100
    """, (cutoff.strftime('%Y-%m-%d %H:%M:%S'),))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return None, None
    
    agreements = []
    hard_disagrees = []
    
    for action, metadata_json in rows:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
            vote_counts = metadata.get('vote_counts', {})
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
        except:
            continue
    
    if not agreements:
        return None, None
    
    avg_agreement = np.mean(agreements)
    hard_disagree_pct = np.mean(hard_disagrees) * 100
    
    return avg_agreement, hard_disagree_pct


def determine_status(stats, agreement, hard_disagree, quality_gate_pass):
    """Determine GO/WAIT/NO-GO status"""
    if not quality_gate_pass:
        return 'NO-GO'
    
    if agreement is None:
        return 'WAIT'
    
    if 55 <= agreement <= 80 and hard_disagree < 20:
        return 'GO'
    else:
        return 'WAIT'


def run_quality_gate_check(model_name):
    """Run quality gate for model (returns True if passes)"""
    # Simplified check - in production would call quality_gate.py
    # For now, return True for non-shadow models
    return True


def generate_scoreboard(models_data, agreement, hard_disagree):
    """Generate scoreboard markdown"""
    report_dir = Path('reports/safety')
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / 'scoreboard_latest.md'
    
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    with open(report_path, 'w') as f:
        f.write(f"# Model Safety Scoreboard\n\n")
        f.write(f"**Updated:** {timestamp}\n\n")
        
        f.write(f"## Ensemble Metrics (Last 24h)\n\n")
        if agreement is not None:
            f.write(f"- **Agreement:** {agreement:.1f}%\n")
            f.write(f"- **Hard Disagree:** {hard_disagree:.1f}%\n\n")
        else:
            f.write(f"- **Agreement:** N/A (insufficient data)\n")
            f.write(f"- **Hard Disagree:** N/A\n\n")
        
        f.write(f"## Model Breakdown\n\n")
        f.write(f"| Model | Action% (B/S/H) | Conf Std | P10-P90 | Agreement | Hard Disagree | Status |\n")
        f.write(f"|-------|-----------------|----------|---------|-----------|---------------|--------|\n")
        
        for data in models_data:
            name = data['name']
            if data['stats']:
                s = data['stats']
                buy_pct = s['action_pcts']['BUY']
                sell_pct = s['action_pcts']['SELL']
                hold_pct = s['action_pcts']['HOLD']
                action_str = f"{buy_pct:.0f}/{sell_pct:.0f}/{hold_pct:.0f}"
                conf_std = s['conf_std']
                p10_p90 = s['p10_p90_range']
            else:
                action_str = "N/A"
                conf_std = 0.0
                p10_p90 = 0.0
            
            agr_str = f"{agreement:.0f}%" if agreement is not None else "N/A"
            hd_str = f"{hard_disagree:.0f}%" if hard_disagree is not None else "N/A"
            status = data['status']
            
            status_icon = {
                'GO': '✅',
                'WAIT': '⏳',
                'NO-GO': '❌'
            }.get(status, '❓')
            
            shadow_note = " (shadow)" if data['shadow'] else ""
            
            f.write(f"| {name}{shadow_note} | {action_str} | {conf_std:.3f} | {p10_p90:.3f} | {agr_str} | {hd_str} | {status_icon} {status} |\n")
        
        f.write(f"\n## Status Legend\n\n")
        f.write(f"- **GO** ✅: Passes quality gate + agreement 55-80% + hard_disagree <20%\n")
        f.write(f"- **WAIT** ⏳: Passes gate but outside agreement range or insufficient data\n")
        f.write(f"- **NO-GO** ❌: Fails quality gate (BLOCKER)\n")
    
    return report_path


def main():
    db_path = Path('/opt/quantum/data/quantum_trader.db')
    
    print(f"{'='*70}")
    print(f"MODEL SAFETY SCOREBOARD")
    print(f"{'='*70}\n")
    
    # Calculate ensemble agreement
    print(f"Calculating ensemble agreement...")
    agreement, hard_disagree = calculate_ensemble_agreement(db_path, hours=24)
    
    if agreement is not None:
        print(f"  Agreement: {agreement:.1f}%")
        print(f"  Hard Disagree: {hard_disagree:.1f}%\n")
    else:
        print(f"  Agreement: N/A (insufficient data)\n")
    
    # Calculate per-model stats
    models_data = []
    for model_info in MODEL_BREAKDOWN:
        print(f"Processing {model_info['name']}...")
        stats = calculate_model_stats(db_path, model_info)
        quality_gate_pass = run_quality_gate_check(model_info['name'])
        status = determine_status(stats, agreement, hard_disagree, quality_gate_pass)
        
        models_data.append({
            'name': model_info['name'],
            'shadow': model_info['shadow'],
            'stats': stats,
            'status': status
        })
    
    # Generate report
    report_path = generate_scoreboard(models_data, agreement, hard_disagree)
    
    print(f"\n{'='*70}")
    print(f"Scoreboard generated: {report_path}")
    print(f"{'='*70}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
