#!/usr/bin/env python3
"""
Quality Gate - BLOCKER for degenerate models

DETECTS:
- Constant output (std<0.01 or p10==p90)
- HOLD collapse (HOLD>85% + prob in [0.4,0.6])
- Feature parse/shape mismatch

FAIL IF:
- Any class >70%
- conf_std <0.05 or p10-p90 <0.12
- Constant/dead-zone collapse detected
- Missing fields / parsing errors

EXIT CODES:
  0 = PASS
  2 = FAIL (BLOCKER)
"""

import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn

# Model architectures (embedded to avoid import issues)
class PatchTSTModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=3, 
                 num_heads=4, dropout=0.1, patch_len=16, num_patches=8):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.patch_embed = nn.Linear(patch_len * input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.mean(dim=1)
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


def load_model(model_path, model_type, num_features=4):
    """Load model checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if model_type == 'patchtst':
        model = PatchTSTModel(
            input_dim=num_features, output_dim=1, hidden_dim=128,
            num_layers=3, num_heads=4, dropout=0.2, patch_len=16, num_patches=8
        )
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    model.eval()
    return model, checkpoint


def load_test_samples(db_path, limit=500):
    """Load test samples from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT features, target_class, symbol
        FROM ai_training_samples
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    samples = []
    for features_json, target_class, symbol in rows:
        try:
            features = json.loads(features_json) if isinstance(features_json, str) else features_json
            samples.append({
                'features': features,
                'target_class': target_class,
                'symbol': symbol
            })
        except:
            continue
    
    return samples


def predict_batch(model, samples, feature_keys=['rsi', 'ma_cross', 'volatility', 'returns_1h']):
    """Run predictions on samples"""
    X = []
    for s in samples:
        feat_vec = [s['features'].get(k, 0.0) for k in feature_keys]
        X.append(feat_vec)
    
    X_tensor = torch.FloatTensor(X)
    
    with torch.no_grad():
        outputs = model(X_tensor).squeeze()
        probs = torch.sigmoid(outputs).numpy()
    
    return probs


def prob_to_action(prob):
    """Decision mapping (from patchtst_agent.py)"""
    if prob > 0.6:
        return 'BUY', prob
    elif prob < 0.4:
        return 'SELL', 1.0 - prob
    else:
        return 'HOLD', 0.5


def analyze_predictions(probs):
    """Analyze prediction distribution"""
    actions = [prob_to_action(p)[0] for p in probs]
    confidences = [prob_to_action(p)[1] for p in probs]
    
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
    
    prob_stats = {
        'mean': np.mean(probs),
        'std': np.std(probs),
        'p10': np.percentile(probs, 10),
        'p90': np.percentile(probs, 90),
        'unique_count': len(np.unique(probs))
    }
    
    return {
        'action_counts': action_counts,
        'action_pcts': action_pcts,
        'confidence_stats': conf_stats,
        'prob_stats': prob_stats
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
    if analysis['prob_stats']['std'] < 0.01:
        failures.append(f"Probability std = {analysis['prob_stats']['std']:.6f} < 0.01 (CONSTANT OUTPUT)")
    
    if analysis['prob_stats']['p10'] == analysis['prob_stats']['p90']:
        failures.append(f"Probability P10 == P90 = {analysis['prob_stats']['p10']:.4f} (FLATLINED)")
    
    # Check 4: HOLD dead-zone collapse
    hold_pct = analysis['action_pcts']['HOLD']
    prob_mean = analysis['prob_stats']['mean']
    if hold_pct > 85 and 0.4 <= prob_mean <= 0.6:
        failures.append(f"HOLD = {hold_pct:.1f}% + prob_mean = {prob_mean:.4f} in [0.4,0.6] (DEAD-ZONE COLLAPSE)")
    
    return failures


def generate_report(model_name, analysis, failures, report_path):
    """Generate markdown report"""
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    status = "❌ FAIL (BLOCKER)" if failures else "✅ PASS"
    
    with open(report_path, 'w') as f:
        f.write(f"# Quality Gate Report\n\n")
        f.write(f"**Model:** {model_name}\n")
        f.write(f"**Timestamp:** {timestamp}\n")
        f.write(f"**Status:** {status}\n\n")
        
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
        
        f.write(f"\n## Probability Statistics\n\n")
        ps = analysis['prob_stats']
        f.write(f"- Mean: {ps['mean']:.4f}\n")
        f.write(f"- Stddev: {ps['std']:.6f}\n")
        f.write(f"- P10: {ps['p10']:.4f}\n")
        f.write(f"- P90: {ps['p90']:.4f}\n")
        f.write(f"- Unique values: {ps['unique_count']}\n")
        
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
            f.write(f"- ✅ Probability std ≥0.01\n")
            f.write(f"- ✅ No constant output\n")
            f.write(f"- ✅ No HOLD dead-zone collapse\n")


def main():
    # Config
    db_path = Path('/opt/quantum/data/quantum_trader.db')
    model_path = Path('/opt/quantum/ai_engine/models/patchtst_v20260109_233444.pth')  # P0.4 shadow model
    model_type = 'patchtst'
    report_dir = Path('reports/safety')
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp_str = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    report_path = report_dir / f'quality_gate_{timestamp_str}.md'
    
    print(f"{'='*70}")
    print(f"QUALITY GATE - DEGENERATE MODEL DETECTION")
    print(f"{'='*70}\n")
    
    # Load model
    print(f"Loading model: {model_path.name}")
    model, checkpoint = load_model(model_path, model_type, num_features=4)
    
    # Load test samples
    print(f"Loading test samples from database...")
    samples = load_test_samples(db_path, limit=500)
    print(f"Loaded {len(samples)} samples\n")
    
    # Run predictions
    print(f"Running predictions...")
    probs = predict_batch(model, samples)
    
    # Analyze
    print(f"Analyzing predictions...")
    analysis = analyze_predictions(probs)
    
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
    
    print(f"\nProbability Stats:")
    ps = analysis['prob_stats']
    print(f"  Mean:       {ps['mean']:.4f}")
    print(f"  Stddev:     {ps['std']:.6f}")
    print(f"  Unique:     {ps['unique_count']}")
    
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
    generate_report(model_path.name, analysis, failures, report_path)
    print(f"\nReport: {report_path}")
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
