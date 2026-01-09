#!/usr/bin/env python3
"""
Parse PatchTST behavior from recent ensemble logs
Extract action and confidence distribution to confirm if stuck
"""
import json
import re
from collections import Counter

# Sample logs from recent 2 hours (from grep output)
SAMPLE_LOGS = """
patchtst': {'action': 'BUY', 'confidence': 0.650386393070221, 'model': 'patchtst_model'}
patchtst': {'action': 'BUY', 'confidence': 0.650386393070221, 'model': 'patchtst_model'}
patchtst': {'action': 'BUY', 'confidence': 0.650386393070221, 'model': 'patchtst_model'}
patchtst': {'action': 'BUY', 'confidence': 0.650386393070221, 'model': 'patchtst_model'}
patchtst': {'action': 'BUY', 'confidence': 0.650386393070221, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
patchtst': {'action': 'HOLD', 'confidence': 0.5, 'model': 'patchtst_model'}
"""

def parse_patchtst_logs():
    """Parse PatchTST predictions from logs"""
    
    # Extract action and confidence using regex
    pattern = r"'action': '([A-Z]+)', 'confidence': ([0-9.]+)"
    
    matches = re.findall(pattern, SAMPLE_LOGS)
    
    actions = []
    confidences = []
    
    for action, conf_str in matches:
        actions.append(action)
        confidences.append(float(conf_str))
    
    # Calculate statistics
    action_counts = Counter(actions)
    
    print("=" * 60)
    print("PATCHTST BEHAVIOR ANALYSIS (Recent 2 hours)")
    print("=" * 60)
    print()
    
    print(f"Total Predictions: {len(actions)}")
    print()
    
    print("ACTION DISTRIBUTION:")
    for action, count in action_counts.most_common():
        pct = (count / len(actions)) * 100
        print(f"  {action:5s}: {count:3d} ({pct:5.1f}%)")
    print()
    
    print("CONFIDENCE DISTRIBUTION:")
    conf_counter = Counter(confidences)
    for conf, count in sorted(conf_counter.items()):
        pct = (count / len(confidences)) * 100
        print(f"  {conf:.3f}: {count:3d} ({pct:5.1f}%)")
    print()
    
    # Check if stuck
    unique_confidences = len(set(confidences))
    print("DIAGNOSIS:")
    print(f"  Unique confidence values: {unique_confidences}")
    
    if unique_confidences <= 2:
        print("  ⚠️  STUCK: Only 1-2 distinct confidence values")
        print("  ⚠️  Model appears flatlined (stuck at fixed outputs)")
    else:
        print("  ✅ HEALTHY: Model shows varied confidence")
    
    print()
    
    # Check for BUY bias
    if action_counts.get('BUY', 0) == 0 and action_counts.get('SELL', 0) == 0:
        print("  ⚠️  ALL HOLD: Model never predicts BUY/SELL")
    elif action_counts.get('HOLD', 0) / len(actions) > 0.7:
        print("  ⚠️  HOLD BIAS: >70% predictions are HOLD")
    else:
        print("  ✅ Action diversity present")
    
    print()
    
    return {
        'total': len(actions),
        'action_counts': dict(action_counts),
        'unique_confidences': unique_confidences,
        'conf_values': sorted(set(confidences))
    }

if __name__ == '__main__':
    result = parse_patchtst_logs()
    print("RAW DATA:")
    print(json.dumps(result, indent=2))
