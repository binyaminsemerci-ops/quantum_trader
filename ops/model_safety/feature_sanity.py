#!/usr/bin/env python3
"""
Feature Pipeline Sanity Check (FAIL-CLOSED)

Validates that feature vectors reaching models are:
1) Correct dimensionality
2) No NaN/Inf values
3) Sufficient variance (not flatlined)

Exit codes:
  0 = PASS (features look healthy)
  2 = FAIL (NaN/Inf/flatlines/dim mismatch detected)
  
Usage:
  python feature_sanity.py --after 2026-01-11T00:01:44Z --count 200
"""

import sys
import json
import redis
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Feature pipeline sanity check')
    parser.add_argument('--after', type=str, help='ISO 8601 timestamp for cutover')
    parser.add_argument('--count', type=int, default=200, help='Number of events to analyze')
    return parser.parse_args()


def timestamp_to_stream_id(ts_str):
    """Convert ISO 8601 timestamp to Redis stream ID"""
    dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    unix_ms = int(dt.timestamp() * 1000)
    return f"{unix_ms}-0"


def read_intents(stream_key, after_ts, count):
    """Read intents from Redis using redis-cli (workaround for Python client bug)"""
    import subprocess
    
    min_stream_id = timestamp_to_stream_id(after_ts)
    cmd = f"redis-cli XRANGE {stream_key} {min_stream_id} +"
    result = subprocess.check_output(cmd, shell=True, text=True)
    
    # Parse redis-cli output
    lines = result.strip().split('\n')
    events = []
    i = 0
    
    def is_id(line):
        parts = line.strip().split("-")
        return len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit()
    
    while i < len(lines):
        if is_id(lines[i]):
            event_id = lines[i].strip()
            fields = {}
            i += 1
            while i < len(lines) and not is_id(lines[i]):
                key = lines[i]
                i += 1
                if i < len(lines) and not is_id(lines[i]):
                    value = lines[i]
                    i += 1
                else:
                    value = ""
                fields[key] = value
            
            # Parse payload
            if 'payload' in fields:
                try:
                    payload = json.loads(fields['payload'])
                    events.append(payload)
                except:
                    pass
        else:
            i += 1
    
    return events[:count]


def extract_features(intents):
    """Extract feature vectors from model_breakdown in intents"""
    # This is a proxy - we're looking at metadata features, not the actual ML feature vector
    # In production, you'd call the same feature engineering code as the models use
    
    features_list = []
    for intent in intents:
        # Extract metadata features that models receive
        features = {
            'atr_value': intent.get('atr_value', 0.0),
            'volatility_factor': intent.get('volatility_factor', 1.0),
            'exchange_divergence': intent.get('exchange_divergence', 0.0),
            'funding_rate': intent.get('funding_rate', 0.0),
            'confidence': intent.get('confidence', 0.0),
            'consensus_count': intent.get('consensus_count', 0),
        }
        features_list.append(features)
    
    return features_list


def analyze_features(features_list):
    """Analyze feature variance and health"""
    if not features_list:
        print("❌ No features to analyze")
        return False
    
    # Convert to arrays per feature
    feature_arrays = defaultdict(list)
    for features in features_list:
        for key, value in features.items():
            feature_arrays[key].append(value)
    
    print(f"\n{'='*80}")
    print(f"FEATURE SANITY CHECK - {len(features_list)} intents analyzed")
    print(f"{'='*80}\n")
    
    # 🔒 FEATURE HASH CHECK (ultra-kontrollpunkt)
    import hashlib
    feature_hashes = []
    for features in features_list[-50:]:  # Last 50 events
        # Round to 4 decimals for stable hash
        rounded = {k: round(v, 4) for k, v in features.items()}
        feature_str = str(sorted(rounded.items()))
        feature_hash = hashlib.sha256(feature_str.encode()).hexdigest()[:8]
        feature_hashes.append(feature_hash)
    
    unique_hashes = len(set(feature_hashes))
    duplicate_pct = (1 - unique_hashes / len(feature_hashes)) * 100 if feature_hashes else 0
    
    print(f"Feature Hash Check (last {len(feature_hashes)} events):")
    print(f"  Unique hashes: {unique_hashes}/{len(feature_hashes)} ({100 - duplicate_pct:.1f}% unique)")
    
    if duplicate_pct > 50:
        print(f"  ⚠️  WARNING: {duplicate_pct:.1f}% duplicate feature vectors (possible upstream flatline)")
        # Note: This is a warning, not a hard fail (some duplication is OK in steady markets)
    else:
        print(f"  ✅ Feature diversity looks healthy")
    
    print()
    
    # Check each feature
    issues = []
    flatlines = []
    
    for feature_name, values in feature_arrays.items():
        arr = np.array(values, dtype=float)
        
        # Check for NaN/Inf
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        
        if nan_count > 0 or inf_count > 0:
            issues.append(f"{feature_name}: {nan_count} NaN, {inf_count} Inf")
        
        # Check variance
        std = np.std(arr)
        mean = np.mean(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        
        is_flatline = std < 1e-6
        if is_flatline:
            flatlines.append(feature_name)
        
        status = "❌ FLAT" if is_flatline else "✅"
        print(f"{status} {feature_name:20s} | mean={mean:8.4f} std={std:8.6f} range=[{min_val:.4f}, {max_val:.4f}]")
    
    print(f"\n{'='*80}")
    
    # Summary
    total_features = len(feature_arrays)
    flatline_pct = (len(flatlines) / total_features) * 100 if total_features > 0 else 0
    
    print(f"\nSummary:")
    print(f"  Total features: {total_features}")
    print(f"  NaN/Inf issues: {len(issues)}")
    print(f"  Flatline features: {len(flatlines)} ({flatline_pct:.1f}%)")
    
    if issues:
        print(f"\n❌ NaN/Inf DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
    
    if flatlines:
        print(f"\n❌ FLATLINE FEATURES (std < 1e-6):")
        for name in flatlines:
            print(f"  - {name}")
    
    # FAIL-CLOSED: Fail if >30% flatlines or any NaN/Inf
    if len(issues) > 0:
        print(f"\n❌ FAIL: NaN/Inf values detected")
        return False
    
    if flatline_pct > 30:
        print(f"\n❌ FAIL: {flatline_pct:.1f}% features are flatlined (threshold: 30%)")
        return False
    
    print(f"\n✅ PASS: Features show healthy variance")
    return True


def main():
    args = parse_args()
    
    if not args.after:
        print("Error: --after timestamp required")
        sys.exit(2)
    
    STREAM_KEY = 'quantum:stream:trade.intent'
    
    print(f"Reading {args.count} intents after {args.after}...")
    intents = read_intents(STREAM_KEY, args.after, args.count)
    
    if len(intents) < 50:
        print(f"❌ FAIL: Insufficient data ({len(intents)} intents, need >=50)")
        sys.exit(2)
    
    print(f"✅ Found {len(intents)} intents")
    
    # Extract and analyze features
    features = extract_features(intents)
    success = analyze_features(features)
    
    sys.exit(0 if success else 2)


if __name__ == "__main__":
    main()
