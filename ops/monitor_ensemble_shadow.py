#!/usr/bin/env python3
"""
Shadow Mode Monitor - PATH 2.3D

Observes ensemble predictor output WITHOUT consumption.
No apply_layer integration. Pure metrics collection.

Metrics:
- Signal distribution (CLOSE vs HOLD)
- Confidence distribution (bucketed)
- Validator drop rate
- Model agreement rate
- Stream lag/health

Duration: 24-72h observation period
"""
import redis
import time
import json
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Any


def get_redis_client():
    """Get Redis client."""
    return redis.Redis(
        host='localhost',
        port=6379,
        decode_responses=True
    )


def get_stream_info(r: redis.Redis, stream: str) -> Dict[str, Any]:
    """Get stream info."""
    try:
        info = r.xinfo_stream(stream)
        return {
            "length": info['length'],
            "first_entry": info.get('first-entry', [None])[0],
            "last_entry": info.get('last-entry', [None])[0],
            "groups": r.xinfo_groups(stream) if info['length'] > 0 else []
        }
    except redis.exceptions.ResponseError:
        return {"error": "Stream does not exist yet"}


def sample_recent_signals(
    r: redis.Redis,
    stream: str,
    count: int = 100
) -> List[Dict[str, Any]]:
    """Sample recent signals for analysis."""
    try:
        # Read last N messages
        messages = r.xrevrange(stream, '+', '-', count=count)
        
        signals = []
        for msg_id, fields in messages:
            signal = {k: v for k, v in fields.items()}
            signal['_msg_id'] = msg_id
            signals.append(signal)
        
        return signals
    except redis.exceptions.ResponseError:
        return []


def analyze_signals(signals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze signal distribution."""
    if not signals:
        return {"error": "No signals to analyze"}
    
    # Action distribution
    action_counts = Counter(s.get('suggested_action') for s in signals)
    
    # Confidence buckets
    confidence_buckets = defaultdict(int)
    for s in signals:
        try:
            conf = float(s.get('confidence', 0))
            if conf < 0.2:
                bucket = "0.0-0.2"
            elif conf < 0.4:
                bucket = "0.2-0.4"
            elif conf < 0.6:
                bucket = "0.4-0.6"
            elif conf < 0.8:
                bucket = "0.6-0.8"
            else:
                bucket = "0.8-1.0"
            confidence_buckets[bucket] += 1
        except (ValueError, TypeError):
            pass
    
    # Model agreement (count unique model combinations)
    model_combinations = Counter(s.get('models_used') for s in signals)
    
    # Symbol distribution
    symbol_counts = Counter(s.get('symbol') for s in signals)
    
    # Expected edge distribution
    edges = []
    for s in signals:
        try:
            edge = float(s.get('expected_edge', 0))
            edges.append(edge)
        except (ValueError, TypeError):
            pass
    
    avg_edge = sum(edges) / len(edges) if edges else 0
    positive_edge_pct = sum(1 for e in edges if e > 0) / len(edges) if edges else 0
    
    return {
        "total_signals": len(signals),
        "action_distribution": dict(action_counts),
        "action_percentages": {
            k: f"{(v/len(signals)*100):.1f}%" 
            for k, v in action_counts.items()
        },
        "confidence_buckets": dict(confidence_buckets),
        "model_combinations": dict(model_combinations.most_common(5)),
        "symbol_distribution": dict(symbol_counts.most_common(10)),
        "expected_edge": {
            "average": round(avg_edge, 4),
            "positive_pct": f"{(positive_edge_pct*100):.1f}%"
        }
    }


def print_dashboard(stats: Dict[str, Any]):
    """Print formatted dashboard."""
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTOR — SHADOW MODE METRICS")
    print(f"PATH 2.3D | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Stream health
    print("\n📊 STREAM HEALTH")
    print(f"   Stream: quantum:stream:signal.score")
    print(f"   Length: {stats['stream_info'].get('length', 'N/A')}")
    print(f"   First: {stats['stream_info'].get('first_entry', 'N/A')}")
    print(f"   Last: {stats['stream_info'].get('last_entry', 'N/A')}")
    
    # Analysis
    analysis = stats.get('analysis', {})
    
    if 'error' in analysis:
        print(f"\n⚠️  {analysis['error']}")
        return
    
    print(f"\n🎯 SIGNAL ANALYSIS (last {analysis['total_signals']} signals)")
    
    # Action distribution
    print("\n   Action Distribution:")
    for action, pct in analysis['action_percentages'].items():
        count = analysis['action_distribution'][action]
        bar = "█" * int(float(pct.rstrip('%')) // 2)
        print(f"      {action:6} {count:4} ({pct:6}) {bar}")
    
    # Confidence buckets
    print("\n   Confidence Distribution:")
    for bucket in ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]:
        count = analysis['confidence_buckets'].get(bucket, 0)
        pct = (count / analysis['total_signals'] * 100) if analysis['total_signals'] > 0 else 0
        bar = "█" * int(pct // 2)
        print(f"      {bucket:8} {count:4} ({pct:5.1f}%) {bar}")
    
    # Expected edge
    print(f"\n   Expected Edge:")
    print(f"      Average: {analysis['expected_edge']['average']}")
    print(f"      Positive: {analysis['expected_edge']['positive_pct']}")
    
    # Top symbols
    print(f"\n   Top Symbols (by signal count):")
    for symbol, count in list(analysis['symbol_distribution'].items())[:5]:
        print(f"      {symbol:10} {count:4}")
    
    # Model combinations
    print(f"\n   Model Combinations:")
    for models, count in list(analysis['model_combinations'].items())[:3]:
        print(f"      {models[:40]:40} {count:4}")
    
    print("\n" + "="*70)
    print("🔍 SHADOW MODE: NO DOWNSTREAM CONSUMERS")
    print("📋 Authority: SCORER ONLY | No execution surface")
    print("="*70 + "\n")


def main():
    """Main monitoring loop."""
    print("\n🎯 Starting Shadow Mode Monitor (PATH 2.3D)")
    print("   Stream: quantum:stream:signal.score")
    print("   Observation only - NO consumption")
    print("   Press Ctrl+C to exit\n")
    
    r = get_redis_client()
    
    try:
        while True:
            # Get stream info
            stream_info = get_stream_info(r, "quantum:stream:signal.score")
            
            # Sample recent signals
            signals = sample_recent_signals(r, "quantum:stream:signal.score", count=100)
            
            # Analyze
            analysis = analyze_signals(signals)
            
            # Print dashboard
            stats = {
                "stream_info": stream_info,
                "analysis": analysis
            }
            print_dashboard(stats)
            
            # Wait before next update
            time.sleep(60)  # Update every 60s
            
    except KeyboardInterrupt:
        print("\n\n✅ Shadow mode monitor stopped")


if __name__ == "__main__":
    main()
