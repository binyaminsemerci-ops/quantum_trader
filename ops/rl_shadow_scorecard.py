#!/usr/bin/env python3
"""
RL Shadow Scorecard - Read-only intent stream analysis
Aggregates RL gate performance metrics from trade.intent stream
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict, Counter

try:
    import redis
except ImportError:
    print("[RL-SCORECARD] ❌ ERROR: redis module not found. Install: pip3 install redis")
    sys.exit(0)


def load_env():
    """Load configuration from environment"""
    return {
        'redis_host': os.getenv('REDIS_HOST', '127.0.0.1'),
        'redis_port': int(os.getenv('REDIS_PORT', 6379)),
        'stream': os.getenv('STREAM', 'quantum:stream:trade.intent'),
        'count': int(os.getenv('COUNT', 2000)),
        'topn': int(os.getenv('TOPN', 10))
    }


def connect_redis(config):
    """Connect to Redis"""
    try:
        r = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True,
            socket_connect_timeout=5
        )
        r.ping()
        return r
    except Exception as e:
        print(f"[RL-SCORECARD] ❌ Redis connection failed: {e}")
        sys.exit(0)


def parse_intent(entry):
    """Parse intent entry from Redis stream"""
    try:
        # entry is tuple: (entry_id, {field: value, ...})
        if not entry or len(entry) != 2:
            return None
        
        entry_id, data = entry
        
        # Get payload field (usually 'payload' or similar)
        payload_str = None
        for key in ['payload', 'data', 'message']:
            if key in data:
                payload_str = data[key]
                break
        
        if not payload_str:
            # Try first value if no known key
            if data:
                payload_str = list(data.values())[0]
        
        if not payload_str:
            return None
        
        # Parse JSON
        payload = json.loads(payload_str)
        
        # Extract fields (handle missing gracefully)
        return {
            'symbol': payload.get('symbol'),
            'rl_gate_pass': payload.get('rl_gate_pass'),
            'rl_gate_reason': payload.get('rl_gate_reason'),
            'rl_effect': payload.get('rl_effect'),
            'rl_confidence': payload.get('rl_confidence'),
            'confidence': payload.get('confidence'),  # ensemble confidence
            'rl_policy_age_sec': payload.get('rl_policy_age_sec')
        }
    except Exception as e:
        # Skip malformed entries silently
        return None


def aggregate_stats(intents):
    """Aggregate statistics per symbol and globally"""
    stats = defaultdict(lambda: {
        'total': 0,
        'gate_pass': 0,
        'gate_fail': 0,
        'gate_reasons': Counter(),
        'rl_effects': Counter(),
        'rl_confidences': [],
        'ens_confidences': [],
        'ens_confidences_pass': [],
        'ens_confidences_fail': [],
        'policy_ages': []
    })
    
    # Global stats
    global_stats = {
        'total_intents': 0,
        'ens_conf_present': 0,
        'all_gate_reasons': Counter()
    }
    
    for intent in intents:
        if not intent or not intent.get('symbol'):
            continue
        
        symbol = intent['symbol']
        s = stats[symbol]
        
        s['total'] += 1
        global_stats['total_intents'] += 1
        
        # Gate pass/fail tracking
        gate_passed = intent.get('rl_gate_pass') is True
        if gate_passed:
            s['gate_pass'] += 1
        else:
            s['gate_fail'] += 1
        
        # Gate reasons (global tracking)
        if intent.get('rl_gate_reason'):
            reason = intent['rl_gate_reason']
            s['gate_reasons'][reason] += 1
            global_stats['all_gate_reasons'][reason] += 1
        
        # RL effects
        if intent.get('rl_effect'):
            s['rl_effects'][intent['rl_effect']] += 1
        
        # RL confidence
        if intent.get('rl_confidence') is not None:
            s['rl_confidences'].append(intent['rl_confidence'])
        
        # Ensemble confidence tracking (global + per pass/fail)
        ens_conf = intent.get('confidence')
        if ens_conf is not None:
            s['ens_confidences'].append(ens_conf)
            global_stats['ens_conf_present'] += 1
            
            if gate_passed:
                s['ens_confidences_pass'].append(ens_conf)
            else:
                s['ens_confidences_fail'].append(ens_conf)
        
        # Policy age
        if intent.get('rl_policy_age_sec') is not None:
            s['policy_ages'].append(intent['rl_policy_age_sec'])
    
    return stats, global_stats


def print_report(stats, global_stats, config):
    """Print enhanced scorecard report"""
    timestamp = datetime.utcnow().isoformat() + 'Z'
    
    print(f"\n{'='*80}")
    print(f"[RL-SCORECARD] 📊 Shadow Performance Report")
    print(f"Timestamp: {timestamp}")
    print(f"Analyzed: {config['count']} most recent intents from {config['stream']}")
    print(f"{'='*80}\n")
    
    # Sort by total intents
    sorted_symbols = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
    top_symbols = sorted_symbols[:config['topn']]
    
    if not top_symbols:
        print("[RL-SCORECARD] ⚠️  No intents found in stream")
        return
    
    print(f"Top {len(top_symbols)} Symbols by Intent Volume:\n")
    
    for rank, (symbol, s) in enumerate(top_symbols, 1):
        pass_rate = (s['gate_pass'] / s['total'] * 100) if s['total'] > 0 else 0
        
        # Eligible rate (pass + cooldown_active)
        cooldown_count = s['gate_reasons'].get('cooldown_active', 0)
        eligible_passes = s['gate_pass'] + cooldown_count
        eligible_rate = (eligible_passes / s['total'] * 100) if s['total'] > 0 else 0
        
        # Calculate effect rates
        total_effects = sum(s['rl_effects'].values())
        would_flip_rate = (s['rl_effects']['would_flip'] / total_effects * 100) if total_effects > 0 else 0
        reinforce_rate = (s['rl_effects']['reinforce'] / total_effects * 100) if total_effects > 0 else 0
        
        # Top 2 gate reasons for this symbol
        top_2_reasons = s['gate_reasons'].most_common(2)
        if len(top_2_reasons) >= 2:
            r1, c1 = top_2_reasons[0]
            r2, c2 = top_2_reasons[1]
            r1_pct = (c1 / s['total'] * 100) if s['total'] > 0 else 0
            r2_pct = (c2 / s['total'] * 100) if s['total'] > 0 else 0
            reasons_str = f"{r1}({r1_pct:.1f}%), {r2}({r2_pct:.1f}%)"
        elif len(top_2_reasons) == 1:
            r1, c1 = top_2_reasons[0]
            r1_pct = (c1 / s['total'] * 100) if s['total'] > 0 else 0
            reasons_str = f"{r1}({r1_pct:.1f}%)"
        else:
            reasons_str = "none"
        
        # Average confidences
        avg_rl_conf = sum(s['rl_confidences']) / len(s['rl_confidences']) if s['rl_confidences'] else 0
        avg_ens_conf = sum(s['ens_confidences']) / len(s['ens_confidences']) if s['ens_confidences'] else 0
        
        # Average policy age
        avg_age = sum(s['policy_ages']) / len(s['policy_ages']) if s['policy_ages'] else 0
        
        print(f"{rank}. {symbol:12s} | intents={s['total']:4d} | pass_rate={pass_rate:5.1f}% | eligible_rate={eligible_rate:5.1f}% | top_reasons={reasons_str}")
        
        if s['gate_pass'] > 0:
            print(f"   └─ RL effects: would_flip={would_flip_rate:.1f}% | reinforce={reinforce_rate:.1f}%")
            print(f"   └─ Avg RL conf={avg_rl_conf:.2f} | Ens conf={avg_ens_conf:.2f} | Policy age={avg_age:.0f}s")
        
        print()
    
    # Summary stats
    total_intents = global_stats['total_intents']
    total_passes = sum(s['gate_pass'] for s in stats.values())
    overall_pass_rate = (total_passes / total_intents * 100) if total_intents > 0 else 0
    
    # Eligible passes (pass + cooldown_active)
    total_cooldown = global_stats['all_gate_reasons'].get('cooldown_active', 0)
    eligible_passes = total_passes + total_cooldown
    eligible_rate = (eligible_passes / total_intents * 100) if total_intents > 0 else 0
    
    print(f"{'='*80}")
    print(f"SUMMARY: {total_intents} total intents | {total_passes} gate passes ({overall_pass_rate:.1f}%)")
    print(f"ELIGIBLE: {eligible_passes} eligible passes (pass + cooldown) | eligible_rate={eligible_rate:.1f}%")
    
    # Global gate reason distribution (Top 8)
    print(f"\nGlobal Gate Reason Distribution (Top 8):")
    for reason, count in global_stats['all_gate_reasons'].most_common(8):
        pct = (count / total_intents * 100) if total_intents > 0 else 0
        print(f"  {reason:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Ensemble confidence insights
    print(f"\nEnsemble Confidence Insights:")
    ens_present_rate = (global_stats['ens_conf_present'] / total_intents * 100) if total_intents > 0 else 0
    
    if global_stats['ens_conf_present'] > 0:
        print(f"  ens_conf_present_rate: {ens_present_rate:.1f}% ({global_stats['ens_conf_present']}/{total_intents})")
        
        # Calculate avg ens_conf when pass vs fail
        all_pass_confs = []
        all_fail_confs = []
        for s in stats.values():
            all_pass_confs.extend(s['ens_confidences_pass'])
            all_fail_confs.extend(s['ens_confidences_fail'])
        
        if all_pass_confs:
            avg_ens_pass = sum(all_pass_confs) / len(all_pass_confs)
            print(f"  ens_conf_avg_when_pass: {avg_ens_pass:.3f} (n={len(all_pass_confs)})")
        
        if all_fail_confs:
            avg_ens_fail = sum(all_fail_confs) / len(all_fail_confs)
            print(f"  ens_conf_avg_when_fail: {avg_ens_fail:.3f} (n={len(all_fail_confs)})")
    else:
        print(f"  ens_conf not present in payloads")
    
    print(f"{'='*80}\n")


def main():
    """Main scorecard execution"""
    try:
        print(f"[RL-SCORECARD] 🚀 Starting shadow performance analysis...")
        
        config = load_env()
        r = connect_redis(config)
        
        print(f"[RL-SCORECARD] 📡 Reading {config['count']} intents from {config['stream']}...")
        
        # Read stream entries (XREVRANGE for most recent)
        try:
            entries = r.xrevrange(config['stream'], count=config['count'])
        except Exception as e:
            print(f"[RL-SCORECARD] ❌ Stream read failed: {e}")
            sys.exit(0)
        
        if not entries:
            print(f"[RL-SCORECARD] ⚠️  No entries found in stream")
            sys.exit(0)
        
        print(f"[RL-SCORECARD] 📊 Parsing {len(entries)} entries...")
        
        # Parse intents
        intents = []
        for entry in entries:
            parsed = parse_intent(entry)
            if parsed:
                intents.append(parsed)
        
        print(f"[RL-SCORECARD] ✅ Parsed {len(intents)} valid intents")
        
        # Aggregate stats (returns tuple now)
        stats, global_stats = aggregate_stats(intents)
        
        # Print enhanced report
        print_report(stats, global_stats, config)
        
        print(f"[RL-SCORECARD] ✅ Report complete\n")
        
    except Exception as e:
        print(f"[RL-SCORECARD] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)


if __name__ == '__main__':
    main()
