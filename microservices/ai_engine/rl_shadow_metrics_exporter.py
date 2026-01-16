#!/usr/bin/env python3
"""
RL Shadow Metrics Exporter for Prometheus
Exports RL shadow system metrics for Grafana dashboards.
"""
import os
import sys
import time
import json
import redis
from collections import Counter, defaultdict
from prometheus_client import start_http_server, Gauge, Counter as PromCounter, Histogram


# Prometheus metrics
rl_policy_age = Gauge('quantum_rl_policy_age_seconds', 'Age of RL policy in seconds', ['symbol'])
rl_gate_pass_rate = Gauge('quantum_rl_gate_pass_rate', 'RL gate pass rate (0-1)', ['symbol'])
rl_cooldown_blocking_rate = Gauge('quantum_rl_cooldown_blocking_rate', 'Cooldown blocking rate (0-1)', ['symbol'])
rl_eligible_rate = Gauge('quantum_rl_eligible_rate', 'Eligible pass rate (pass + cooldown)', ['symbol'])
rl_confidence_avg = Gauge('quantum_rl_confidence_avg', 'Average RL confidence', ['symbol'])
rl_ensemble_confidence_pass = Gauge('quantum_rl_ensemble_confidence_pass', 'Avg ensemble confidence when gate passes')
rl_ensemble_confidence_fail = Gauge('quantum_rl_ensemble_confidence_fail', 'Avg ensemble confidence when gate fails')
rl_intents_analyzed = PromCounter('quantum_rl_intents_analyzed_total', 'Total intents analyzed')
rl_gate_passes = PromCounter('quantum_rl_gate_passes_total', 'Total gate passes', ['symbol'])
rl_gate_failures = PromCounter('quantum_rl_gate_failures_total', 'Total gate failures', ['symbol', 'reason'])
rl_would_flip_rate = Gauge('quantum_rl_would_flip_rate', 'Would flip action rate', ['symbol'])
rl_reinforce_rate = Gauge('quantum_rl_reinforce_rate', 'Reinforce action rate', ['symbol'])


def load_env():
    """Load configuration from environment variables."""
    return {
        'redis_host': os.getenv('REDIS_HOST', '127.0.0.1'),
        'redis_port': int(os.getenv('REDIS_PORT', '6379')),
        'metrics_port': int(os.getenv('METRICS_PORT', '9091')),
        'update_interval': int(os.getenv('UPDATE_INTERVAL_SEC', '60')),
        'stream_name': os.getenv('STREAM_NAME', 'quantum:stream:trade.intent'),
        'sample_count': int(os.getenv('SAMPLE_COUNT', '500')),
        'policy_prefix': os.getenv('POLICY_PREFIX', 'quantum:rl:policy:')
    }


def connect_redis(config):
    """Connect to Redis."""
    try:
        r = redis.Redis(
            host=config['redis_host'],
            port=config['redis_port'],
            decode_responses=True
        )
        r.ping()
        print(f"[RL-METRICS] ✅ Connected to Redis {config['redis_host']}:{config['redis_port']}")
        return r
    except Exception as e:
        print(f"[RL-METRICS] ❌ Redis connection failed: {e}")
        sys.exit(1)


def parse_intent(entry):
    """Parse intent from stream entry."""
    try:
        msg_id, fields = entry
        if not isinstance(fields, dict):
            return None
        
        # Parse JSON payload
        payload_str = fields.get('payload')
        if not payload_str:
            return None
        
        payload = json.loads(payload_str)
        
        return {
            'symbol': payload.get('symbol'),
            'rl_gate_passed': payload.get('rl_gate_pass', False),
            'rl_gate_reason': payload.get('rl_gate_reason'),
            'rl_effect': payload.get('rl_effect'),
            'rl_confidence': float(payload.get('rl_confidence', 0)),
            'confidence': float(payload.get('confidence', 0)) if payload.get('confidence') else None,
            'rl_policy_age_sec': float(payload.get('rl_policy_age_sec', 0)) if payload.get('rl_policy_age_sec') else 0
        }
    except Exception as e:
        return None


def analyze_intents(redis_client, config):
    """Analyze recent intents and update metrics."""
    try:
        # Read recent intents
        entries = redis_client.xrevrange(config['stream_name'], count=config['sample_count'])
        
        if not entries:
            print(f"[RL-METRICS] ⚠️ No intents found in stream")
            return
        
        # Parse intents
        intents = []
        for entry in entries:
            intent = parse_intent(entry)
            if intent and intent['symbol']:
                intents.append(intent)
        
        if not intents:
            print(f"[RL-METRICS] ⚠️ No valid intents parsed")
            return
        
        # Aggregate by symbol
        stats = defaultdict(lambda: {
            'total': 0,
            'gate_pass': 0,
            'gate_fail': 0,
            'gate_reasons': Counter(),
            'rl_effects': Counter(),
            'rl_confidences': [],
            'ens_confidences_pass': [],
            'ens_confidences_fail': [],
            'policy_ages': []
        })
        
        for intent in intents:
            s = stats[intent['symbol']]
            s['total'] += 1
            rl_intents_analyzed.inc()
            
            if intent['rl_gate_passed']:
                s['gate_pass'] += 1
                rl_gate_passes.labels(symbol=intent['symbol']).inc()
                if intent['confidence'] is not None:
                    s['ens_confidences_pass'].append(intent['confidence'])
            else:
                s['gate_fail'] += 1
                reason = intent['rl_gate_reason'] or 'unknown'
                s['gate_reasons'][reason] += 1
                rl_gate_failures.labels(symbol=intent['symbol'], reason=reason).inc()
                if intent['confidence'] is not None:
                    s['ens_confidences_fail'].append(intent['confidence'])
            
            if intent['rl_effect']:
                s['rl_effects'][intent['rl_effect']] += 1
            
            if intent['rl_confidence'] > 0:
                s['rl_confidences'].append(intent['rl_confidence'])
            
            if intent['rl_policy_age_sec'] > 0:
                s['policy_ages'].append(intent['rl_policy_age_sec'])
        
        # Update Prometheus metrics
        for symbol, s in stats.items():
            if s['total'] == 0:
                continue
            
            # Pass rate
            pass_rate = s['gate_pass'] / s['total']
            rl_gate_pass_rate.labels(symbol=symbol).set(pass_rate)
            
            # Cooldown blocking rate
            cooldown_count = s['gate_reasons'].get('cooldown_active', 0)
            cooldown_rate = cooldown_count / s['total']
            rl_cooldown_blocking_rate.labels(symbol=symbol).set(cooldown_rate)
            
            # Eligible rate (pass + cooldown)
            eligible_rate = (s['gate_pass'] + cooldown_count) / s['total']
            rl_eligible_rate.labels(symbol=symbol).set(eligible_rate)
            
            # RL confidence
            if s['rl_confidences']:
                avg_rl_conf = sum(s['rl_confidences']) / len(s['rl_confidences'])
                rl_confidence_avg.labels(symbol=symbol).set(avg_rl_conf)
            
            # Policy age
            if s['policy_ages']:
                avg_age = sum(s['policy_ages']) / len(s['policy_ages'])
                rl_policy_age.labels(symbol=symbol).set(avg_age)
            
            # RL effects
            total_effects = sum(s['rl_effects'].values())
            if total_effects > 0:
                would_flip_rate = s['rl_effects']['would_flip'] / total_effects
                reinforce_rate = s['rl_effects']['reinforce'] / total_effects
                rl_would_flip_rate.labels(symbol=symbol).set(would_flip_rate)
                rl_reinforce_rate.labels(symbol=symbol).set(reinforce_rate)
        
        # Global ensemble confidence
        all_pass_confs = []
        all_fail_confs = []
        for s in stats.values():
            all_pass_confs.extend(s['ens_confidences_pass'])
            all_fail_confs.extend(s['ens_confidences_fail'])
        
        if all_pass_confs:
            avg_pass = sum(all_pass_confs) / len(all_pass_confs)
            rl_ensemble_confidence_pass.set(avg_pass)
        
        if all_fail_confs:
            avg_fail = sum(all_fail_confs) / len(all_fail_confs)
            rl_ensemble_confidence_fail.set(avg_fail)
        
        print(f"[RL-METRICS] 📊 Updated metrics: {len(intents)} intents, {len(stats)} symbols")
        
    except Exception as e:
        print(f"[RL-METRICS] ❌ Error analyzing intents: {e}")


def main():
    """Main loop."""
    config = load_env()
    redis_client = connect_redis(config)
    
    # Start Prometheus HTTP server
    start_http_server(config['metrics_port'])
    print(f"[RL-METRICS] 🚀 Prometheus metrics server started on port {config['metrics_port']}")
    print(f"[RL-METRICS] 📡 Analyzing stream: {config['stream_name']} (sample={config['sample_count']})")
    print(f"[RL-METRICS] ⏱️  Update interval: {config['update_interval']}s")
    
    while True:
        analyze_intents(redis_client, config)
        time.sleep(config['update_interval'])


if __name__ == '__main__':
    main()
