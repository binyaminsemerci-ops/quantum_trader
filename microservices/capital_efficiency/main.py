#!/usr/bin/env python3
"""
P3.1 Capital Efficiency Brain - Fund-Grade Capital Allocation Scoring

Computes efficiency_score per symbol using:
- P3.0 Alpha Attribution (performance_factor)
- P2.9 Allocation Target (target_usd)
- Execution PnL stream (volatility + drawdown proxies)
- Holding time penalty

Fail-open: Missing/stale inputs => low confidence, stale_flags, LKG values
Shadow/enforce modes: Shadow writes stream+metrics only, enforce writes hashes
"""

import os
import sys
import time
import json
import logging
import redis
from datetime import datetime
from collections import defaultdict
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configuration from env
P31_MODE = os.getenv('P31_MODE', 'shadow')
P31_PORT = int(os.getenv('P31_PORT', '8062'))
P31_LOOP_SEC = int(os.getenv('P31_LOOP_SEC', '5'))

REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

EXECUTION_STREAM = os.getenv('EXECUTION_STREAM', 'quantum:stream:execution.result')
EFFICIENCY_STREAM = os.getenv('EFFICIENCY_STREAM', 'quantum:stream:capital.efficiency.decision')

P31_KEY_TTL_SEC = int(os.getenv('P31_KEY_TTL_SEC', '600'))
P31_STALE_SEC_P30 = int(os.getenv('P31_STALE_SEC_P30', '300'))
P31_STALE_SEC_P29 = int(os.getenv('P31_STALE_SEC_P29', '300'))

# EWMA parameters
P31_SCORE_ALPHA = float(os.getenv('P31_SCORE_ALPHA', '0.25'))
P31_VOL_ALPHA = float(os.getenv('P31_VOL_ALPHA', '0.20'))
P31_DD_ALPHA = float(os.getenv('P31_DD_ALPHA', '0.20'))

# Weights
P31_W_VOL = float(os.getenv('P31_W_VOL', '0.80'))
P31_W_DD = float(os.getenv('P31_W_DD', '1.20'))
P31_W_HOLD = float(os.getenv('P31_W_HOLD', '0.60'))

# Holding penalty
P31_HOLD_PENALTY_MAX_SEC = int(os.getenv('P31_HOLD_PENALTY_MAX_SEC', '3600'))

# Safety
P31_RUN_ONCE = os.getenv('P31_RUN_ONCE', 'false').lower() == 'true'
P31_MAX_EVENTS_PER_LOOP = int(os.getenv('P31_MAX_EVENTS_PER_LOOP', '200'))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
p31_loops_total = Counter('p31_loops_total', 'Total processing loops')
p31_loop_duration = Histogram('p31_loop_duration_seconds', 'Loop processing time')
p31_exec_events_total = Counter('p31_exec_events_total', 'Execution events processed', ['symbol'])
p31_inputs_missing_total = Counter('p31_inputs_missing_total', 'Missing inputs', ['symbol', 'which'])
p31_inputs_stale_total = Counter('p31_inputs_stale_total', 'Stale inputs', ['symbol', 'which'])
p31_efficiency_score = Gauge('p31_efficiency_score', 'Current efficiency score', ['symbol'])
p31_confidence = Gauge('p31_confidence', 'Current confidence', ['symbol'])
p31_written_total = Counter('p31_written_total', 'Outputs written', ['symbol'])
p31_lkg_used_total = Counter('p31_lkg_used_total', 'Last Known Good values used', ['symbol'])

# Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

# State per symbol
class SymbolState:
    def __init__(self):
        self.vol_ewma = 0.0
        self.dd_ewma = 0.0
        self.eff_ewma = 0.5  # Start neutral
        self.last_exec_ts = 0
        self.lkg_p30_perf = 0.5
        self.lkg_p29_target = 100.0

state = defaultdict(SymbolState)

def clamp01(x):
    return max(0.0, min(1.0, x))

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ewma(prev, new, alpha):
    return alpha * new + (1 - alpha) * prev

def get_p30_attribution(symbol, now_ts):
    """Fetch P3.0 attribution, check staleness"""
    key = f"quantum:alpha:attribution:{symbol}"
    try:
        data = redis_client.hgetall(key)
        if not data:
            p31_inputs_missing_total.labels(symbol=symbol, which='P30').inc()
            return None, True, "P30_MISSING"
        
        ts = int(data.get('ts', 0))
        if now_ts - ts > P31_STALE_SEC_P30:
            p31_inputs_stale_total.labels(symbol=symbol, which='P30').inc()
            return data, False, "P30_STALE"
        
        return data, False, None
    except Exception as e:
        logger.error(f"Error fetching P3.0 for {symbol}: {e}")
        p31_inputs_missing_total.labels(symbol=symbol, which='P30').inc()
        return None, True, "P30_ERROR"

def get_p29_allocation(symbol, now_ts):
    """Fetch P2.9 allocation target, check staleness"""
    key = f"quantum:allocation:target:{symbol}"
    try:
        data = redis_client.hgetall(key)
        if not data:
            p31_inputs_missing_total.labels(symbol=symbol, which='P29').inc()
            return None, True, "P29_MISSING"
        
        ts = int(data.get('ts', 0))
        if now_ts - ts > P31_STALE_SEC_P29:
            p31_inputs_stale_total.labels(symbol=symbol, which='P29').inc()
            return data, False, "P29_STALE"
        
        return data, False, None
    except Exception as e:
        logger.error(f"Error fetching P2.9 for {symbol}: {e}")
        p31_inputs_missing_total.labels(symbol=symbol, which='P29').inc()
        return None, True, "P29_ERROR"

def normalize_performance(pf, pf_range='0_1'):
    """Normalize performance_factor to [0, 1]"""
    if pf_range == '-1_1':
        return (pf + 1.0) / 2.0
    return clamp01(pf)

def compute_efficiency(symbol, now_ts):
    """Compute efficiency score with fail-open behavior"""
    st = state[symbol]
    
    # Fetch inputs
    p30_data, p30_missing, p30_flag = get_p30_attribution(symbol, now_ts)
    p29_data, p29_missing, p29_flag = get_p29_allocation(symbol, now_ts)
    
    # Collect stale flags
    stale_flags = []
    if p30_flag:
        stale_flags.append(p30_flag)
    if p29_flag:
        stale_flags.append(p29_flag)
    
    # Fail-open: use LKG if missing/stale
    if p30_missing:
        pf = st.lkg_p30_perf
        p30_conf = 0.0
        p31_lkg_used_total.labels(symbol=symbol).inc()
        logger.warning(f"{symbol}: P3.0 missing, using LKG perf={pf:.2f}")
    else:
        pf = float(p30_data.get('performance_factor', 0.5))
        p30_conf = float(p30_data.get('confidence', 0.0))
        st.lkg_p30_perf = pf  # Update LKG
    
    if p29_missing:
        target_usd = st.lkg_p29_target
        p29_conf = 0.0
        p31_lkg_used_total.labels(symbol=symbol).inc()
        logger.warning(f"{symbol}: P2.9 missing, using LKG target=${target_usd:.0f}")
    else:
        target_usd = float(p29_data.get('target_usd', 100.0))
        p29_conf = float(p29_data.get('confidence', 0.0))
        st.lkg_p29_target = target_usd  # Update LKG
    
    # Normalize performance to [0, 1]
    pf01 = normalize_performance(pf, pf_range='0_1')
    
    # Volatility and drawdown (normalized by target)
    target_safe = max(target_usd, 1.0)
    vol_norm = st.vol_ewma / target_safe
    dd_norm = st.dd_ewma / target_safe
    
    # Holding time penalty
    silence_sec = now_ts - st.last_exec_ts if st.last_exec_ts > 0 else 0
    hold_penalty = clamp01(silence_sec / P31_HOLD_PENALTY_MAX_SEC)
    
    # Efficiency formula
    denom = 1.0 + (P31_W_VOL * vol_norm) + (P31_W_DD * dd_norm) + (P31_W_HOLD * hold_penalty)
    raw = pf01 / denom
    score = clamp01(raw)
    
    # EWMA smoothing
    eff_ewma = ewma(st.eff_ewma, score, P31_SCORE_ALPHA)
    st.eff_ewma = eff_ewma
    
    # Capital pressure
    if eff_ewma >= 0.65 and p30_conf >= 0.5:
        pressure = "INCREASE"
    elif eff_ewma <= 0.35 or dd_norm > 0.5:
        pressure = "DECREASE"
    else:
        pressure = "HOLD"
    
    # Reallocation weight
    realloc_weight = clamp((eff_ewma - 0.5) * 2.0, -1.0, 1.0)
    
    # Overall confidence (min of input confidences, degraded by stale flags)
    conf = min(p30_conf, p29_conf)
    if stale_flags:
        conf *= 0.5  # Degrade confidence for stale inputs
    
    # Inputs OK flag
    inputs_ok = 1 if not (p30_missing or p29_missing) else 0
    
    return {
        'efficiency_score': score,
        'efficiency_raw': raw,
        'efficiency_ewma': eff_ewma,
        'capital_pressure': pressure,
        'reallocation_weight': realloc_weight * conf,
        'confidence': conf,
        'inputs_ok': inputs_ok,
        'stale_flags': ','.join(stale_flags) if stale_flags else '',
        'p30_perf': pf,
        'p30_conf': p30_conf,
        'p29_target_usd': target_usd,
        'volatility_ewma': st.vol_ewma,
        'dd_proxy': st.dd_ewma,
        'hold_time_penalty': hold_penalty,
        'ts': now_ts,
        'mode': P31_MODE,
        'version': 'p31_v1'
    }

def write_efficiency_output(symbol, metrics):
    """Write efficiency hash and decision stream"""
    # Always write to stream (shadow + enforce)
    try:
        stream_msg = {
            'symbol': symbol,
            'efficiency_score': str(metrics['efficiency_score']),
            'capital_pressure': metrics['capital_pressure'],
            'reallocation_weight': str(metrics['reallocation_weight']),
            'confidence': str(metrics['confidence']),
            'stale_flags': metrics['stale_flags'],
            'ts': str(metrics['ts']),
            'mode': metrics['mode'],
            'trace': f"p30_conf={metrics['p30_conf']:.2f},p29_tgt={metrics['p29_target_usd']:.0f},inputs_ok={metrics['inputs_ok']}"
        }
        redis_client.xadd(EFFICIENCY_STREAM, stream_msg, maxlen=10000, approximate=True)
    except Exception as e:
        logger.error(f"Failed to write decision stream for {symbol}: {e}")
    
    # Write hash only in enforce mode (shadow skips this)
    if P31_MODE == 'enforce':
        try:
            key = f"quantum:capital:efficiency:{symbol}"
            # Convert all values to strings for HSET
            hash_data = {k: str(v) for k, v in metrics.items()}
            redis_client.hset(key, mapping=hash_data)
            redis_client.expire(key, P31_KEY_TTL_SEC)
            p31_written_total.labels(symbol=symbol).inc()
        except Exception as e:
            logger.error(f"Failed to write hash for {symbol}: {e}")
    
    # Update Prometheus gauges
    p31_efficiency_score.labels(symbol=symbol).set(metrics['efficiency_score'])
    p31_confidence.labels(symbol=symbol).set(metrics['confidence'])

def process_execution_events():
    """Read execution stream and update per-symbol state"""
    last_id_key = "quantum:p31:last_id"
    last_id = redis_client.get(last_id_key) or '0'
    
    try:
        messages = redis_client.xread(
            {EXECUTION_STREAM: last_id},
            count=P31_MAX_EVENTS_PER_LOOP,
            block=0
        )
        
        if not messages:
            return
        
        for stream, entries in messages:
            for msg_id, data in entries:
                last_id = msg_id
                
                symbol = data.get('symbol')
                if not symbol:
                    continue
                
                # Extract PnL (try multiple field names)
                pnl_usd = 0.0
                for field in ['pnl_usd', 'realized_pnl', 'pnl']:
                    if field in data:
                        try:
                            pnl_usd = float(data[field])
                            break
                        except:
                            pass
                
                # Update timestamp
                ts = int(data.get('ts', data.get('timestamp', int(time.time()))))
                
                # Update state
                st = state[symbol]
                st.last_exec_ts = ts
                
                # Update volatility EWMA
                st.vol_ewma = ewma(st.vol_ewma, abs(pnl_usd), P31_VOL_ALPHA)
                
                # Update drawdown EWMA (only negative PnL)
                neg_pnl = max(-pnl_usd, 0.0)
                st.dd_ewma = ewma(st.dd_ewma, neg_pnl, P31_DD_ALPHA)
                
                p31_exec_events_total.labels(symbol=symbol).inc()
        
        # Save last processed ID
        redis_client.set(last_id_key, last_id, ex=86400)
    
    except Exception as e:
        logger.error(f"Error processing execution stream: {e}")

def process_loop():
    """Main processing loop"""
    with p31_loop_duration.time():
        now_ts = int(time.time())
        
        # Process execution events to update state
        process_execution_events()
        
        # Compute efficiency for all active symbols
        symbols = set(state.keys())
        
        # Also check for symbols with P3.0/P2.9 data
        try:
            for key in redis_client.scan_iter("quantum:alpha:attribution:*", count=100):
                symbol = key.split(':')[-1]
                symbols.add(symbol)
        except:
            pass
        
        for symbol in symbols:
            try:
                metrics = compute_efficiency(symbol, now_ts)
                write_efficiency_output(symbol, metrics)
                
                logger.info(
                    f"{symbol}: eff={metrics['efficiency_score']:.3f} "
                    f"ewma={metrics['efficiency_ewma']:.3f} "
                    f"pressure={metrics['capital_pressure']} "
                    f"conf={metrics['confidence']:.2f} "
                    f"flags={metrics['stale_flags'] or 'OK'}"
                )
            except Exception as e:
                logger.error(f"Error computing efficiency for {symbol}: {e}")
        
        p31_loops_total.inc()

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            health = {
                'status': 'healthy',
                'service': 'p31_capital_efficiency',
                'mode': P31_MODE,
                'version': 'p31_v1'
            }
            self.wfile.write(json.dumps(health).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress default logging

def main():
    logger.info(f"P3.1 Capital Efficiency Brain starting...")
    logger.info(f"  Mode: {P31_MODE}")
    logger.info(f"  Port: {P31_PORT}")
    logger.info(f"  Loop: {P31_LOOP_SEC}s")
    logger.info(f"  Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
    logger.info(f"  Execution stream: {EXECUTION_STREAM}")
    logger.info(f"  Efficiency stream: {EFFICIENCY_STREAM}")
    
    # Start Prometheus metrics server
    start_http_server(P31_PORT)
    logger.info(f"Metrics server started on :{P31_PORT}/metrics")
    
    # Start health check endpoint (on a different port to avoid conflict)
    import threading
    health_server = HTTPServer(('0.0.0.0', P31_PORT + 1), HealthHandler)
    health_thread = threading.Thread(target=health_server.serve_forever, daemon=True)
    health_thread.start()
    logger.info(f"Health endpoint on :{P31_PORT + 1}/health")
    
    # Main loop
    try:
        while True:
            process_loop()
            
            if P31_RUN_ONCE:
                logger.info("P31_RUN_ONCE=true, exiting")
                break
            
            time.sleep(P31_LOOP_SEC)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
