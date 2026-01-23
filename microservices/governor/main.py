#!/usr/bin/env python3
"""
P3.2 Governor - Fund-Grade Limits + Auto-Disarm

Enforces rate limits and safety gates for Apply Layer execution.
Writes Redis permit keys that Apply Layer checks before executing trades.
Can automatically disarm the system (force dry_run) under unsafe conditions.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import logging
import redis
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict
from prometheus_client import Counter, Gauge, start_http_server

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
METRIC_ALLOW = Counter('quantum_govern_allow_total', 'Plans allowed', ['symbol'])
METRIC_BLOCK = Counter('quantum_govern_block_total', 'Plans blocked', ['symbol', 'reason'])
METRIC_DISARM = Counter('quantum_govern_disarm_total', 'Auto-disarm events', ['reason'])
METRIC_EXEC_HOUR = Gauge('quantum_govern_exec_count_hour', 'Executions in last hour', ['symbol'])
METRIC_EXEC_5MIN = Gauge('quantum_govern_exec_count_5min', 'Executions in last 5min', ['symbol'])

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB = int(os.getenv('REDIS_DB', '0'))
    
    # Rate limits
    MAX_EXEC_PER_HOUR = int(os.getenv('GOV_MAX_EXEC_PER_HOUR', '3'))
    MAX_EXEC_PER_5MIN = int(os.getenv('GOV_MAX_EXEC_PER_5MIN', '2'))
    MAX_REDUCE_NOTIONAL_PER_DAY_USD = float(os.getenv('GOV_MAX_REDUCE_NOTIONAL_PER_DAY_USD', '5000'))
    MAX_REDUCE_QTY_PER_DAY = float(os.getenv('GOV_MAX_REDUCE_QTY_PER_DAY', '0.02'))
    
    # Kill score gate
    KILL_SCORE_CRITICAL = float(os.getenv('GOV_KILL_SCORE_CRITICAL', '0.8'))
    
    # Auto-disarm
    ENABLE_AUTO_DISARM = os.getenv('GOV_ENABLE_AUTO_DISARM', 'true').lower() == 'true'
    DISARM_ON_ERROR_COUNT = int(os.getenv('GOV_DISARM_ON_ERROR_COUNT', '5'))
    DISARM_ON_BURST_BREACH = os.getenv('GOV_DISARM_ON_BURST_BREACH', 'true').lower() == 'true'
    
    # Apply Layer config path (for disarm action)
    APPLY_CONFIG_PATH = os.getenv('APPLY_CONFIG_PATH', '/etc/quantum/apply-layer.env')
    
    # Metrics
    METRICS_PORT = int(os.getenv('METRICS_PORT', '8044'))
    
    # Streams
    STREAM_PLANS = os.getenv('STREAM_PLANS', 'quantum:stream:apply.plan')
    STREAM_RESULTS = os.getenv('STREAM_RESULTS', 'quantum:stream:apply.result')
    STREAM_EVENTS = os.getenv('STREAM_EVENTS', 'quantum:stream:governor.events')

# ============================================================================
# GOVERNOR CORE
# ============================================================================
class Governor:
    def __init__(self, config):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Execution tracking (in-memory + Redis backup)
        self.exec_history = defaultdict(list)  # {symbol: [timestamp, ...]}
        self.error_count = 0
        self.last_disarm_check = time.time()
        
        logger.info("Governor initialized")
        logger.info(f"Max exec/hour: {config.MAX_EXEC_PER_HOUR}, Max exec/5min: {config.MAX_EXEC_PER_5MIN}")
        logger.info(f"Auto-disarm: {config.ENABLE_AUTO_DISARM}, Kill score critical: {config.KILL_SCORE_CRITICAL}")
    
    def run(self):
        """Main loop: watch plans, evaluate, issue permits"""
        logger.info("Governor starting main loop")
        last_id = '$'  # Start from latest
        
        while True:
            try:
                # Read new plans from stream
                messages = self.redis.xread(
                    {self.config.STREAM_PLANS: last_id},
                    count=10,
                    block=1000  # 1s timeout
                )
                
                if not messages:
                    continue
                
                for stream_name, stream_messages in messages:
                    for message_id, data in stream_messages:
                        last_id = message_id
                        self._evaluate_plan(message_id, data)
                
                # Periodic tasks
                self._update_metrics()
                self._check_auto_disarm()
                
            except KeyboardInterrupt:
                logger.info("Governor shutting down")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.error_count += 1
                time.sleep(1)
    
    def _evaluate_plan(self, plan_id, data):
        """Evaluate a plan and issue permit or block"""
        try:
            symbol = data.get('symbol', 'UNKNOWN')
            action = data.get('action', 'UNKNOWN')
            kill_score = float(data.get('kill_score', '0'))
            close_qty = float(data.get('close_qty', '0'))
            price = float(data.get('price', '0')) if data.get('price') else None
            
            logger.info(f"{symbol}: Evaluating plan {plan_id[:8]} (action={action}, kill_score={kill_score:.3f})")
            
            # Gate 1: Kill score critical threshold
            if kill_score >= self.config.KILL_SCORE_CRITICAL:
                if action != 'CLOSE':
                    self._block_plan(plan_id, symbol, 'kill_score_critical_non_close')
                    return
            
            # Gate 2: Hourly rate limit
            recent_hour = self._get_exec_count_window(symbol, hours=1)
            if recent_hour >= self.config.MAX_EXEC_PER_HOUR:
                self._block_plan(plan_id, symbol, 'hourly_limit_exceeded')
                return
            
            # Gate 3: Burst protection (5min)
            recent_5min = self._get_exec_count_window(symbol, minutes=5)
            if recent_5min >= self.config.MAX_EXEC_PER_5MIN:
                self._block_plan(plan_id, symbol, 'burst_limit_exceeded')
                
                # Trigger auto-disarm if configured
                if self.config.ENABLE_AUTO_DISARM and self.config.DISARM_ON_BURST_BREACH:
                    self._trigger_disarm('burst_limit_breach', {
                        'symbol': symbol,
                        'exec_5min': recent_5min,
                        'limit': self.config.MAX_EXEC_PER_5MIN
                    })
                return
            
            # Gate 4: Daily notional/qty limit
            if not self._check_daily_limit(symbol, close_qty, price):
                self._block_plan(plan_id, symbol, 'daily_limit_exceeded')
                return
            
            # All gates passed - issue permit
            self._issue_permit(plan_id, symbol)
            
        except Exception as e:
            logger.error(f"Error evaluating plan {plan_id}: {e}", exc_info=True)
            self.error_count += 1
            # Fail closed: do not issue permit
            self._block_plan(plan_id, symbol, 'evaluation_error')
    
    def _block_plan(self, plan_id, symbol, reason):
        """Block a plan (no permit issued)"""
        logger.warning(f"{symbol}: BLOCKED plan {plan_id[:8]} - {reason}")
        METRIC_BLOCK.labels(symbol=symbol, reason=reason).inc()
        
        # Store block record
        block_key = f"quantum:governor:block:{plan_id}"
        self.redis.setex(block_key, 3600, json.dumps({
            'reason': reason,
            'timestamp': time.time(),
            'symbol': symbol
        }))
    
    def _issue_permit(self, plan_id, symbol):
        """Issue execution permit for a plan"""
        logger.info(f"{symbol}: ALLOW plan {plan_id[:8]} (permit issued)")
        
        # Write permit key (Apply Layer will check this)
        permit_key = f"quantum:permit:{plan_id}"
        self.redis.setex(permit_key, 3600, json.dumps({
            'granted': True,
            'timestamp': time.time(),
            'symbol': symbol
        }))
        
        # Track execution (assume will execute - apply layer confirms later)
        self.exec_history[symbol].append(time.time())
        self._trim_exec_history(symbol)
        
        METRIC_ALLOW.labels(symbol=symbol).inc()
        
        # Store in Redis for persistence across restarts
        exec_key = f"quantum:governor:exec:{symbol}"
        self.redis.lpush(exec_key, str(time.time()))
        self.redis.ltrim(exec_key, 0, 99)  # Keep last 100
        self.redis.expire(exec_key, 86400)  # 24h
    
    def _get_exec_count_window(self, symbol, hours=0, minutes=0):
        """Count executions in time window"""
        now = time.time()
        window = hours * 3600 + minutes * 60
        cutoff = now - window
        
        # Load from Redis if in-memory cache is empty
        if not self.exec_history[symbol]:
            exec_key = f"quantum:governor:exec:{symbol}"
            timestamps = self.redis.lrange(exec_key, 0, -1)
            self.exec_history[symbol] = [float(ts) for ts in timestamps if float(ts) > cutoff]
        
        # Count recent
        recent = [ts for ts in self.exec_history[symbol] if ts > cutoff]
        return len(recent)
    
    def _trim_exec_history(self, symbol):
        """Remove old timestamps from in-memory cache"""
        cutoff = time.time() - 3600  # Keep last hour in memory
        self.exec_history[symbol] = [ts for ts in self.exec_history[symbol] if ts > cutoff]
    
    def _check_daily_limit(self, symbol, qty, price):
        """Check daily notional or quantity limit"""
        # Get today's executions
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"quantum:governor:daily:{symbol}:{today}"
        
        current_total = float(self.redis.get(daily_key) or 0)
        
        # Calculate new total
        if price and price > 0:
            # Notional-based
            notional = qty * price
            new_total = current_total + notional
            limit = self.config.MAX_REDUCE_NOTIONAL_PER_DAY_USD
            if new_total > limit:
                logger.warning(f"{symbol}: Daily notional limit exceeded ({new_total:.2f} > {limit})")
                return False
            # Update
            self.redis.setex(daily_key, 86400, str(new_total))
        else:
            # Qty-based fallback
            new_total = current_total + qty
            limit = self.config.MAX_REDUCE_QTY_PER_DAY
            if new_total > limit:
                logger.warning(f"{symbol}: Daily qty limit exceeded ({new_total:.4f} > {limit})")
                return False
            # Update
            self.redis.setex(daily_key, 86400, str(new_total))
        
        return True
    
    def _trigger_disarm(self, reason, context):
        """Force Apply Layer to dry_run mode (auto-disarm)"""
        # Idempotency: check if already disarmed today
        today = datetime.utcnow().strftime('%Y-%m-%d')
        disarm_key = f"quantum:governor:disarm:{today}"
        
        if self.redis.exists(disarm_key):
            logger.info(f"Disarm already triggered today ({reason}), skipping")
            return
        
        logger.critical(f"AUTO-DISARM TRIGGERED: {reason}")
        logger.info(f"Context: {json.dumps(context)}")
        
        try:
            # Method: Set APPLY_MODE=dry_run in config
            config_path = self.config.APPLY_CONFIG_PATH
            
            # Backup config
            backup_path = f"{config_path}.bak.{int(time.time())}"
            subprocess.run(['cp', config_path, backup_path], check=True)
            logger.info(f"Backed up config to {backup_path}")
            
            # Update config
            subprocess.run([
                'sed', '-i', 's/^APPLY_MODE=.*/APPLY_MODE=dry_run/',
                config_path
            ], check=True)
            logger.info(f"Set APPLY_MODE=dry_run in {config_path}")
            
            # Restart Apply Layer
            subprocess.run(['systemctl', 'restart', 'quantum-apply-layer'], check=True)
            logger.info("Restarted quantum-apply-layer.service")
            
            # Mark disarm done (24h TTL)
            self.redis.setex(disarm_key, 86400, json.dumps({
                'reason': reason,
                'context': context,
                'timestamp': time.time()
            }))
            
            # Emit event to stream
            self.redis.xadd(self.config.STREAM_EVENTS, {
                'event': 'AUTO_DISARM',
                'reason': reason,
                'context': json.dumps(context),
                'timestamp': time.time(),
                'action_taken': 'APPLY_MODE=dry_run + restart'
            })
            
            METRIC_DISARM.labels(reason=reason).inc()
            logger.critical("AUTO-DISARM COMPLETE - System is now in dry_run mode")
            
        except Exception as e:
            logger.error(f"Failed to execute disarm: {e}", exc_info=True)
            # Still mark as attempted to avoid spam
            self.redis.setex(disarm_key, 3600, json.dumps({
                'reason': reason,
                'error': str(e),
                'timestamp': time.time()
            }))
    
    def _check_auto_disarm(self):
        """Periodic check for auto-disarm conditions"""
        if not self.config.ENABLE_AUTO_DISARM:
            return
        
        now = time.time()
        if now - self.last_disarm_check < 60:  # Check every 60s
            return
        
        self.last_disarm_check = now
        
        # Check error count
        if self.error_count >= self.config.DISARM_ON_ERROR_COUNT:
            self._trigger_disarm('repeated_errors', {
                'error_count': self.error_count,
                'threshold': self.config.DISARM_ON_ERROR_COUNT
            })
            self.error_count = 0  # Reset after disarm
    
    def _update_metrics(self):
        """Update Prometheus metrics"""
        for symbol in self.exec_history.keys():
            count_hour = self._get_exec_count_window(symbol, hours=1)
            count_5min = self._get_exec_count_window(symbol, minutes=5)
            METRIC_EXEC_HOUR.labels(symbol=symbol).set(count_hour)
            METRIC_EXEC_5MIN.labels(symbol=symbol).set(count_5min)

# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=== P3.2 GOVERNOR STARTING ===")
    logger.info(f"Metrics port: {Config.METRICS_PORT}")
    
    # Start Prometheus metrics server
    start_http_server(Config.METRICS_PORT)
    logger.info(f"Metrics server started on port {Config.METRICS_PORT}")
    
    # Initialize Governor
    governor = Governor(Config)
    
    # Run main loop
    try:
        governor.run()
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Governor stopped")

if __name__ == '__main__':
    main()
