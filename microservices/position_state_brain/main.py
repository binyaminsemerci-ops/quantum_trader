#!/usr/bin/env python3
"""
P3.3 Position State Brain - Exchange↔Redis↔Ledger Sanity Check

Reconciles 3 sources of truth before allowing Apply Layer to execute:
1. Exchange truth (Binance testnet: positionAmt, side, entryPrice, markPrice)
2. Redis truth (harvest proposals, apply plans/results)
3. Internal ledger (shadow ledger from executed orders)

Outputs:
- Position snapshots: quantum:position:snapshot:<symbol>
- Ledger state: quantum:position:ledger:<symbol>
- Execution permits: quantum:permit:p33:<plan_id> (60s TTL)

Decisions:
- OK_TO_EXECUTE: safe_close_qty computed, permit granted
- BLOCK: reason_code + deny permit
- RECONCILE_REQUIRED: side/qty mismatch, deny permit
"""

import os
import sys
import time
import json
import hmac
import hashlib
import urllib.request
import urllib.parse
import logging
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import redis
except ImportError:
    print("ERROR: redis-py not installed. Install with: pip install redis")
    sys.exit(1)

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("WARN: prometheus_client not installed, metrics disabled")

logging.basicConfig(
    level=os.getenv("P33_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """P3.3 Configuration"""
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    
    # Binance testnet credentials
    BINANCE_TESTNET_API_KEY: str = os.getenv("BINANCE_TESTNET_API_KEY", "")
    BINANCE_TESTNET_API_SECRET: str = os.getenv("BINANCE_TESTNET_API_SECRET", "")
    
    # Symbol allowlist
    ALLOWLIST: str = os.getenv("P33_ALLOWLIST", os.getenv("APPLY_ALLOWLIST", "BTCUSDT"))
    
    # Polling interval
    POLL_INTERVAL: int = int(os.getenv("P33_POLL_SEC", "5"))
    
    # Sanity check thresholds
    STALE_THRESHOLD_SEC: int = int(os.getenv("P33_STALE_THRESHOLD_SEC", "10"))
    COOLDOWN_SEC: int = int(os.getenv("P33_COOLDOWN_SEC", "15"))
    QTY_TOLERANCE_PCT: float = float(os.getenv("P33_QTY_TOLERANCE_PCT", "1.0"))
    
    # Permit TTL
    PERMIT_TTL: int = int(os.getenv("P33_PERMIT_TTL", "60"))
    
    # Metrics port
    METRICS_PORT: int = int(os.getenv("P33_METRICS_PORT", "8045"))


class BinanceTestnetClient:
    """Minimal Binance Futures Testnet client for position data"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://testnet.binancefuture.com"
    
    def _sign_request(self, params: Dict[str, Any]) -> str:
        """Sign request with HMAC SHA256"""
        query_string = urllib.parse.urlencode(params)
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Any:
        """Make HTTP request to Binance API"""
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign_request(params)
        
        url = f"{self.base_url}{endpoint}"
        if params:
            url += f"?{urllib.parse.urlencode(params)}"
        
        req = urllib.request.Request(url, method=method)
        req.add_header('X-MBX-APIKEY', self.api_key)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            logger.error(f"Binance API error {e.code}: {error_body}")
            raise
        except Exception as e:
            logger.error(f"Binance API request failed: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for symbol"""
        try:
            positions = self._request('GET', '/fapi/v2/positionRisk', signed=True)
            for pos in positions:
                if pos['symbol'] == symbol:
                    return {
                        'symbol': symbol,
                        'positionAmt': float(pos['positionAmt']),
                        'entryPrice': float(pos['entryPrice']) if pos['entryPrice'] else 0.0,
                        'markPrice': float(pos['markPrice']) if pos['markPrice'] else 0.0,
                        'unRealizedProfit': float(pos['unRealizedProfit']),
                        'leverage': int(pos['leverage']),
                        'side': 'LONG' if float(pos['positionAmt']) > 0 else ('SHORT' if float(pos['positionAmt']) < 0 else 'NONE')
                    }
            return None
        except Exception as e:
            logger.error(f"{symbol}: Failed to get position: {e}")
            return None
    
    def get_mark_price(self, symbol: str) -> Optional[float]:
        """Get current mark price"""
        try:
            result = self._request('GET', '/fapi/v1/premiumIndex', params={'symbol': symbol})
            return float(result['markPrice'])
        except Exception as e:
            logger.error(f"{symbol}: Failed to get mark price: {e}")
            return None


class PositionStateBrain:
    """P3.3 Position State Brain - Exchange↔Redis↔Ledger Reconciliation"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Redis connection
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Binance client
        self.binance_client = None
        if config.BINANCE_TESTNET_API_KEY and config.BINANCE_TESTNET_API_SECRET:
            self.binance_client = BinanceTestnetClient(
                config.BINANCE_TESTNET_API_KEY,
                config.BINANCE_TESTNET_API_SECRET
            )
            logger.info("Binance testnet client initialized")
        else:
            logger.error("Missing Binance credentials - P3.3 cannot function")
            sys.exit(1)
        
        # Parse allowlist
        self.symbols = [s.strip() for s in config.ALLOWLIST.split(',')]
        logger.info(f"Monitoring symbols: {self.symbols}")
        
        # Metrics
        if PROMETHEUS_AVAILABLE:
            self.metric_permit_allow = Counter('p33_permit_allow_total', 'P3.3 permits granted', ['symbol'])
            self.metric_permit_deny = Counter('p33_permit_deny_total', 'P3.3 permits denied', ['symbol', 'reason'])
            self.metric_snapshot_age = Gauge('p33_snapshot_age_seconds', 'Exchange snapshot age', ['symbol'])
            self.metric_ledger_amt = Gauge('p33_ledger_position_amt', 'Ledger position amount', ['symbol'])
            self.metric_exchange_amt = Gauge('p33_exchange_position_amt', 'Exchange position amount', ['symbol'])
            
            # Start metrics server
            start_http_server(config.METRICS_PORT)
            logger.info(f"Metrics server started on port {config.METRICS_PORT}")
    
    def update_exchange_snapshot(self, symbol: str) -> bool:
        """Fetch and store exchange position snapshot"""
        position = self.binance_client.get_position(symbol)
        
        if position is None:
            logger.warning(f"{symbol}: Failed to fetch exchange position")
            return False
        
        snapshot_key = f"quantum:position:snapshot:{symbol}"
        snapshot_data = {
            'position_amt': position['positionAmt'],
            'side': position['side'],
            'entry_price': position['entryPrice'],
            'mark_price': position['markPrice'],
            'unrealized_pnl': position['unRealizedProfit'],
            'leverage': position['leverage'],
            'ts_epoch': int(time.time()),
            'source': 'binance_testnet'
        }
        
        self.redis.hset(snapshot_key, mapping=snapshot_data)
        self.redis.expire(snapshot_key, 3600)  # 1 hour TTL
        
        if PROMETHEUS_AVAILABLE:
            self.metric_exchange_amt.labels(symbol=symbol).set(position['positionAmt'])
        
        logger.debug(f"{symbol}: Snapshot updated (amt={position['positionAmt']}, side={position['side']})")
        return True
    
    def get_exchange_snapshot(self, symbol: str) -> Optional[Dict]:
        """Get cached exchange snapshot"""
        snapshot_key = f"quantum:position:snapshot:{symbol}"
        data = self.redis.hgetall(snapshot_key)
        
        if not data:
            return None
        
        return {
            'position_amt': float(data.get('position_amt', 0)),
            'side': data.get('side', 'NONE'),
            'entry_price': float(data.get('entry_price', 0)),
            'mark_price': float(data.get('mark_price', 0)),
            'ts_epoch': int(data.get('ts_epoch', 0)),
            'source': data.get('source', 'unknown')
        }
    
    def update_ledger(self, symbol: str, result: Dict):
        """Update internal ledger from executed result"""
        if not result.get('executed'):
            return
        
        ledger_key = f"quantum:position:ledger:{symbol}"
        
        # Extract order info from result
        steps_results = result.get('steps_results', [])
        if not steps_results:
            return
        
        last_step = steps_results[-1]
        order_id = last_step.get('order_id', '')
        executed_qty = float(last_step.get('executed_qty', 0))
        side = last_step.get('side', 'UNKNOWN')
        
        # Update ledger
        ledger_data = {
            'last_order_id': order_id,
            'last_result_plan_id': result['plan_id'],
            'last_executed_qty': executed_qty,
            'last_side': side,
            'updated_at': int(time.time())
        }
        
        # Read current ledger amt, update based on execution
        current_data = self.redis.hgetall(ledger_key)
        current_amt = float(current_data.get('last_known_amt', 0)) if current_data else 0.0
        
        # Reduce position (reduceOnly orders)
        if side == 'SELL' and current_amt > 0:
            new_amt = max(0, current_amt - executed_qty)
        elif side == 'BUY' and current_amt < 0:
            new_amt = min(0, current_amt + executed_qty)
        else:
            new_amt = current_amt  # No change if unexpected
        
        ledger_data['last_known_amt'] = new_amt
        
        self.redis.hset(ledger_key, mapping=ledger_data)
        self.redis.expire(ledger_key, 86400)  # 24 hour TTL
        
        if PROMETHEUS_AVAILABLE:
            self.metric_ledger_amt.labels(symbol=symbol).set(new_amt)
        
        logger.info(f"{symbol}: Ledger updated (order={order_id}, new_amt={new_amt:.4f})")
    
    def get_ledger(self, symbol: str) -> Optional[Dict]:
        """Get internal ledger state"""
        ledger_key = f"quantum:position:ledger:{symbol}"
        data = self.redis.hgetall(ledger_key)
        
        if not data:
            return None
        
        return {
            'last_known_amt': float(data.get('last_known_amt', 0)),
            'last_order_id': data.get('last_order_id', ''),
            'last_result_plan_id': data.get('last_result_plan_id', ''),
            'updated_at': int(data.get('updated_at', 0))
        }
    
    def evaluate_plan(self, plan_id: str, symbol: str, data: Dict) -> Dict:
        """Evaluate plan and issue P3.3 permit with sanity checks"""
        decision = data.get('decision')
        action = data.get('action', '')
        
        # Only evaluate EXECUTE decisions
        if decision != 'EXECUTE':
            logger.debug(f"{symbol}: Plan {plan_id[:8]} decision={decision}, skipping P3.3")
            return {'evaluated': False, 'reason': 'not_execute_decision'}
        
        # Get exchange snapshot
        snapshot = self.get_exchange_snapshot(symbol)
        
        # Check 1: Stale snapshot
        if not snapshot:
            return self._deny_permit(plan_id, symbol, 'no_exchange_snapshot', {})
        
        age = int(time.time()) - snapshot['ts_epoch']
        if age > self.config.STALE_THRESHOLD_SEC:
            if PROMETHEUS_AVAILABLE:
                self.metric_snapshot_age.labels(symbol=symbol).set(age)
            return self._deny_permit(plan_id, symbol, 'stale_exchange_state', {'age_seconds': age})
        
        # Check 2: No position on exchange
        exchange_amt = snapshot['position_amt']
        if abs(exchange_amt) < 0.0001:
            return self._deny_permit(plan_id, symbol, 'no_position', {'exchange_amt': exchange_amt})
        
        # Get ledger state
        ledger = self.get_ledger(symbol)
        
        # Check 3: Side mismatch (if ledger exists)
        if ledger:
            ledger_amt = ledger['last_known_amt']
            exchange_side = snapshot['side']
            ledger_side = 'LONG' if ledger_amt > 0 else ('SHORT' if ledger_amt < 0 else 'NONE')
            
            if exchange_side != 'NONE' and ledger_side != 'NONE' and exchange_side != ledger_side:
                return self._deny_permit(plan_id, symbol, 'reconcile_required_side_mismatch', {
                    'exchange_side': exchange_side,
                    'ledger_side': ledger_side
                })
            
            # Check 4: Qty mismatch beyond tolerance
            qty_diff = abs(abs(exchange_amt) - abs(ledger_amt))
            tolerance = max(0.001, abs(exchange_amt) * (self.config.QTY_TOLERANCE_PCT / 100))
            
            if qty_diff > tolerance:
                return self._deny_permit(plan_id, symbol, 'reconcile_required_qty_mismatch', {
                    'exchange_amt': exchange_amt,
                    'ledger_amt': ledger_amt,
                    'diff': qty_diff,
                    'tolerance': tolerance
                })
        
        # Check 5: Cooldown (order-in-flight guard)
        if ledger:
            time_since_last = int(time.time()) - ledger['updated_at']
            if time_since_last < self.config.COOLDOWN_SEC:
                return self._deny_permit(plan_id, symbol, 'cooldown_in_flight', {
                    'seconds_since_last': time_since_last,
                    'cooldown_sec': self.config.COOLDOWN_SEC
                })
        
        # Check 6: Compute safe_close_qty
        # Parse requested qty from action
        if action == 'FULL_CLOSE_PROPOSED':
            requested_qty = abs(exchange_amt)
        elif action == 'PARTIAL_75':
            requested_qty = abs(exchange_amt) * 0.75
        elif action == 'PARTIAL_50':
            requested_qty = abs(exchange_amt) * 0.50
        else:
            requested_qty = abs(exchange_amt)  # Default to full
        
        # Clamp to actual position
        safe_close_qty = min(requested_qty, abs(exchange_amt))
        
        # Round to step size (BTCUSDT = 0.001)
        step_size = 0.001
        safe_close_qty = round(safe_close_qty / step_size) * step_size
        
        # Grant permit
        return self._grant_permit(plan_id, symbol, safe_close_qty, exchange_amt, ledger['last_known_amt'] if ledger else 0.0)
    
    def _grant_permit(self, plan_id: str, symbol: str, safe_close_qty: float, exchange_amt: float, ledger_amt: float) -> Dict:
        """Grant P3.3 execution permit"""
        permit_key = f"quantum:permit:p33:{plan_id}"
        
        permit_data = {
            'allow': True,
            'symbol': symbol,
            'safe_close_qty': safe_close_qty,
            'exchange_position_amt': exchange_amt,
            'ledger_amt': ledger_amt,
            'created_at': time.time(),
            'reason': 'sanity_checks_passed'
        }
        
        self.redis.setex(permit_key, self.config.PERMIT_TTL, json.dumps(permit_data))
        
        if PROMETHEUS_AVAILABLE:
            self.metric_permit_allow.labels(symbol=symbol).inc()
        
        logger.info(f"{symbol}: P3.3 ALLOW plan {plan_id[:8]} (safe_qty={safe_close_qty:.4f}, exchange_amt={exchange_amt:.4f})")
        
        return {'evaluated': True, 'allow': True, 'safe_close_qty': safe_close_qty}
    
    def _deny_permit(self, plan_id: str, symbol: str, reason: str, context: Dict) -> Dict:
        """Deny P3.3 execution permit"""
        permit_key = f"quantum:permit:p33:{plan_id}"
        
        permit_data = {
            'allow': False,
            'symbol': symbol,
            'reason': reason,
            'context': context,
            'created_at': time.time()
        }
        
        self.redis.setex(permit_key, self.config.PERMIT_TTL, json.dumps(permit_data))
        
        if PROMETHEUS_AVAILABLE:
            self.metric_permit_deny.labels(symbol=symbol, reason=reason).inc()
        
        logger.warning(f"{symbol}: P3.3 DENY plan {plan_id[:8]} reason={reason} context={context}")
        
        return {'evaluated': True, 'allow': False, 'reason': reason}
    
    def process_apply_results(self, symbol: str):
        """Process recent apply results to update ledger"""
        result_stream = "quantum:stream:apply.result"
        
        # Get latest result for this symbol
        try:
            results = self.redis.xrevrange(result_stream, count=10)
            
            for msg_id, fields in results:
                if fields.get('symbol') == symbol and fields.get('executed') == 'True':
                    # Update ledger from this result
                    result_data = {
                        'plan_id': fields.get('plan_id', ''),
                        'symbol': symbol,
                        'executed': True,
                        'steps_results': json.loads(fields.get('steps_results', '[]'))
                    }
                    self.update_ledger(symbol, result_data)
                    break  # Only process most recent executed result
        except Exception as e:
            logger.error(f"{symbol}: Failed to process apply results: {e}")
    
    def process_apply_plans_stream(self):
        """Event-driven: consume apply.plan stream via consumer group"""
        plan_stream = "quantum:stream:apply.plan"
        consumer_group = "p33"
        consumer_id = f"p33-{os.getpid()}"
        
        # Create consumer group (idempotent)
        try:
            self.redis.xgroup_create(plan_stream, consumer_group, id='$', mkstream=True)
            logger.info(f"Consumer group '{consumer_group}' created on {plan_stream}")
        except Exception as e:
            if "BUSYGROUP" not in str(e):
                logger.warning(f"Consumer group create error: {e}")
        
        try:
            # XREADGROUP: block 1000ms, read up to 10 messages
            messages = self.redis.xreadgroup(
                groupname=consumer_group,
                consumername=consumer_id,
                streams={plan_stream: '>'},
                count=10,
                block=1000  # 1 second block
            )
            
            if not messages:
                return  # No new messages
            
            for stream_name, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    # Parse plan_id (REQUIRED - no fallback)
                    plan_id = fields.get('plan_id')
                    if not plan_id:
                        logger.error(f"Message {msg_id}: Missing plan_id field - ACK and skip")
                        self.redis.xack(plan_stream, consumer_group, msg_id)
                        continue
                    
                    symbol = fields.get('symbol', 'UNKNOWN')
                    decision = fields.get('decision', '')
                    
                    # Only evaluate EXECUTE decisions
                    if decision != 'EXECUTE':
                        self.redis.xack(plan_stream, consumer_group, msg_id)
                        continue
                    
                    # Filter by allowlist
                    if symbol not in self.symbols:
                        logger.debug(f"{symbol}: Not in allowlist - ACK plan {plan_id[:8]}")
                        self.redis.xack(plan_stream, consumer_group, msg_id)
                        continue
                    
                    # Evaluate plan (issue or deny permit)
                    try:
                        logger.info(f"{symbol}: Evaluating plan {plan_id[:8]} from stream msg {msg_id}")
                        self.evaluate_plan(plan_id, symbol, fields)
                    except Exception as eval_err:
                        logger.error(f"{symbol}: Evaluation error for plan {plan_id[:8]}: {eval_err}")
                    
                    # ACK message after processing
                    self.redis.xack(plan_stream, consumer_group, msg_id)
                    
        except Exception as e:
            logger.error(f"Stream read error: {e}")
    
    def run(self):
        """Main loop - event-driven with periodic snapshot refresh"""
        logger.info("P3.3 Position State Brain starting (event-driven mode)")
        logger.info(f"Snapshot refresh: {self.config.POLL_INTERVAL}s")
        logger.info(f"Stale threshold: {self.config.STALE_THRESHOLD_SEC}s")
        logger.info(f"Cooldown: {self.config.COOLDOWN_SEC}s")
        logger.info(f"Permit TTL: {self.config.PERMIT_TTL}s")
        logger.info("Consumer group: p33 on quantum:stream:apply.plan")
        
        last_snapshot_update = 0
        
        while True:
            try:
                # Update exchange snapshots periodically (every POLL_INTERVAL seconds)
                now = time.time()
                if now - last_snapshot_update >= self.config.POLL_INTERVAL:
                    for symbol in self.symbols:
                        self.update_exchange_snapshot(symbol)
                        self.process_apply_results(symbol)
                    last_snapshot_update = now
                
                # EVENT-DRIVEN: Read from apply.plan stream (blocks 1s)
                self.process_apply_plans_stream()
                
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)


def main():
    config = Config()
    brain = PositionStateBrain(config)
    brain.run()


if __name__ == "__main__":
    main()
