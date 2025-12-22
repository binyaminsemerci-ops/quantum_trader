#!/usr/bin/env python3
"""
Phase 6: Auto Execution Layer
Safe, regulated trading execution connecting AI Engine → Exchange
"""
import os
import time
import json
import redis
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Redis connection
r = redis.Redis(
    host=os.getenv("REDIS_HOST", "redis"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True
)

# Exchange configuration
EXCHANGE = os.getenv("EXCHANGE", "binance")
TESTNET = os.getenv("TESTNET", "true").lower() == "true"
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

# Binance API (conditional import)
try:
    from binance.client import Client
    from binance.enums import *
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if TESTNET:
        client = Client(api_key, api_secret, testnet=True)
        logger.info("🧪 Using Binance TESTNET")
    else:
        client = Client(api_key, api_secret)
        logger.info("📈 Using Binance MAINNET")
        
    BINANCE_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ python-binance not installed, using paper trading mode")
    client = None
    BINANCE_AVAILABLE = False
    PAPER_TRADING = True

# Risk management settings
RISK_LIMIT = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))  # 1% risk per trade
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "3"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "1000"))  # USDT
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", "4.0"))  # Circuit breaker at 4%

# Circuit breaker state
circuit_breaker_active = False
circuit_breaker_until = 0


class AutoExecutor:
    """
    Autonomous trading execution layer
    Features:
    - Signal-to-order execution
    - Leverage and position sizing from Risk Brain
    - Order tracking and fill logging
    - Circuit breaker on errors
    - Full logging to governance dashboard
    """
    
    def __init__(self):
        self.paper_balance = 10000.0  # Starting paper trading balance
        self.positions = {}  # Track open positions
        self.trade_count = 0
        self.successful_trades = 0
        self.failed_trades = 0
        
        logger.info("=" * 60)
        logger.info("Phase 6: Auto Execution Layer Initialized")
        logger.info("=" * 60)
        logger.info(f"Exchange: {EXCHANGE.upper()}")
        logger.info(f"Mode: {'🧪 TESTNET' if TESTNET else '📈 MAINNET'}")
        logger.info(f"Paper Trading: {'✅ Yes' if PAPER_TRADING else '❌ No'}")
        logger.info(f"Risk Per Trade: {RISK_LIMIT*100}%")
        logger.info(f"Max Leverage: {MAX_LEVERAGE}x")
        logger.info(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        logger.info(f"Circuit Breaker: Drawdown > {MAX_DRAWDOWN}%")
        logger.info("=" * 60)

    def get_balance(self, asset: str = "USDT") -> float:
        """Get account balance"""
        if PAPER_TRADING:
            return self.paper_balance
        
        try:
            if BINANCE_AVAILABLE and client:
                if TESTNET:
                    info = client.futures_account_balance()
                    for item in info:
                        if item['asset'] == asset:
                            return float(item['balance'])
                else:
                    info = client.get_asset_balance(asset=asset)
                    return float(info['free'])
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error getting balance: {e}")
            return 0.0

    def calculate_position_size(self, symbol: str, balance: float, confidence: float) -> float:
        """Calculate position size based on risk management"""
        # Base risk amount
        risk_amount = balance * RISK_LIMIT
        
        # Adjust by confidence
        confidence_multiplier = min(confidence / CONFIDENCE_THRESHOLD, 1.5)
        adjusted_risk = risk_amount * confidence_multiplier
        
        # Apply leverage
        position_size = adjusted_risk * MAX_LEVERAGE
        
        # Cap at maximum position size
        position_size = min(position_size, MAX_POSITION_SIZE)
        
        # Round to 3 decimals for most exchanges
        return round(position_size, 3)

    def place_order(
        self, 
        symbol: str, 
        side: str, 
        qty: float, 
        price: Optional[float] = None, 
        leverage: int = 1
    ) -> Optional[Dict]:
        """Place order on exchange"""
        global circuit_breaker_active, circuit_breaker_until
        
        # Check circuit breaker
        if circuit_breaker_active:
            if time.time() < circuit_breaker_until:
                logger.warning(f"🚨 Circuit breaker active - skipping order")
                return None
            else:
                circuit_breaker_active = False
                logger.info("✅ Circuit breaker reset")
        
        if PAPER_TRADING:
            # Simulate paper trading order
            order = {
                "orderId": f"PAPER_{int(time.time())}",
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
                "price": price or 50000.0,  # Dummy price
                "status": "FILLED",
                "timestamp": int(time.time() * 1000),
                "paper": True
            }
            
            # Update paper balance
            if side == "SELL":
                self.paper_balance -= qty
            
            logger.info(f"📝 Paper order: {symbol} {side} {qty} @ leverage {leverage}x")
            return order
        
        try:
            if not BINANCE_AVAILABLE or not client:
                logger.error("❌ Binance client not available")
                return None
            
            # Set leverage
            client.futures_change_leverage(symbol=symbol, leverage=leverage)
            
            # Place market order
            if side.upper() == "BUY":
                order = client.futures_create_order(
                    symbol=symbol,
                    side="BUY",
                    type="MARKET",
                    quantity=qty
                )
            elif side.upper() == "SELL":
                order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL",
                    type="MARKET",
                    quantity=qty
                )
            else:
                logger.error(f"❌ Invalid side: {side}")
                return None
            
            # Get fill price and position details
            fill_price = float(order.get('avgPrice', price or 0))
            contract_qty = float(order.get('executedQty', qty))
            notional = fill_price * contract_qty
            
            logger.info(f"✅ Order placed: {symbol} {side} {contract_qty} contracts ({notional:.2f} USDT) @ {leverage}x")
            self.successful_trades += 1
            
            # Set TP/SL automatically after order placement
            try:
                # Get current market price for precision
                ticker = client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                
                # Dynamic TP/SL based on leverage (LSF-inspired)
                # Higher leverage = tighter stops
                if leverage >= 10:
                    tp_pct = 0.012  # 1.2% TP for ultra-high leverage
                    sl_pct = 0.006  # 0.6% SL
                elif leverage >= 5:
                    tp_pct = 0.015  # 1.5% TP for high leverage
                    sl_pct = 0.008  # 0.8% SL
                else:
                    tp_pct = 0.02   # 2% TP for low leverage
                    sl_pct = 0.01   # 1% SL
                
                # Calculate TP/SL prices based on position direction
                if side.upper() == "BUY":
                    take_profit_price = fill_price * (1 + tp_pct)
                    stop_loss_price = fill_price * (1 - sl_pct)
                else:  # SELL (SHORT)
                    take_profit_price = fill_price * (1 - tp_pct)
                    stop_loss_price = fill_price * (1 + sl_pct)
                
                # Determine price precision dynamically
                price_str = str(current_price)
                if '.' in price_str:
                    decimals = len(price_str.split('.')[1])
                    # Use appropriate precision (max 8 decimals for crypto)
                    price_precision = min(decimals, 8)
                else:
                    price_precision = 2
                
                # Round prices
                take_profit_price = round(take_profit_price, price_precision)
                stop_loss_price = round(stop_loss_price, price_precision)
                
                # Safety check: Ensure stop loss price is positive and reasonable
                if stop_loss_price <= 0:
                    raise ValueError(f"Invalid SL price: {stop_loss_price} (must be > 0)")
                
                if take_profit_price <= 0:
                    raise ValueError(f"Invalid TP price: {take_profit_price} (must be > 0)")
                
                # Validate price spread
                if side.upper() == "BUY":
                    if stop_loss_price >= fill_price:
                        raise ValueError(f"SL {stop_loss_price} must be < entry {fill_price} for LONG")
                    if take_profit_price <= fill_price:
                        raise ValueError(f"TP {take_profit_price} must be > entry {fill_price} for LONG")
                else:  # SELL
                    if stop_loss_price <= fill_price:
                        raise ValueError(f"SL {stop_loss_price} must be > entry {fill_price} for SHORT")
                    if take_profit_price >= fill_price:
                        raise ValueError(f"TP {take_profit_price} must be < entry {fill_price} for SHORT")
                
                # Place Take Profit order
                tp_order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side.upper() == "BUY" else "BUY",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=take_profit_price,
                    closePosition=True
                )
                logger.info(f"✅ TP set @ ${take_profit_price} ({tp_pct*100:+.1f}%)")
                
                # Place Stop Loss order
                sl_order = client.futures_create_order(
                    symbol=symbol,
                    side="SELL" if side.upper() == "BUY" else "BUY",
                    type="STOP_MARKET",
                    stopPrice=stop_loss_price,
                    closePosition=True
                )
                logger.info(f"✅ SL set @ ${stop_loss_price} ({-sl_pct*100:.1f}%)")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to set TP/SL: {e}")
                # Don't fail the main order if TP/SL fails
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Order error: {e}")
            self.failed_trades += 1
            
            # Trigger circuit breaker after 3 consecutive failures
            if self.failed_trades >= 3:
                circuit_breaker_active = True
                circuit_breaker_until = time.time() + 300  # 5 minutes
                logger.warning(f"🚨 CIRCUIT BREAKER ACTIVATED - 5 minute cooldown")
            
            return None

    def log_trade(
        self, 
        symbol: str, 
        action: str, 
        qty: float, 
        price: float,
        confidence: float,
        pnl: float = 0.0
    ):
        """Log trade to Redis for governance and analytics"""
        record = {
            "symbol": symbol,
            "action": action,
            "qty": qty,
            "price": price,
            "confidence": confidence,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat(),
            "leverage": MAX_LEVERAGE,
            "paper": PAPER_TRADING,
            "testnet": TESTNET
        }
        
        try:
            # Store in Redis list
            r.lpush("trade_log", json.dumps(record))
            r.ltrim("trade_log", 0, 999)  # Keep last 1000 trades
            
            # Update trade count
            r.incr("total_trades")
            
            # Update metrics
            r.hincrby("execution_metrics", "total_orders", 1)
            if pnl > 0:
                r.hincrby("execution_metrics", "profitable_trades", 1)
            
            logger.info(f"📊 Trade logged: {symbol} {action} {qty} conf={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")

    def check_drawdown(self, signal: Dict) -> bool:
        """Check if drawdown exceeds circuit breaker threshold"""
        drawdown = signal.get("drawdown", 0.0)
        
        if drawdown > MAX_DRAWDOWN:
            logger.warning(
                f"🚨 CIRCUIT BREAKER: {signal['symbol']} drawdown={drawdown:.2f}% "
                f"(threshold: {MAX_DRAWDOWN}%)"
            )
            return True
        
        return False

    def process_signal(self, signal: Dict) -> bool:
        """Process a single trading signal"""
        try:
            symbol = signal.get("symbol", "").upper()
            action = signal.get("action", "").upper()
            confidence = signal.get("confidence", 0.0)
            pnl = signal.get("pnl", 0.0)
            price = signal.get("price", 0.0)
            
            # Validation
            if not symbol or not action:
                logger.debug("⚠️ Invalid signal: missing symbol or action")
                return False
            
            if action not in ["BUY", "SELL", "CLOSE"]:
                logger.debug(f"⚠️ Invalid action: {action}")
                return False
            
            # Check confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                logger.debug(
                    f"⚠️ Signal rejected: {symbol} confidence={confidence:.2f} "
                    f"< {CONFIDENCE_THRESHOLD}"
                )
                return False
            
            # Check drawdown circuit breaker
            if self.check_drawdown(signal):
                return False
            
            # Get balance and calculate position size
            balance = self.get_balance()
            qty = self.calculate_position_size(symbol, balance, confidence)
            
            if qty < 0.001:  # Minimum order size
                logger.warning(f"⚠️ Position size too small: {qty}")
                return False
            
            # Place order
            order = self.place_order(symbol, action, qty, price, MAX_LEVERAGE)
            
            if order:
                # Log successful trade
                self.log_trade(symbol, action, qty, price, confidence, pnl)
                self.trade_count += 1
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error processing signal: {e}")
            return False

    def get_live_signals(self) -> List[Dict]:
        """Fetch live trading signals from EventBus Redis Stream"""
        try:
            # Read from EventBus stream quantum:stream:trade.intent
            stream_name = "quantum:stream:trade.intent"
            
            # Read last 50 messages from stream
            messages = r.xrevrange(stream_name, count=50)
            
            if not messages:
                return []
            
            signals = []
            for message_id, data in messages:
                try:
                    # Parse payload field
                    payload = data.get(b'payload', data.get('payload', b'{}'))
                    if isinstance(payload, bytes):
                        payload = payload.decode('utf-8')
                    
                    signal = json.loads(payload)
                    
                    # Filter out HOLD signals
                    if signal.get('side') != 'HOLD':
                        signals.append(signal)
                        
                except Exception as e:
                    logger.debug(f"Failed to parse signal: {e}")
                    continue
            
            return signals[:20]  # Return max 20 most recent signals
            
        except Exception as e:
            logger.error(f"❌ Error fetching signals from EventBus: {e}")
            return []

    def update_metrics(self):
        """Update execution metrics in Redis"""
        try:
            metrics = {
                "total_trades": self.trade_count,
                "successful_trades": self.successful_trades,
                "failed_trades": self.failed_trades,
                "success_rate": (
                    self.successful_trades / self.trade_count * 100 
                    if self.trade_count > 0 else 0
                ),
                "balance": self.get_balance(),
                "circuit_breaker": circuit_breaker_active,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            r.set("executor_metrics", json.dumps(metrics))
            
        except Exception as e:
            logger.error(f"❌ Failed to update metrics: {e}")

    def run_execution_loop(self):
        """Main execution loop - runs 24/7"""
        logger.info("🚀 Starting auto execution loop...")
        logger.info("Processing signals every 10 seconds")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                
                # Get live signals
                signals = self.get_live_signals()
                
                if signals:
                    logger.info(
                        f"[Cycle {cycle}] Processing {len(signals)} signal(s)..."
                    )
                    
                    processed = 0
                    for signal in signals:
                        if self.process_signal(signal):
                            processed += 1
                    
                    logger.info(
                        f"[Cycle {cycle}] Processed {processed}/{len(signals)} signals"
                    )
                else:
                    logger.debug(f"[Cycle {cycle}] No signals to process")
                
                # Update metrics
                self.update_metrics()
                
                # Log status every 10 cycles (100 seconds)
                if cycle % 10 == 0:
                    balance = self.get_balance()
                    logger.info(
                        f"[Status] Balance: ${balance:.2f} | "
                        f"Trades: {self.trade_count} | "
                        f"Success Rate: {self.successful_trades}/{self.trade_count}"
                    )
                
                # Sleep before next cycle
                time.sleep(10)
                
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down executor...")
                break
            except Exception as e:
                logger.error(f"❌ Execution loop error: {e}")
                time.sleep(15)


def main():
    """Entry point"""
    try:
        executor = AutoExecutor()
        executor.run_execution_loop()
    except Exception as e:
        logger.critical(f"🚨 FATAL: Executor failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
