#!/usr/bin/env python3
"""
Phase 6: Auto Execution Layer
Safe, regulated trading execution connecting AI Engine → Exchange
"""
import os
import sys
import time
import json
import redis
import logging
from datetime import datetime
from typing import Dict, List, Optional

# P1-B: JSON logging with correlation_id
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from shared.logging_config import (
    setup_json_logging, set_correlation_id, get_correlation_id,
    log_intent_received, log_order_submit, log_order_response, log_order_error,
    log_corr_assigned
)

# Setup JSON logging
logger = setup_json_logging('auto_executor', level=os.getenv('LOG_LEVEL', 'INFO'))

# Add microservices to path for intelligent leverage engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

try:
    from microservices.exitbrain_v3_5.intelligent_leverage_engine import (
        IntelligentLeverageEngine,
        LeverageCalculation
    )
    from microservices.exitbrain_v3_5.exit_brain import (
        ExitBrainV35,
        SignalContext,
        ExitPlan
    )
    LEVERAGE_ENGINE_AVAILABLE = True
    logger.info("✅ Intelligent Leverage Engine imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Intelligent Leverage Engine not available: {e}")
    LEVERAGE_ENGINE_AVAILABLE = False

# P1-B: Import Execution Policy
try:
    from execution_policy import (
        ExecutionPolicy,
        PolicyConfig,
        PolicyDecision,
        PortfolioState
    )
    EXECUTION_POLICY_AVAILABLE = True
    logger.info("✅ Execution Policy imported successfully")
except ImportError as e:
    logger.warning(f"⚠️ Execution Policy not available: {e}")
    EXECUTION_POLICY_AVAILABLE = False

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
    from binance.helpers import round_step_size
    
    # Initialize Binance client
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if api_key and api_secret:
        if TESTNET:
            # For testnet: use testnet=True + manual URL override
            client = Client(api_key, api_secret, testnet=True)
            # Override to futures testnet endpoints
            client.API_URL = "https://testnet.binancefuture.com"
            client.FUTURES_URL = "https://testnet.binancefuture.com/fapi"
            client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/fapi"
            # Increase recvWindow for timing tolerance (default is 5000ms)
            client.timestamp_offset = 1000  # Add 1 second buffer
            logger.info("🧪 Using Binance Futures TESTNET")
            logger.info(f"📡 API_URL: {client.API_URL}")
            logger.info(f"📡 FUTURES_URL: {client.FUTURES_URL}")
        else:
            client = Client(api_key, api_secret)
            logger.info("📈 Using Binance MAINNET")
        BINANCE_AVAILABLE = True
        logger.info(f"✅ Binance client initialized | mode={'TESTNET' if TESTNET else 'MAINNET'} | paper={PAPER_TRADING}")
    else:
        # P0 FIX: Don't silently fallback if PAPER_TRADING=false
        if not PAPER_TRADING:
            logger.error("❌ FATAL_BINANCE_UNAVAILABLE: Credentials missing but PAPER_TRADING=false")
            logger.error("❌ Set BINANCE_API_KEY and BINANCE_API_SECRET or enable PAPER_TRADING=true")
            sys.exit(1)
        else:
            logger.warning("⚠️ Binance credentials not found - using paper trading mode")
            client = None
            BINANCE_AVAILABLE = False
except Exception as e:
    # P0 FIX: Don't silently fallback if PAPER_TRADING=false
    if not PAPER_TRADING:
        logger.error(f"❌ FATAL_BINANCE_UNAVAILABLE: Client initialization failed: {e}")
        logger.error("❌ Check credentials and network connectivity or enable PAPER_TRADING=true")
        sys.exit(1)
    else:
        logger.warning(f"⚠️ Binance client initialization failed: {e}")
        client = None
        BINANCE_AVAILABLE = False

# Wrapper functions for Binance API calls with recvWindow
def safe_futures_call(func_name, *args, **kwargs):
    """Wrapper to add recvWindow to all Binance futures API calls"""
    if not client:
        raise Exception("Binance client not initialized")
    
    # List of methods that don't require recvWindow (unsigned endpoints)
    unsigned_methods = ['futures_time', 'futures_exchange_info', 'futures_ticker', 
                       'futures_orderbook', 'futures_klines', 'futures_trades']
    
    # Add recvWindow for signed requests only (default 5000ms, increase to 10000ms)
    if func_name not in unsigned_methods and 'recvWindow' not in kwargs:
        kwargs['recvWindow'] = 10000
    
    func = getattr(client, func_name)
    return func(*args, **kwargs)

# Invalid symbols to filter out (not available on Binance testnet)
INVALID_SYMBOLS = {"KASUSDT", "FTMUSDT", "KAUSUSDT"}  # Add more as needed

# Risk management settings
RISK_LIMIT = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))  # 1% risk per trade
# ✅ AI-DRIVEN: ILF-v2 dynamically calculates leverage 5-80x based on confidence + volatility
MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "80"))  # Upper bound for ILF-v2
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "1000"))  # USDT
# ✅ AI-DRIVEN: Lowered from 0.55 to 0.45 to accept more AI signals
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.45"))
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
        self.symbol_info_cache = {}  # Cache for symbol exchange info
        self.leverage_brackets_cache = {}  # Cache for leverage brackets
        
        # P1-B: Initialize Execution Policy
        if EXECUTION_POLICY_AVAILABLE:
            self.execution_policy = ExecutionPolicy(PolicyConfig.from_env())
            logger.info("🛡️ Execution Policy initialized with capital controls")
        else:
            self.execution_policy = None
            logger.warning("⚠️ Execution Policy not available - using legacy logic")
        
        # Initialize Intelligent Leverage Engine
        if LEVERAGE_ENGINE_AVAILABLE:
            self.leverage_engine = IntelligentLeverageEngine(config={
                "min_leverage": float(os.getenv("MIN_LEVERAGE", "5.0")),
                "max_leverage": float(os.getenv("MAX_LEVERAGE", "80.0")),
                "safety_cap": 0.9
            })
            logger.info("🧠 Intelligent Leverage Engine initialized (5-80x dynamic)")
        else:
            self.leverage_engine = None
            logger.warning("⚠️ Using fallback MAX_LEVERAGE from environment")
        
        # Initialize ExitBrain v3.5 for sophisticated TP/SL formulas
        try:
            self.exit_brain = ExitBrainV35(
                redis_client=r,
                config={
                    "base_tp_pct": 0.01,  # 1% base TP
                    "base_sl_pct": 0.005,  # 0.5% base SL
                    "dynamic_reward": True  # Enable PnL feedback
                }
            )
            logger.info("🧠 ExitBrain v3.5 initialized with LSF formulas")
        except Exception as e:
            self.exit_brain = None
            logger.error(f"❌ Failed to initialize ExitBrain: {e}")
        
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
                    info = safe_futures_call('futures_account_balance')
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

    def get_max_leverage(self, symbol: str) -> int:
        """Get maximum allowed leverage for a symbol from Binance leverage brackets"""
        # Check cache first
        if symbol in self.leverage_brackets_cache:
            return self.leverage_brackets_cache[symbol]
        
        try:
            if not BINANCE_AVAILABLE or not client:
                return 20  # Safe default
            
            # Get leverage brackets
            brackets = safe_futures_call('futures_leverage_bracket', symbol=symbol)
            
            if brackets and len(brackets) > 0:
                # First bracket contains max leverage for minimum notional
                symbol_data = brackets[0] if isinstance(brackets, list) else brackets
                if 'brackets' in symbol_data:
                    max_lev = symbol_data['brackets'][0]['initialLeverage']
                    self.leverage_brackets_cache[symbol] = max_lev
                    return max_lev
            
            # Fallback
            return 20
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get leverage bracket for {symbol}: {e}")
            return 20  # Safe default

    def get_symbol_info(self, symbol: str) -> dict:
        """Get symbol trading rules from Binance (with caching)"""
        # Check cache first
        if symbol in self.symbol_info_cache:
            return self.symbol_info_cache[symbol]
        
        try:
            if not BINANCE_AVAILABLE or not client:
                # Default fallback values
                return {
                    'stepSize': '0.001',
                    'minQty': '0.001',
                    'maxQty': '100000',
                    'tickSize': '0.01',
                    'precision': 3
                }
            
            # Get exchange info
            exchange_info = client.futures_exchange_info()
            
            for symbol_data in exchange_info['symbols']:
                if symbol_data['symbol'] == symbol:
                    # Extract LOT_SIZE filter (quantity rules)
                    lot_size = next((f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                    price_filter = next((f for f in symbol_data['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
                    
                    # Get max leverage for this symbol (from symbol data, not filters)
                    max_leverage = symbol_data.get('maxLeverage', 125)  # Default to 125x if not specified
                    
                    if lot_size and price_filter:
                        step_size = lot_size['stepSize']
                        # Calculate precision from stepSize (e.g., "0.001" = 3 decimals)
                        precision = len(step_size.rstrip('0').split('.')[-1]) if '.' in step_size else 0
                        
                        info = {
                            'stepSize': step_size,
                            'minQty': lot_size['minQty'],
                            'maxQty': lot_size['maxQty'],
                            'maxLeverage': max_leverage,
                            'tickSize': price_filter['tickSize'],
                            'precision': precision
                        }
                        
                        # Cache it
                        self.symbol_info_cache[symbol] = info
                        logger.info(f"📏 [{symbol}] Precision: qty={precision} (stepSize: {step_size}), price (tickSize: {price_filter['tickSize']})")
                        return info
            
            # Fallback if symbol not found
            logger.warning(f"⚠️ Symbol {symbol} not found in exchange info, using defaults")
            return {
                'stepSize': '0.001',
                'minQty': '0.001',
                'maxQty': '100000',
                'tickSize': '0.01',
                'precision': 3
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting symbol info for {symbol}: {e}")
            return {
                'stepSize': '0.001',
                'minQty': '0.001',
                'maxQty': '100000',
                'tickSize': '0.01',
                'precision': 3
            }

    def calculate_position_size(self, symbol: str, balance: float, confidence: float) -> float:
        """Calculate position size based on risk management"""
        # Base risk amount
        risk_amount = balance * RISK_LIMIT
        
        # Adjust by confidence
        confidence_multiplier = min(confidence / CONFIDENCE_THRESHOLD, 1.5)
        adjusted_risk = risk_amount * confidence_multiplier
        
        # Apply leverage
        position_value_usdt = adjusted_risk * MAX_LEVERAGE
        
        # Cap at maximum position size
        position_value_usdt = min(position_value_usdt, MAX_POSITION_SIZE)
        
        # Get current mark price to convert USDT to contracts
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            mark_price = float(ticker['price'])
        except Exception as e:
            logger.error(f"❌ [{symbol}] Failed to get mark price: {e}")
            return 0.0
        
        # Convert USDT value to number of contracts
        position_size_contracts = position_value_usdt / mark_price
        
        # Get symbol-specific precision
        symbol_info = self.get_symbol_info(symbol)
        precision = symbol_info['precision']
        min_qty = float(symbol_info['minQty'])
        
        # Round to correct precision
        position_size_contracts = round(position_size_contracts, precision)
        
        # Convert to int if precision is 0 (whole numbers only)
        if precision == 0:
            position_size_contracts = int(position_size_contracts)
        
        logger.info(f"💰 [{symbol}] Position: {position_size_contracts} contracts (${position_value_usdt:.2f} @ ${mark_price:.4f}) precision={precision} min={min_qty}")

        # Ensure meets minimum
        if position_size_contracts < min_qty:
            position_size_contracts = min_qty if precision > 0 else int(min_qty)
            logger.info(f"⬆️ [{symbol}] Adjusted to minimum: {min_qty} contracts")

        return position_size_contracts

    def calculate_dynamic_leverage(self, symbol: str, confidence: float, volatility: float = 0.02) -> int:
        """
        Calculate optimal leverage using Intelligent Leverage Engine
        
        Args:
            symbol: Trading symbol
            confidence: AI signal confidence [0-1]
            volatility: Market volatility (default 0.02 = 2%)
            
        Returns:
            Optimal leverage between MIN_LEVERAGE and MAX_LEVERAGE
        """
        if not self.leverage_engine:
            # Fallback to environment variable
            return MAX_LEVERAGE
        
        try:
            # Calculate recent PnL trend (simplified)
            pnl_trend = 0.0  # TODO: Calculate from recent trades
            if self.successful_trades > 0:
                win_rate = self.successful_trades / max(self.trade_count, 1)
                pnl_trend = (win_rate - 0.5) * 2  # Convert 0-1 to -1 to +1
            
            # Use intelligent leverage engine
            result = self.leverage_engine.calculate_leverage(
                confidence=confidence,
                volatility=volatility,
                pnl_trend=pnl_trend,
                symbol_risk=1.0,  # TODO: Add symbol-specific risk weights
                margin_util=0.0,  # TODO: Calculate actual margin utilization
                exch_divergence=0.0,  # TODO: Add cross-exchange data
                funding_rate=0.0  # TODO: Get funding rate from Binance
            )
            
            # Get symbol's actual max leverage from Binance leverage brackets
            symbol_max_leverage = self.get_max_leverage(symbol)
            
            # Cap to symbol's actual max leverage
            leverage = int(result.leverage)
            leverage = min(leverage, symbol_max_leverage, MAX_LEVERAGE)
            
            logger.info(
                f"🧠 [{symbol}] Dynamic Leverage: {leverage}x "
                f"(ILF: {int(result.leverage)}x, max: {symbol_max_leverage}x, "
                f"confidence={confidence:.2f}, vol={volatility:.3f})"
            )
            
            return leverage
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate dynamic leverage: {e}")
            return MAX_LEVERAGE

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
            
            # Set margin type to ISOLATED (safer than CROSS)
            try:
                safe_futures_call('futures_change_margin_type', symbol=symbol, marginType='ISOLATED')
                logger.info(f"🛡️ [{symbol}] Margin type set to ISOLATED")
            except Exception as e:
                # Ignore error if already in ISOLATED mode
                if 'No need to change margin type' not in str(e):
                    logger.warning(f"⚠️ [{symbol}] Could not change margin type: {e}")
            
            # Set leverage
            safe_futures_call('futures_change_leverage', symbol=symbol, leverage=leverage)
            
            # P0 PROOF LOG: About to submit order
            endpoint_host = client.FUTURES_URL if client else "unknown"
            logger.info(f"🚀 ORDER_SUBMIT | symbol={symbol} | side={side} | qty={qty} | "
                      f"type=MARKET | endpoint={endpoint_host} | mode={'TESTNET' if TESTNET else 'MAINNET'}")
            
            # Place market order
            if side.upper() == "BUY":
                order = safe_futures_call('futures_create_order',
                    symbol=symbol,
                    side="BUY",
                    positionSide="LONG",
                    type="MARKET",
                    quantity=qty
                )
            elif side.upper() == "SELL":
                order = safe_futures_call('futures_create_order',
                    symbol=symbol,
                    side="SELL",
                    positionSide="SHORT",
                    type="MARKET",
                    quantity=qty
                )
            else:
                logger.error(f"❌ Invalid side: {side}")
                return None
            
            # P0 PROOF LOG: Order response received
            logger.info(f"✅ ORDER_RESPONSE | orderId={order.get('orderId')} | "
                      f"status={order.get('status')} | symbol={symbol} | "
                      f"updateTime={order.get('updateTime')} | response={json.dumps(order)[:200]}")
            
            # DEBUG: Log full Binance response
            logger.info(f"🔍 [{symbol}] Binance order response: {order}")
            
            # Get fill price and position details
            fill_price = float(order.get('avgPrice', 0))
            contract_qty = float(order.get('executedQty', qty))
            
            # If avgPrice not in response, get current market price
            if fill_price == 0:
                ticker = client.get_symbol_ticker(symbol=symbol)
                fill_price = float(ticker['price'])
            
            notional = fill_price * contract_qty
            
            logger.info(f"✅ Order placed: {symbol} {side} {contract_qty} contracts ({notional:.2f} USDT) @ ${fill_price:.4f} with {leverage}x leverage")
            self.successful_trades += 1
            
            # Set TP/SL automatically after order placement using ExitBrain formulas
            try:
                # Use fill_price for calculations (already fetched above)
                current_price = fill_price
                
                # Use ExitBrain v3.5 formulas for intelligent TP/SL calculation
                # This respects LSF (Leverage Sensitivity Factor) and adaptive formulas
                tp_pct = None
                sl_pct = None
                
                if self.exit_brain:
                    try:
                        # Create signal context for ExitBrain
                        signal_context = SignalContext(
                            symbol=symbol,
                            side="long" if side.upper() == "BUY" else "short",
                            confidence=0.8,  # Default, should be passed from signal
                            entry_price=fill_price,
                            atr_value=0.02,  # Default 2%, should be from market data
                            timestamp=time.time()
                        )
                        
                        # Build exit plan using ExitBrain v3.5
                        exit_plan = self.exit_brain.build_exit_plan(
                            signal=signal_context,
                            pnl_trend=0.0,  # TODO: Get from PnL tracker
                            symbol_risk=1.0,  # Default
                            margin_util=0.0,  # TODO: Calculate from account
                            exch_divergence=0.0,  # TODO: Get from cross-exchange
                            funding_rate=0.0  # TODO: Get from Binance
                        )
                        
                        tp_pct = exit_plan.take_profit_pct
                        sl_pct = exit_plan.stop_loss_pct
                        
                        logger.info(
                            f"🧠 [{symbol}] ExitBrain: TP={tp_pct*100:.2f}% SL={sl_pct*100:.2f}% "
                            f"Leverage={exit_plan.leverage:.1f}x | {exit_plan.reasoning}"
                        )
                        
                    except Exception as eb_error:
                        logger.warning(f"⚠️ ExitBrain calculation failed: {eb_error}, using fallback")
                
                # Fallback to simple leverage-based TP/SL if ExitBrain unavailable
                if tp_pct is None or sl_pct is None:
                    if leverage >= 10:
                        tp_pct = 0.012  # 1.2% TP
                        sl_pct = 0.006  # 0.6% SL
                    elif leverage >= 5:
                        tp_pct = 0.015  # 1.5% TP
                        sl_pct = 0.008  # 0.8% SL
                    else:
                        tp_pct = 0.02   # 2% TP
                        sl_pct = 0.01   # 1% SL
                    logger.info(f"📊 [{symbol}] Fallback TP/SL: {tp_pct*100:.1f}%/{sl_pct*100:.1f}%")
                
                # Calculate TP/SL prices based on position direction
                if side.upper() == "BUY":
                    take_profit_price = fill_price * (1 + tp_pct)
                    stop_loss_price = fill_price * (1 - sl_pct)
                else:  # SELL (SHORT)
                    take_profit_price = fill_price * (1 - tp_pct)
                    stop_loss_price = fill_price * (1 + sl_pct)
                
                # Get symbol info for price precision
                symbol_info = self.get_symbol_info(symbol)
                tick_size = float(symbol_info.get('tickSize', '0.01'))
                
                # Calculate price precision from tick size
                if '.' in str(tick_size):
                    price_precision = len(str(tick_size).rstrip('0').split('.')[1])
                else:
                    price_precision = 0
                
                # Round prices to tick size
                take_profit_price = round(take_profit_price, price_precision)
                stop_loss_price = round(stop_loss_price, price_precision)
                
                # Safety check: Ensure prices are positive and reasonable
                if stop_loss_price <= 0 or fill_price <= 0:
                    raise ValueError(f"Invalid SL price: {stop_loss_price} or fill price: {fill_price} (must be > 0)")
                
                if take_profit_price <= 0:
                    raise ValueError(f"Invalid TP price: {take_profit_price} (must be > 0)")
                
                # Validate price spread (with tolerance for rounding)
                price_tolerance = tick_size * 2
                if side.upper() == "BUY":
                    if stop_loss_price >= (fill_price - price_tolerance):
                        raise ValueError(f"SL {stop_loss_price} must be < entry {fill_price} for LONG")
                    if take_profit_price <= (fill_price + price_tolerance):
                        raise ValueError(f"TP {take_profit_price} must be > entry {fill_price} for LONG")
                else:  # SELL (SHORT)
                    if stop_loss_price <= (fill_price + price_tolerance):
                        raise ValueError(f"SL {stop_loss_price} must be > entry {fill_price} for SHORT")
                    if take_profit_price >= (fill_price - price_tolerance):
                        raise ValueError(f"TP {take_profit_price} must be < entry {fill_price} for SHORT")
                
# Place Take Profit order with positionSide for hedge mode
                tp_order = safe_futures_call('futures_create_order',
                    symbol=symbol,
                    side="SELL" if side.upper() == "BUY" else "BUY",
                    positionSide="LONG" if side.upper() == "BUY" else "SHORT",
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=take_profit_price,
                    closePosition=True
                )
                logger.info(f"✅ TP set @ ${take_profit_price} ({tp_pct*100:+.1f}%)")  
                
                # Place Stop Loss order with positionSide for hedge mode
                sl_order = safe_futures_call('futures_create_order',
                    symbol=symbol,
                    side="SELL" if side.upper() == "BUY" else "BUY",
                    positionSide="LONG" if side.upper() == "BUY" else "SHORT",
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
            # P0 PROOF LOG: Order error
            error_type = type(e).__name__
            binance_code = getattr(e, 'code', 'N/A')
            logger.error(f"🚨 ORDER_ERROR | symbol={symbol} | error_type={error_type} | "
                       f"binance_code={binance_code} | message={str(e)[:200]}")
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
            
            # ✅ NEW: Publish trade.closed event for continuous learning
            if pnl != 0.0:  # Only publish when PnL is calculated (position closed)
                self.publish_trade_closed(symbol, action, price, confidence, pnl)
            
            logger.info(f"📊 Trade logged: {symbol} {action} {qty} conf={confidence:.2f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log trade: {e}")

    def publish_trade_closed(
        self,
        symbol: str,
        action: str,
        price: float,
        confidence: float,
        pnl: float
    ):
        """Publish trade.closed event to Redis stream for continuous learning"""
        try:
            event_data = {
                "event_type": "trade.closed",
                "symbol": symbol,
                "side": action,
                "exit_price": price,
                "pnl_percent": pnl,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "model": "ensemble",
                "leverage": MAX_LEVERAGE
            }
            
            # Publish to trade.closed stream
            r.xadd(
                "quantum:stream:trade.closed",
                event_data,
                maxlen=1000  # Keep last 1000 closed trades
            )
            
            logger.info(f"📤 Published trade.closed: {symbol} PnL={pnl:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to publish trade.closed: {e}")

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
    
    def build_portfolio_state(self) -> 'PortfolioState':
        """
        Build current portfolio state for execution policy.
        
        Returns:
            PortfolioState with current positions, exposures, and capital.
        """
        try:
            if not BINANCE_AVAILABLE or not client or PAPER_TRADING:
                # Fallback to empty portfolio for paper trading
                balance = self.get_balance()
                return PortfolioState(
                    total_positions=0,
                    positions_by_symbol={},
                    positions_by_regime={},
                    total_exposure_usdt=0.0,
                    exposure_by_symbol={},
                    available_capital_usdt=balance,
                    last_trade_time=0.0,
                    last_trade_by_symbol={}
                )
            
            # Get all positions from Binance
            all_positions = safe_futures_call('futures_position_information')
            
            positions_by_symbol = {}
            exposure_by_symbol = {}
            positions_by_regime = {}
            total_exposure = 0.0
            total_count = 0
            
            for pos in all_positions:
                position_amt = float(pos.get('positionAmt', 0))
                if abs(position_amt) == 0:
                    continue  # Skip empty positions
                
                symbol = pos.get('symbol')
                entry_price = float(pos.get('entryPrice', 0))
                mark_price = float(pos.get('markPrice', entry_price))
                
                # Calculate exposure (notional value)
                exposure = abs(position_amt * mark_price)
                
                # Build position dict
                position_dict = {
                    "side": "long" if position_amt > 0 else "short",
                    "quantity": abs(position_amt),
                    "entry_price": entry_price,
                    "mark_price": mark_price,
                    "exposure_usdt": exposure,
                    "confidence": 0.8,  # Unknown, default
                    "leverage": int(pos.get('leverage', 1)),
                    "regime": "unknown"  # Unknown without signal context
                }
                
                # Add to positions_by_symbol
                if symbol not in positions_by_symbol:
                    positions_by_symbol[symbol] = []
                positions_by_symbol[symbol].append(position_dict)
                
                # Track exposure
                exposure_by_symbol[symbol] = exposure_by_symbol.get(symbol, 0.0) + exposure
                total_exposure += exposure
                total_count += 1
                
                # Track by regime (unknown for now)
                regime = position_dict['regime']
                positions_by_regime[regime] = positions_by_regime.get(regime, 0) + 1
            
            # Get available capital
            balance = self.get_balance()
            available_capital = max(0, balance - total_exposure * 0.1)  # Reserve 10% margin
            
            # Get last trade times (from self.positions cache or default)
            last_trade_by_symbol = getattr(self, 'last_trade_by_symbol', {})
            last_trade_time = getattr(self, 'last_trade_time', 0.0)
            
            portfolio = PortfolioState(
                total_positions=total_count,
                positions_by_symbol=positions_by_symbol,
                positions_by_regime=positions_by_regime,
                total_exposure_usdt=total_exposure,
                exposure_by_symbol=exposure_by_symbol,
                available_capital_usdt=available_capital,
                last_trade_time=last_trade_time,
                last_trade_by_symbol=last_trade_by_symbol
            )
            
            logger.debug(
                f"📊 Portfolio: {portfolio.total_positions} positions, "
                f"${portfolio.total_exposure_usdt:.0f} exposure, "
                f"${portfolio.available_capital_usdt:.0f} available"
            )
            
            return portfolio
            
        except Exception as e:
            logger.error(f"❌ Failed to build portfolio state: {e}")
            # Return empty portfolio on error
            balance = self.get_balance()
            return PortfolioState(
                total_positions=0,
                positions_by_symbol={},
                positions_by_regime={},
                total_exposure_usdt=0.0,
                exposure_by_symbol={},
                available_capital_usdt=balance,
                last_trade_time=0.0,
                last_trade_by_symbol={}
            )

    def has_open_position(self, symbol: str) -> bool:
        """Check if there's already an open position for this symbol"""
        try:
            if PAPER_TRADING:
                return False  # Allow multiple paper trades for testing
            
            if not BINANCE_AVAILABLE or not client:
                logger.warning(f"⚠️ [{symbol}] BINANCE_AVAILABLE={BINANCE_AVAILABLE}, client={client is not None}")
                return False
            
            positions = safe_futures_call('futures_position_information', symbol=symbol)
            logger.info(f"🔍 [{symbol}] Checking position - got {len(positions)} results")
            
            for pos in positions:
                position_amt = float(pos.get('positionAmt', 0))
                logger.info(f"   Position {pos.get('symbol')}: amt={position_amt}, entryPrice={pos.get('entryPrice')}")
                if abs(position_amt) > 0:
                    logger.info(f"📍 [{symbol}] FOUND existing position: {position_amt}")
                    return True
            
            logger.info(f"✅ [{symbol}] No existing position found")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to check position for {symbol}: {e}")
            return False  # Proceed with caution if check fails
    
    def has_tp_sl_orders(self, symbol: str) -> bool:
        """Check if TP/SL orders exist for this symbol"""
        try:
            if PAPER_TRADING:
                logger.info(f"⏭️ [{symbol}] Skipping TP/SL check (PAPER_TRADING)")
                return False
            
            if not BINANCE_AVAILABLE or not client:
                logger.warning(f"⚠️ [{symbol}] Cannot check TP/SL (no client)")
                return False
            
            open_orders = safe_futures_call('futures_get_open_orders', symbol=symbol)
            logger.info(f"📋 [{symbol}] Checking {len(open_orders)} open orders for TP/SL")
            
            # Check if any orders are TP or SL types
            for order in open_orders:
                order_type = order.get('type', '')
                logger.info(f"   Order: {order.get('orderId')} type={order_type} side={order.get('side')}")
                if order_type in ['TAKE_PROFIT_MARKET', 'STOP_MARKET', 'TAKE_PROFIT', 'STOP_LOSS']:
                    logger.info(f"🛡️ [{symbol}] Has existing TP/SL: {order_type}")
                    return True
            
            logger.info(f"❗ [{symbol}] NO TP/SL orders found!")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to check TP/SL orders for {symbol}: {e}")
            return False
    
    def set_tp_sl_for_existing(self, symbol: str, signal: Dict) -> bool:
        """Set TP/SL for an existing position based on new signal"""
        try:
            if not BINANCE_AVAILABLE or not client:
                return False
            
            # Get current position details
            positions = safe_futures_call('futures_position_information', symbol=symbol)
            position = None
            for pos in positions:
                if abs(float(pos.get('positionAmt', 0))) > 0:
                    position = pos
                    break
            
            if not position:
                logger.warning(f"⚠️ [{symbol}] No position found")
                return False
            
            position_amt = float(position.get('positionAmt', 0))
            entry_price = float(position.get('entryPrice', 0))
            current_leverage = int(position.get('leverage', 1))
            
            # Determine side
            side = "BUY" if position_amt > 0 else "SELL"
            
            # Get signal's leverage recommendation
            signal_leverage = int(signal.get('leverage', MAX_LEVERAGE))
            
            # Update leverage if different
            if current_leverage != signal_leverage:
                try:
                    safe_futures_call('futures_change_leverage', symbol=symbol, leverage=signal_leverage)
                    logger.info(f"📊 [{symbol}] Leverage updated: {current_leverage}x → {signal_leverage}x")
                except Exception as e:
                    logger.warning(f"⚠️ [{symbol}] Failed to update leverage: {e}")
            
            # Calculate TP/SL prices based on entry price using ExitBrain formulas
            volatility = signal.get('volatility_factor', 0.02)
            confidence = signal.get('confidence', 0.8)
            
            # Use ExitBrain v3.5 for intelligent TP/SL calculation
            tp_pct = None
            sl_pct = None
            
            if self.exit_brain:
                try:
                    # Create signal context
                    signal_context = SignalContext(
                        symbol=symbol,
                        side="long" if side == "BUY" else "short",
                        confidence=confidence,
                        entry_price=entry_price,
                        atr_value=volatility,
                        timestamp=time.time()
                    )
                    
                    # Build exit plan
                    exit_plan = self.exit_brain.build_exit_plan(
                        signal=signal_context,
                        pnl_trend=0.0,
                        symbol_risk=1.0,
                        margin_util=0.0,
                        exch_divergence=0.0,
                        funding_rate=0.0
                    )
                    
                    tp_pct = exit_plan.take_profit_pct
                    sl_pct = exit_plan.stop_loss_pct
                    
                    logger.info(
                        f"🧠 [{symbol}] ExitBrain TP/SL: {tp_pct*100:.2f}%/{sl_pct*100:.2f}% | "
                        f"{exit_plan.reasoning}"
                    )
                    
                except Exception as eb_error:
                    logger.warning(f"⚠️ [{symbol}] ExitBrain failed: {eb_error}, using fallback")
            
            # Fallback to leverage-based TP/SL if ExitBrain unavailable
            # 🎯 SAFE LEVELS - Account for funding costs (-0.75%) + spread (0.3%) = -1.05%
            if tp_pct is None or sl_pct is None:
                # Higher leverage = tighter ranges, but NEVER below funding costs
                if signal_leverage >= 10:
                    tp_pct = 0.018  # 1.8% TP (was 0.8%)
                    sl_pct = 0.012  # 1.2% SL (was 0.4%)
                elif signal_leverage >= 5:
                    tp_pct = 0.025  # 2.5% TP (was 1.0%)
                    sl_pct = 0.015  # 1.5% SL (was 0.5%)
                else:
                    tp_pct = 0.035  # 3.5% TP (was 1.5%)
                    sl_pct = 0.020  # 2.0% SL (was 0.7%)
                logger.info(f"🎯 [{symbol}] Safe TP/SL (accounts for funding): {tp_pct*100:.1f}%/{sl_pct*100:.1f}%")
            
            # Calculate prices based on position direction
            # ⚠️ CRITICAL: Add 0.2% buffer to account for mark price vs last price divergence
            # Mark price can be 2-5% off from last price, causing premature triggers
            mark_price_buffer = 0.002  # 0.2% safety buffer
            
            if side == "BUY":
                take_profit_price = entry_price * (1 + tp_pct + mark_price_buffer)
                stop_loss_price = entry_price * (1 - sl_pct - mark_price_buffer)
            else:  # SELL (SHORT)
                take_profit_price = entry_price * (1 - tp_pct - mark_price_buffer)
                stop_loss_price = entry_price * (1 + sl_pct + mark_price_buffer)
            
            # Get symbol info for price precision
            symbol_info = self.get_symbol_info(symbol)
            tick_size_str = symbol_info.get('tickSize', '0.01')
            tick_size = float(tick_size_str)
            
            # ✅ CRITICAL FIX: Calculate price precision from tickSize string (handles scientific notation)
            # Use original string from exchange, not float (avoids "1e-05" problem)
            if '.' in tick_size_str:
                price_precision = len(tick_size_str.rstrip('0').split('.')[1])
            else:
                price_precision = 0
            
            # DEBUG: Log BEFORE rounding
            logger.info(
                f"🔍 [{symbol}] BEFORE rounding: TP={take_profit_price:.10f}, SL={stop_loss_price:.10f}, "
                f"tickSize={tick_size}, precision={price_precision}"
            )
            
            # Round prices to exchange precision
            take_profit_price = round(take_profit_price, price_precision)
            stop_loss_price = round(stop_loss_price, price_precision)
            
            # ✅ CRITICAL FIX: Validate prices are not zero (would cause "Stop price less than zero" error)
            if take_profit_price <= 0 or stop_loss_price <= 0:
                logger.error(
                    f"❌ [{symbol}] Invalid TP/SL prices after rounding: TP={take_profit_price}, SL={stop_loss_price}. "
                    f"Entry={entry_price}, TP%={tp_pct*100:.2f}%, SL%={sl_pct*100:.2f}%, precision={price_precision}"
                )
                # Try with minimum price movement instead
                if side == "BUY":
                    take_profit_price = entry_price + (tick_size * max(1, int(entry_price * tp_pct / tick_size)))
                    stop_loss_price = entry_price - (tick_size * max(1, int(entry_price * sl_pct / tick_size)))
                else:  # SHORT
                    take_profit_price = entry_price - (tick_size * max(1, int(entry_price * tp_pct / tick_size)))
                    stop_loss_price = entry_price + (tick_size * max(1, int(entry_price * sl_pct / tick_size)))
                logger.info(f"🔧 [{symbol}] Adjusted to tick-based prices: TP={take_profit_price}, SL={stop_loss_price}")
            
            # DEBUG: Log calculated prices
            logger.info(
                f"🔍 [{symbol}] TP/SL Prices: entry={entry_price}, "
                f"TP={take_profit_price}, SL={stop_loss_price}, "
                f"TP%={tp_pct*100:.2f}%, SL%={sl_pct*100:.2f}%"
            )
            
            # ⚡ CHECK if TP/SL need updating before cancelling
            try:
                open_orders = safe_futures_call('futures_get_open_orders', symbol=symbol)
                existing_tp = None
                existing_sl = None
                
                for order in open_orders:
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT']:
                        existing_tp = float(order.get('stopPrice', 0))
                    elif order['type'] in ['STOP_MARKET', 'STOP']:
                        existing_sl = float(order.get('stopPrice', 0))
                
                # Check if TP/SL are close enough to target (within 0.1%)
                tp_needs_update = True
                sl_needs_update = True
                
                if existing_tp:
                    price_diff_pct = abs(existing_tp - take_profit_price) / entry_price
                    if price_diff_pct < 0.001:  # Less than 0.1% difference
                        tp_needs_update = False
                        logger.debug(f"✓ [{symbol}] TP already correct @ ${existing_tp:.6f}")
                
                if existing_sl:
                    price_diff_pct = abs(existing_sl - stop_loss_price) / entry_price
                    if price_diff_pct < 0.001:  # Less than 0.1% difference
                        sl_needs_update = False
                        logger.debug(f"✓ [{symbol}] SL already correct @ ${existing_sl:.6f}")
                
                # Only update if needed
                if not tp_needs_update and not sl_needs_update:
                    logger.debug(f"⏭️ [{symbol}] TP/SL already at correct levels, skipping update")
                    return True
                
                # Cancel orders that need updating
                for order in open_orders:
                    if order['type'] in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] and tp_needs_update:
                        try:
                            safe_futures_call('futures_cancel_order', symbol=symbol, orderId=order['orderId'])
                            logger.info(f"🗑️ [{symbol}] Cancelled old TP order")
                        except Exception as cancel_error:
                            logger.warning(f"⚠️ [{symbol}] Failed to cancel TP: {cancel_error}")
                    elif order['type'] in ['STOP_MARKET', 'STOP'] and sl_needs_update:
                        try:
                            safe_futures_call('futures_cancel_order', symbol=symbol, orderId=order['orderId'])
                            logger.info(f"🗑️ [{symbol}] Cancelled old SL order")
                        except Exception as cancel_error:
                            logger.warning(f"⚠️ [{symbol}] Failed to cancel SL: {cancel_error}")
            except Exception as e:
                logger.warning(f"⚠️ [{symbol}] Failed to check existing orders: {e}")
                # Continue anyway to place orders
                tp_needs_update = True
                sl_needs_update = True
            
            # Place TP order if needed
            if tp_needs_update:
                try:
                    tp_order = safe_futures_call('futures_create_order',
                        symbol=symbol,
                        side="SELL" if side == "BUY" else "BUY",
                        positionSide="LONG" if side == "BUY" else "SHORT",
                        type="TAKE_PROFIT_MARKET",
                        stopPrice=take_profit_price,
                        closePosition=True
                    )
                    logger.info(f"✅ [{symbol}] TP set @ ${take_profit_price} ({tp_pct*100:+.1f}%)")
                except Exception as e:
                    logger.error(f"❌ [{symbol}] Failed to set TP: {e}")
            
            # Place SL order if needed
            if sl_needs_update:
                try:
                    sl_order = safe_futures_call('futures_create_order',
                        symbol=symbol,
                        side="SELL" if side == "BUY" else "BUY",
                        positionSide="LONG" if side == "BUY" else "SHORT",
                        type="STOP_MARKET",
                        stopPrice=stop_loss_price,
                        closePosition=True
                    )
                    logger.info(f"✅ [{symbol}] SL set @ ${stop_loss_price} ({-sl_pct*100:.1f}%)")
                except Exception as e:
                    logger.error(f"❌ [{symbol}] Failed to set SL: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to set TP/SL for existing position {symbol}: {e}")
            return False

    def process_signal(self, signal: Dict) -> bool:
        """
        Process a single trading signal with P1-B Execution Policy.
        
        Flow: signal → allow_new_entry() → compute_order_size() → place_order()
        """
        try:
            symbol = signal.get("symbol", "").upper()
            side = signal.get("side", signal.get("action", "")).upper()  # Support both 'side' and 'action'
            confidence = signal.get("confidence", 0.0)
            pnl = signal.get("pnl", 0.0)
            price = signal.get("entry_price", signal.get("price", 0.0))  # Support both field names
            regime = signal.get("regime", "unknown")
            
            # Validation
            if not symbol or not side:
                logger.debug("⚠️ Invalid signal: missing symbol or side")
                return False
            
            if side not in ["BUY", "SELL", "CLOSE"]:
                logger.debug(f"⚠️ Invalid side: {side}")
                return False
            
            # P1-B: Use Execution Policy if available
            if self.execution_policy:
                # Build current portfolio state
                portfolio = self.build_portfolio_state()
                
                # Prepare intent for policy check
                intent = {
                    "symbol": symbol,
                    "side": side,
                    "confidence": confidence,
                    "entry_price": price,
                    "price": price,
                    "quantity": 0.0,  # Will be calculated by policy if allowed
                    "regime": regime,
                    "leverage": signal.get("leverage", MAX_LEVERAGE)
                }
                
                # Check if entry is allowed
                decision, reason = self.execution_policy.allow_new_entry(intent, portfolio)
                
                if decision not in [PolicyDecision.ALLOW_NEW_ENTRY, PolicyDecision.ALLOW_SCALE_IN]:
                    # Entry blocked - log and return
                    logger.info(
                        f"🚫 [{symbol}] Entry blocked: {decision.value} | {reason}"
                    )
                    
                    # If position exists, update TP/SL instead
                    if decision == PolicyDecision.BLOCK_EXISTING_POSITION and self.has_open_position(symbol):
                        logger.info(f"🔄 [{symbol}] Updating TP/SL for existing position")
                        return self.set_tp_sl_for_existing(symbol, signal)
                    
                    return False
                
                # Entry allowed - compute order size via policy
                risk_score = signal.get("risk_score", 1.0)
                qty = self.execution_policy.compute_order_size(intent, portfolio, risk_score)
                
                if qty <= 0:
                    logger.warning(f"⚠️ [{symbol}] Policy returned zero quantity")
                    return False
                
                # Update intent with calculated quantity
                intent["quantity"] = qty
                
                # Update last trade tracking
                now = time.time()
                if not hasattr(self, 'last_trade_time'):
                    self.last_trade_time = 0.0
                if not hasattr(self, 'last_trade_by_symbol'):
                    self.last_trade_by_symbol = {}
                
                self.last_trade_time = now
                self.last_trade_by_symbol[symbol] = now
                
                logger.info(
                    f"✅ [{symbol}] Policy approved: {decision.value} | "
                    f"qty={qty:.4f} | confidence={confidence:.2f} | {reason}"
                )
            
            else:
                # LEGACY LOGIC (fallback if policy unavailable)
                logger.warning(f"⚠️ [{symbol}] Using legacy execution logic (no policy)")
                
                # Check confidence threshold
                if confidence < CONFIDENCE_THRESHOLD:
                    logger.debug(
                        f"⚠️ Signal rejected: {symbol} confidence={confidence:.2f} "
                        f"< {CONFIDENCE_THRESHOLD}"
                    )
                    return False
                
                # Check if position already exists
                if self.has_open_position(symbol):
                    logger.info(f"🔄 [{symbol}] Updating TP/SL for existing position")
                    return self.set_tp_sl_for_existing(symbol, signal)
                
                # Get balance and calculate position size (old method)
                balance = self.get_balance()
                qty = self.calculate_position_size(symbol, balance, confidence)
                
                if qty < 0.001:  # Minimum order size
                    logger.warning(f"⚠️ Position size too small: {qty}")
                    return False
            
            # Check drawdown circuit breaker (both policy and legacy)
            if self.check_drawdown(signal):
                return False
            
            # Calculate dynamic leverage based on confidence and market conditions
            volatility = signal.get("volatility", 0.02)  # Default 2% if not provided
            leverage = self.calculate_dynamic_leverage(symbol, confidence, volatility)
            
            # Place order with dynamic leverage
            order = self.place_order(symbol, side, qty, price, leverage)
            
            if order:
                # Log successful trade
                self.log_trade(symbol, side, qty, price, confidence, pnl)
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
                    
                    # Filter out HOLD signals and invalid symbols
                    symbol = signal.get('symbol', '')
                    if signal.get('side') != 'HOLD' and symbol not in INVALID_SYMBOLS:
                        # P0 PROOF LOG: Intent received
                        logger.info(f"🎯 INTENT_RECEIVED | stream_id={message_id} | symbol={symbol} | "
                                  f"side={signal.get('side')} | confidence={signal.get('confidence', 0):.3f}")
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
        # P0 FIX: Startup self-test before running executor
        if not PAPER_TRADING and BINANCE_AVAILABLE and client:
            logger.info("🔍 Running startup self-test...")
            try:
                # Test 1: Server time
                server_time = safe_futures_call('futures_time')
                logger.info(f"✅ Server time: {server_time['serverTime']}")
                
                # Test 2: Exchange info
                exchange_info = safe_futures_call('futures_exchange_info')
                logger.info(f"✅ Exchange info: {len(exchange_info['symbols'])} symbols available")
                
                # Test 3: Account endpoint
                account = safe_futures_call('futures_account')
                logger.info(f"✅ Account balance: {account['totalWalletBalance']} USDT")
                
                logger.info("✅ Startup self-test PASSED")
            except Exception as test_error:
                logger.error(f"❌ FATAL_BINANCE_UNAVAILABLE: Startup self-test FAILED: {test_error}")
                logger.error("❌ Cannot connect to Binance API - check credentials and network")
                sys.exit(1)
        
        executor = AutoExecutor()
        executor.run_execution_loop()
    except Exception as e:
        logger.critical(f"🚨 FATAL: Executor failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
