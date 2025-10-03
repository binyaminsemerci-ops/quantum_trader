"""
Binance Trading Engine - Handles automated trading with XGBoost AI model
Supports USDC-M Futures and Cross Margin trading on both USDC and USDT pairs
"""

import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime, timezone

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    # Binance package not installed - create mock classes for development
    class Client:
        def __init__(self, *args, **kwargs):
            pass

    class BinanceAPIException(Exception):
        pass


from ai_engine.agents.xgb_agent import XGBAgent
from backend.database import session_scope, Trade, TradeLog
from config.config import load_config

logger = logging.getLogger(__name__)


class BinanceTradeEngine:
    """
    Main trading engine that connects XGBoost AI decisions to Binance API
    Handles both USDC-M futures and cross margin spot trading
    """

    def __init__(self):
        self.config = load_config()
        self.client = None
        self.ai_agent = XGBAgent()
        self.is_running = False
        self.active_symbols = []

        # Trading parameters
        self.max_position_size_usdc = 100.0  # Max position size in USDC
        self.min_confidence_threshold = 0.65  # Minimum AI confidence to trade
        self.risk_per_trade = 0.02  # 2% risk per trade

        # Initialize Binance client
        self._init_binance_client()
        # Gate real order execution behind enable_real_trading flag
        self.real_trading_enabled = bool(self.config.enable_real_trading)

    def _init_binance_client(self):
        """Initialize Binance client with API keys"""
        api_key = self.config.binance_api_key
        api_secret = self.config.binance_api_secret

        if not api_key or not api_secret:
            logger.error("Binance API keys not found in config!")
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET must be set")

        try:
            testnet_flag = bool(self.config.binance_use_testnet)
            self.client = Client(
                api_key, api_secret, testnet=testnet_flag
            )  # use testnet when configured

            # Test connection
            account_info = {}
            try:
                account_info = self.client.get_account()
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Could not fetch account info immediately: {e}")
            logger.info(
                "Binance client initialized successfully (%s)",
                "testnet" if testnet_flag else "live",
            )
            if account_info:
                logger.info(
                    f"Account status: {account_info.get('accountType', 'Unknown')}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize Binance client: {e}")
            raise

    def get_trading_symbols(self) -> List[str]:
        """Get list of symbols to trade - both USDC and USDT futures"""
        usdc_futures = [
            "BTCUSDC",
            "ETHUSDC",
            "SOLUSDC",
            "ADAUSDC",
            "XRPUSDC",
            "DOGEUSDC",
            "AVAXUSDC",
            "DOTUSDC",
            "LINKUSDC",
            "MATICUSDC",
        ]
        usdt_futures = [
            "BTCUSDT",
            "ETHUSDT",
            "SOLUSDT",
            "ADAUSDT",
            "XRPUSDT",
            "DOGEUSDT",
            "AVAXUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "MATICUSDT",
        ]
        spot_pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "ADAUSDC"]

        symbols = set(usdc_futures + usdt_futures + spot_pairs)

        # Merge configured symbol groups (mainbase, layer1, layer2)
        try:
            cfg = self.config
            for grp in (cfg.mainbase_symbols, cfg.layer1_symbols, cfg.layer2_symbols):
                for sym in grp:
                    if sym:  # ensure non-empty
                        symbols.add(sym)
        except Exception:
            pass

        # Ensure BNB and NEAR, OP, ARB etc if present in config but missing
        # Already covered by merge, here just finalize sorted for determinism
        final_list = sorted(symbols)
        return final_list

    def get_account_balance(self) -> Dict[str, float]:
        """Get current account balances for USDC and USDT"""
        try:
            # Futures balances
            futures_balance = self.client.futures_account_balance()

            # Spot balances
            spot_account = self.client.get_account()

            balances = {}

            # Extract futures balances
            for balance in futures_balance:
                asset = balance["asset"]
                if asset in ["USDC", "USDT"]:
                    balances[f"{asset}_futures"] = float(balance["balance"])

            # Extract spot balances
            for balance in spot_account["balances"]:
                asset = balance["asset"]
                if asset in ["USDC", "USDT"]:
                    balances[f"{asset}_spot"] = float(balance["free"])

            return balances

        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return {}

    def get_position_size(self, symbol: str, price: float, confidence: float) -> float:
        """Calculate position size based on risk management and confidence"""
        try:
            # Get available balance
            balances = self.get_account_balance()

            # Determine which balance to use based on symbol
            if "USDC" in symbol:
                if symbol.endswith("USDC"):  # Futures
                    available = balances.get("USDC_futures", 0)
                else:  # Spot
                    available = balances.get("USDC_spot", 0)
            else:  # USDT pairs
                if symbol.endswith("USDT"):
                    available = balances.get("USDT_futures", 0)
                else:
                    available = balances.get("USDT_spot", 0)

            # Calculate position size based on risk and confidence
            risk_amount = available * self.risk_per_trade * confidence
            position_size = min(
                risk_amount / price, self.max_position_size_usdc / price
            )

            # Apply minimum order size filters (get from exchange info)
            min_qty = self.get_min_order_size(symbol)
            position_size = max(position_size, min_qty)

            return round(position_size, 6)  # Round to 6 decimal places

        except Exception as e:
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0.0

    def get_min_order_size(self, symbol: str) -> float:
        """Get minimum order size for symbol from exchange info"""
        try:
            if symbol.endswith(("USDC", "USDT")):  # Futures
                info = self.client.futures_exchange_info()
                for s in info["symbols"]:
                    if s["symbol"] == symbol:
                        for filter_item in s["filters"]:
                            if filter_item["filterType"] == "LOT_SIZE":
                                return float(filter_item["minQty"])
            else:  # Spot
                info = self.client.get_exchange_info()
                for s in info["symbols"]:
                    if s["symbol"] == symbol:
                        for filter_item in s["filters"]:
                            if filter_item["filterType"] == "LOT_SIZE":
                                return float(filter_item["minQty"])

            return 0.001  # Default minimum

        except Exception as e:
            logger.error(f"Error getting min order size for {symbol}: {e}")
            return 0.001

    def execute_trade(
        self, symbol: str, action: str, quantity: float, confidence: float
    ) -> Dict[str, Any]:
        """Execute a trade on Binance"""
        try:
            if not self.real_trading_enabled:
                logger.info(
                    "[DRY-RUN] %s %s qty=%s (confidence=%.2f) skipped because ENABLE_REAL_TRADING is False",
                    action,
                    symbol,
                    quantity,
                    confidence,
                )
                return {
                    "success": True,
                    "dry_run": True,
                    "symbol": symbol,
                    "side": action,
                    "quantity": quantity,
                    "confidence": confidence,
                    "note": "Real trading disabled",
                }
            # Determine if this is futures or spot
            is_futures = symbol.endswith(("USDC", "USDT"))

            # Map AI action to Binance side
            side = "BUY" if action == "BUY" else "SELL"

            # Execute order
            if is_futures:
                # Futures order
                order = self.client.futures_create_order(
                    symbol=symbol, side=side, type="MARKET", quantity=quantity
                )
            else:
                # Spot order
                order = self.client.order_market(
                    symbol=symbol, side=side, quantity=quantity
                )

            # Log the trade
            self._log_trade(symbol, action, quantity, confidence, order)

            logger.info(
                f"Trade executed: {action} {quantity} {symbol} (confidence: {confidence:.2f})"
            )

            return {
                "success": True,
                "order_id": order.get("orderId"),
                "symbol": symbol,
                "side": action,
                "quantity": quantity,
                "confidence": confidence,
                "order_details": order,
            }

        except BinanceAPIException as e:
            logger.error(f"Binance API error executing {action} {symbol}: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Error executing trade {action} {symbol}: {e}")
            return {"success": False, "error": str(e)}

    def _log_trade(
        self, symbol: str, action: str, quantity: float, confidence: float, order: Dict
    ):
        """Log trade to database"""
        try:
            price = (
                float(order.get("fills", [{}])[0].get("price", 0))
                if order.get("fills")
                else 0
            )

            with session_scope() as session:
                # Create trade record
                trade = Trade(
                    symbol=symbol,
                    side=action,
                    qty=quantity,
                    price=price,
                    timestamp=datetime.now(timezone.utc),
                )

                # Create trade log
                trade_log = TradeLog(
                    symbol=symbol,
                    side=action,
                    qty=quantity,
                    price=price,
                    status="FILLED",
                    reason=f"AI Decision (confidence: {confidence:.2f})",
                    timestamp=datetime.now(timezone.utc),
                )

                session.add(trade)
                session.add(trade_log)

        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    async def get_market_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get OHLCV data for AI analysis"""
        try:
            # Get klines (candlestick data)
            if symbol.endswith(("USDC", "USDT")):  # Futures
                klines = self.client.futures_klines(
                    symbol=symbol, interval="1m", limit=limit
                )
            else:  # Spot
                klines = self.client.get_klines(
                    symbol=symbol, interval="1m", limit=limit
                )

            # Convert to format expected by AI agent
            ohlcv_data = []
            for kline in klines:
                ohlcv_data.append(
                    {
                        "timestamp": kline[0],
                        "open": float(kline[1]),
                        "high": float(kline[2]),
                        "low": float(kline[3]),
                        "close": float(kline[4]),
                        "volume": float(kline[5]),
                    }
                )

            return ohlcv_data

        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return []

    async def analyze_and_trade_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol with AI and execute trades if confidence is high enough"""
        try:
            # Get market data
            ohlcv_data = await self.get_market_data(symbol)
            if not ohlcv_data:
                return {"symbol": symbol, "action": "NO_DATA"}

            # Get AI prediction
            prediction = self.ai_agent.predict_for_symbol(ohlcv_data)
            action = prediction.get("action", "HOLD")
            confidence = prediction.get("score", 0.0)

            logger.info(f"{symbol}: AI says {action} (confidence: {confidence:.2f})")

            # Check if confidence meets threshold
            if confidence < self.min_confidence_threshold:
                return {
                    "symbol": symbol,
                    "action": "HOLD",
                    "reason": f"Low confidence ({confidence:.2f} < {self.min_confidence_threshold})",
                    "confidence": confidence,
                }

            # Execute trade if not HOLD
            if action in ["BUY", "SELL"]:
                # Get current price
                current_price = ohlcv_data[-1]["close"]

                # Calculate position size
                quantity = self.get_position_size(symbol, current_price, confidence)

                if quantity > 0:
                    # Execute the trade
                    trade_result = self.execute_trade(
                        symbol, action, quantity, confidence
                    )
                    trade_result["ai_prediction"] = prediction
                    return trade_result
                else:
                    return {
                        "symbol": symbol,
                        "action": "SKIP",
                        "reason": "Insufficient balance or position size too small",
                        "confidence": confidence,
                    }

            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "ai_prediction": prediction,
            }

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {"symbol": symbol, "action": "ERROR", "error": str(e)}

    async def run_trading_cycle(self):
        """Run one complete trading cycle across all symbols"""
        logger.info("Starting trading cycle...")

        # Get symbols to analyze
        symbols = self.get_trading_symbols()

        # Analyze all symbols concurrently (with rate limiting)
        semaphore = asyncio.Semaphore(5)  # Limit concurrent API calls

        async def analyze_with_semaphore(symbol):
            async with semaphore:
                return await self.analyze_and_trade_symbol(symbol)

        tasks = [analyze_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_trades = 0
        total_analyzed = 0

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in analysis: {result}")
                continue

            total_analyzed += 1
            if result.get("success"):
                successful_trades += 1

        logger.info(
            f"Trading cycle complete: {total_analyzed} symbols analyzed, {successful_trades} trades executed"
        )
        return results

    async def start_trading(self, cycle_interval_minutes: int = 5):
        """Start the automated trading loop"""
        logger.info(
            "Starting automated trading with %s minute intervals (real_trading=%s)",
            cycle_interval_minutes,
            self.real_trading_enabled,
        )
        self.is_running = True

        while self.is_running:
            try:
                # Run trading cycle
                await self.run_trading_cycle()

                # Wait for next cycle
                await asyncio.sleep(cycle_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def stop_trading(self):
        """Stop the automated trading loop"""
        logger.info("Stopping automated trading")
        self.is_running = False

    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status and statistics"""
        try:
            balances = self.get_account_balance()

            # Get recent trades from database
            with session_scope() as session:
                recent_trades = (
                    session.query(Trade)
                    .order_by(Trade.timestamp.desc())
                    .limit(10)
                    .all()
                )

            return {
                "is_running": self.is_running,
                "balances": balances,
                "recent_trades": [
                    {
                        "symbol": t.symbol,
                        "side": t.side,
                        "qty": t.qty,
                        "price": t.price,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    }
                    for t in recent_trades
                ],
                "ai_model_loaded": self.ai_agent.model is not None,
                "trading_symbols_count": len(self.get_trading_symbols()),
            }

        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            return {"error": str(e)}


# Global trading engine instance
trading_engine = None


def get_trading_engine() -> BinanceTradeEngine:
    """Get or create the global trading engine instance"""
    global trading_engine
    if trading_engine is None:
        trading_engine = BinanceTradeEngine()
    return trading_engine
