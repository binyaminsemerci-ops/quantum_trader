#!/usr/bin/env python3
"""
AI Auto Trading Service

Komplett implementering av AI-drevet automatisk handel:
- Integrerer XGB Agent for prediksjoner
- Implementerer risk management
- Automatisk posisjonsstørrelse basert på AI-tillit
- Real-time markedsdata og handelssignaler
- Automatisk stoppordre og take-profit
- Logging og overvåking av AI-handel
"""

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading
from dataclasses import dataclass, asdict
import sqlite3

# Import existing components (adjust paths for backend location)
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.continuous_learning_engine import create_continuous_learning_service

# Default symbols for testing (fallback if config import fails)
DEFAULT_SYMBOLS = ["BTCUSDC", "ETHUSDC", "ADAUSDC"]

try:
    from ai_engine.agents.xgb_agent import make_default_agent, XGBAgent
except ImportError:
    # Mock XGBAgent for testing
    class XGBAgent:
        def __init__(self):
            self.model_name = "mock_xgb"

        def predict_for_symbol(self, symbol: str, limit: int = 100):
            # Mock prediction - random but realistic values
            import random

            return {
                "prediction": random.uniform(-0.05, 0.05),  # -5% to +5%
                "confidence": random.uniform(0.5, 0.95),
                "features_used": limit,
                "model_version": "mock_v1.0",
            }

    def make_default_agent():
        return XGBAgent()


logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """AI trading signal with metadata."""

    symbol: str
    action: str  # BUY, SELL, HOLD
    confidence: float  # 0.0 - 1.0
    predicted_return: float
    timestamp: datetime
    features_used: List[str]
    model_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


@dataclass
class TradeExecution:
    """Trade execution record."""

    signal_id: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    order_id: Optional[str]
    status: str  # pending, filled, cancelled, rejected
    timestamp: datetime
    ai_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "timestamp": self.timestamp.isoformat()}


class AIAutoTradingService:
    """Main AI Auto Trading Service."""

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        max_position_size: float = 1000.0,  # Max USD per position
        min_confidence: float = 0.6,  # Minimum AI confidence
        max_daily_trades: int = 50,  # Daily trade limit
        stop_loss_pct: float = 0.02,  # 2% stop loss
        take_profit_pct: float = 0.04,
    ):  # 4% take profit

        self.symbols = (
            symbols or list(DEFAULT_SYMBOLS)[:5]
        )  # Limit to 5 symbols initially
        self.max_position_size = max_position_size
        self.min_confidence = min_confidence
        self.max_daily_trades = max_daily_trades
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # AI Agent
        self.ai_agent: Optional[XGBAgent] = None
        self.model_loaded = False

        # State tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.is_running = False
        self.trading_thread: Optional[threading.Thread] = None

        # Performance tracking
        self.signals_generated = 0
        self.trades_executed = 0
        self.successful_trades = 0
        self.total_pnl = 0.0

        # Database for persistence
        self.db_path = Path("ai_trading.db")
        self._init_database()

        # Lock for thread safety
        self._lock = threading.Lock()

        # Continuous learning engine integration
        self.continuous_learning_engine = create_continuous_learning_service(
            self.symbols
        )
        self.live_data_enabled = True

        # Configuration for external access/updates
        self.config = {
            "position_size": self.max_position_size,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "min_confidence": self.min_confidence,
            "max_positions": len(self.symbols),
            "risk_limit": self.max_position_size * len(self.symbols),
            "live_data_enabled": self.live_data_enabled,
        }

    def _init_database(self):
        """Initialize trading database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    predicted_return REAL,
                    timestamp TEXT NOT NULL,
                    features_used TEXT,
                    model_version TEXT
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    ai_confidence REAL NOT NULL
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    signals_generated INTEGER,
                    trades_executed INTEGER,
                    successful_trades INTEGER,
                    total_pnl REAL,
                    accuracy_rate REAL,
                    timestamp TEXT NOT NULL
                )
            """
            )

    def initialize_ai_agent(self) -> bool:
        """Initialize and load AI agent."""
        try:
            logger.info("🤖 Loading AI agent...")
            self.ai_agent = make_default_agent()

            if self.ai_agent.model is not None:
                self.model_loaded = True
                logger.info("✅ AI model loaded successfully")
                return True
            else:
                logger.warning("⚠️ AI model not found - will use fallback predictions")
                self.model_loaded = False
                return True  # Can still work with fallback

        except Exception as e:
            logger.error(f"❌ Failed to load AI agent: {e}")
            return False

    def get_market_data(
        self, symbol: str, limit: int = 100
    ) -> Optional[List[Dict[str, Any]]]:
        """Get recent market data for symbol."""
        try:
            # Mock market data for now - replace with real exchange API
            import random

            base_price = 50000 if "BTC" in symbol else 3000 if "ETH" in symbol else 100
            data = []

            for i in range(limit):
                timestamp = datetime.now(timezone.utc) - timedelta(minutes=i)
                price_change = random.uniform(-0.02, 0.02)  # ±2% random walk
                if i == 0:
                    price = base_price
                else:
                    price = data[-1]["close"] * (1 + price_change)

                data.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "open": price * (1 + random.uniform(-0.001, 0.001)),
                        "high": price * (1 + abs(random.uniform(0, 0.01))),
                        "low": price * (1 - abs(random.uniform(0, 0.01))),
                        "close": price,
                        "volume": random.uniform(100, 1000),
                    }
                )

            return list(reversed(data))  # Oldest first

        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    def generate_trading_signal(self, symbol: str) -> Optional[TradingSignal]:
        """Generate AI trading signal for symbol."""
        try:
            # Get market data
            ohlcv_data = self.get_market_data(
                symbol, limit=120
            )  # Need enough for indicators
            if not ohlcv_data:
                return None

            # Get AI prediction
            if not self.ai_agent:
                logger.error("AI agent not initialized")
                return None

            prediction = self.ai_agent.predict_for_symbol(ohlcv_data)

            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                action=prediction["action"],
                confidence=prediction["score"],
                predicted_return=prediction.get("predicted_return", 0.0),
                timestamp=datetime.now(timezone.utc),
                features_used=prediction.get("features", []),
                model_version=prediction.get("model_version", "xgb_v1"),
            )

            # Log signal to database
            self._log_signal(signal)

            with self._lock:
                self.signals_generated += 1

            logger.info(
                f"📊 Signal for {symbol}: {signal.action} (confidence: {signal.confidence:.2f})"
            )

            return signal

        except Exception as e:
            logger.error(f"Failed to generate signal for {symbol}: {e}")
            return None

    def _log_signal(self, signal: TradingSignal):
        """Log signal to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trading_signals 
                    (symbol, action, confidence, predicted_return, timestamp, features_used, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        signal.symbol,
                        signal.action,
                        signal.confidence,
                        signal.predicted_return,
                        signal.timestamp.isoformat(),
                        json.dumps(signal.features_used),
                        signal.model_version,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")

    def calculate_position_size(
        self, signal: TradingSignal, current_price: float
    ) -> float:
        """Calculate position size based on AI confidence and risk management."""
        try:
            # Base position size
            base_size = self.max_position_size

            # Adjust by confidence (higher confidence = larger position)
            confidence_multiplier = signal.confidence

            # Adjust by predicted return (higher return = larger position, but capped)
            return_multiplier = min(1.5, max(0.5, 1 + signal.predicted_return))

            # Calculate USD amount
            usd_amount = base_size * confidence_multiplier * return_multiplier

            # Convert to quantity
            quantity = usd_amount / current_price

            logger.debug(
                f"Position size calculation: {usd_amount:.2f} USD = {quantity:.6f} {signal.symbol}"
            )

            return quantity

        except Exception as e:
            logger.error(f"Failed to calculate position size: {e}")
            return 0.0

    def should_execute_trade(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Determine if trade should be executed based on risk rules."""

        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            return (
                False,
                f"Confidence {signal.confidence:.2f} below minimum {self.min_confidence}",
            )

        # Check daily trade limit
        today = datetime.now(timezone.utc).date()
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today

        if self.daily_trade_count >= self.max_daily_trades:
            return False, f"Daily trade limit {self.max_daily_trades} reached"

        # Check if already have position in this symbol
        if signal.symbol in self.active_positions:
            return False, f"Already have active position in {signal.symbol}"

        # Only execute BUY/SELL signals, not HOLD
        if signal.action == "HOLD":
            return False, "HOLD signal - no action needed"

        return True, "All checks passed"

    def execute_trade(self, signal: TradingSignal) -> Optional[TradeExecution]:
        """Execute trade based on signal."""
        try:
            # Get current price (mock for now)
            market_data = self.get_market_data(signal.symbol, limit=1)
            if not market_data:
                logger.error(f"No market data for {signal.symbol}")
                return None

            current_price = market_data[0]["close"]

            # Check if should execute
            should_execute, reason = self.should_execute_trade(signal)
            if not should_execute:
                logger.info(f"Skipping trade for {signal.symbol}: {reason}")
                return None

            # Calculate position size
            quantity = self.calculate_position_size(signal, current_price)
            if quantity <= 0:
                logger.warning(f"Invalid quantity for {signal.symbol}: {quantity}")
                return None

            # Create trade execution record
            trade = TradeExecution(
                signal_id=f"{signal.symbol}_{int(signal.timestamp.timestamp())}",
                symbol=signal.symbol,
                side=signal.action.lower(),
                quantity=quantity,
                price=current_price,
                order_id=None,  # Will be filled by exchange
                status="pending",
                timestamp=datetime.now(timezone.utc),
                ai_confidence=signal.confidence,
            )

            # Simulate order execution (replace with real exchange API)
            success = self._simulate_order_execution(trade)

            if success:
                trade.status = "filled"
                trade.order_id = f"order_{int(time.time())}"

                # Track position
                self.active_positions[signal.symbol] = {
                    "side": trade.side,
                    "quantity": trade.quantity,
                    "entry_price": trade.price,
                    "timestamp": trade.timestamp,
                    "stop_loss": self._calculate_stop_loss(trade),
                    "take_profit": self._calculate_take_profit(trade),
                }

                with self._lock:
                    self.trades_executed += 1
                    self.daily_trade_count += 1

                logger.info(
                    f"✅ Trade executed: {trade.side.upper()} {trade.quantity:.6f} {trade.symbol} @ {trade.price:.2f}"
                )
            else:
                trade.status = "rejected"
                logger.warning(f"❌ Trade rejected for {signal.symbol}")

            # Log trade
            self._log_trade(trade)

            return trade

        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            return None

    def _simulate_order_execution(self, trade: TradeExecution) -> bool:
        """Simulate order execution (replace with real exchange API)."""
        # Simple simulation - 95% success rate
        import random

        return random.random() > 0.05

    def _calculate_stop_loss(self, trade: TradeExecution) -> float:
        """Calculate stop loss price."""
        if trade.side == "buy":
            return trade.price * (1 - self.stop_loss_pct)
        else:  # sell
            return trade.price * (1 + self.stop_loss_pct)

    def _calculate_take_profit(self, trade: TradeExecution) -> float:
        """Calculate take profit price."""
        if trade.side == "buy":
            return trade.price * (1 + self.take_profit_pct)
        else:  # sell
            return trade.price * (1 - self.take_profit_pct)

    def _log_trade(self, trade: TradeExecution):
        """Log trade to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trade_executions 
                    (signal_id, symbol, side, quantity, price, order_id, status, timestamp, ai_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trade.signal_id,
                        trade.symbol,
                        trade.side,
                        trade.quantity,
                        trade.price,
                        trade.order_id,
                        trade.status,
                        trade.timestamp.isoformat(),
                        trade.ai_confidence,
                    ),
                )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def check_positions(self):
        """Check active positions for stop loss/take profit."""
        try:
            for symbol, position in list(self.active_positions.items()):
                # Get current price
                market_data = self.get_market_data(symbol, limit=1)
                if not market_data:
                    continue

                current_price = market_data[0]["close"]
                side = position["side"]

                # Check stop loss
                should_close = False
                reason = ""

                if side == "buy":
                    if current_price <= position["stop_loss"]:
                        should_close = True
                        reason = "Stop loss triggered"
                    elif current_price >= position["take_profit"]:
                        should_close = True
                        reason = "Take profit triggered"
                else:  # sell position
                    if current_price >= position["stop_loss"]:
                        should_close = True
                        reason = "Stop loss triggered"
                    elif current_price <= position["take_profit"]:
                        should_close = True
                        reason = "Take profit triggered"

                if should_close:
                    self._close_position(symbol, position, current_price, reason)

        except Exception as e:
            logger.error(f"Error checking positions: {e}")

    def _close_position(
        self, symbol: str, position: Dict[str, Any], close_price: float, reason: str
    ):
        """Close position and calculate P&L."""
        try:
            entry_price = position["entry_price"]
            quantity = position["quantity"]
            side = position["side"]

            # Calculate P&L
            if side == "buy":
                pnl = (close_price - entry_price) * quantity
            else:  # sell
                pnl = (entry_price - close_price) * quantity

            # Update performance tracking
            with self._lock:
                self.total_pnl += pnl
                if pnl > 0:
                    self.successful_trades += 1

            # Remove from active positions
            del self.active_positions[symbol]

            logger.info(f"🔚 Position closed: {symbol} - {reason} - P&L: {pnl:.2f} USD")

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        with self._lock:
            accuracy_rate = (
                self.successful_trades / max(1, self.trades_executed)
            ) * 100

            return {
                "signals_generated": self.signals_generated,
                "trades_executed": self.trades_executed,
                "successful_trades": self.successful_trades,
                "accuracy_rate": accuracy_rate,
                "total_pnl": self.total_pnl,
                "active_positions": len(self.active_positions),
                "daily_trade_count": self.daily_trade_count,
                "model_loaded": self.model_loaded,
                "is_running": self.is_running,
            }

    def trading_loop(self):
        """Main trading loop."""
        logger.info("🚀 AI Auto Trading loop started")

        while self.is_running:
            try:
                # Generate signals for all symbols
                for symbol in self.symbols:
                    if not self.is_running:  # Check if should stop
                        break

                    signal = self.generate_trading_signal(symbol)
                    if signal:
                        self.execute_trade(signal)

                    # Small delay between symbols
                    time.sleep(1)

                # Check existing positions
                if self.is_running:
                    self.check_positions()

                # Wait before next cycle (e.g., every 30 seconds)
                if self.is_running:
                    time.sleep(30)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(10)  # Wait before retrying

        logger.info("🛑 AI Auto Trading loop stopped")

    def start_trading(self, symbols: Optional[List[str]] = None) -> bool:
        """Start auto trading with continuous learning and live data feeds."""
        if self.is_running:
            logger.warning("Trading already running")
            return False

        # Update symbols if provided
        if symbols:
            self.symbols = symbols
            # Update continuous learning engine with new symbols
            self.continuous_learning_engine = create_continuous_learning_service(
                self.symbols
            )
            # Update config as well
            self.config["max_positions"] = len(symbols)
            self.config["risk_limit"] = self.max_position_size * len(symbols)

        # Initialize AI agent
        if not self.initialize_ai_agent():
            logger.error("Failed to initialize AI agent")
            return False

        # Start continuous learning engine for live data feeds
        if self.live_data_enabled:
            try:
                self.continuous_learning_engine.start_continuous_learning()
                logger.info("🚀 Continuous Learning Engine started!")
                logger.info("🐦 Live Twitter sentiment monitoring: ACTIVE")
                logger.info("📊 Real-time market data feeds: ACTIVE")
                logger.info("🤖 Continuous model retraining: ACTIVE")
            except Exception as e:
                logger.error(f"Failed to start continuous learning: {e}")
                # Continue with trading even if learning engine fails

        # Start trading loop in background thread
        self.is_running = True
        self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.trading_thread.start()

        logger.info(f"✅ AI Auto Trading started for symbols: {self.symbols}")
        logger.info(
            "🔄 System now continuously learning from live Twitter, news & market data!"
        )
        return True

    def stop_trading(self):
        """Stop auto trading and continuous learning."""
        if not self.is_running:
            logger.warning("Trading not running")
            return

        self.is_running = False

        # Stop continuous learning engine
        if hasattr(self, "continuous_learning_engine") and self.live_data_enabled:
            try:
                self.continuous_learning_engine.stop_continuous_learning()
                logger.info("🛑 Continuous Learning Engine stopped")
            except Exception as e:
                logger.error(f"Error stopping continuous learning: {e}")

        if self.trading_thread:
            self.trading_thread.join(timeout=5)

        logger.info("⏹️ AI Auto Trading stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current AI trading status and performance metrics."""
        with self._lock:
            # Calculate win rate
            win_rate = (self.successful_trades / max(1, self.trades_executed)) * 100

            # Get continuous learning status if available
            learning_status = {}
            if hasattr(self, "continuous_learning_engine") and self.live_data_enabled:
                try:
                    learning_status = (
                        self.continuous_learning_engine.get_learning_status()
                    )
                except Exception as e:
                    logger.error(f"Error getting learning status: {e}")
                    learning_status = {"error": str(e)}

            return {
                "is_running": self.is_running,
                "model_loaded": self.model_loaded,
                "symbols": self.symbols,
                "total_signals": self.signals_generated,
                "successful_trades": self.successful_trades,
                "total_trades": self.trades_executed,
                "total_pnl": round(self.total_pnl, 2),
                "win_rate": round(win_rate, 2),
                "active_positions": len(self.active_positions),
                "daily_trades_today": self.daily_trade_count,
                "max_daily_trades": self.max_daily_trades,
                "position_size": self.max_position_size,
                "min_confidence": self.min_confidence,
                "stop_loss_pct": self.stop_loss_pct * 100,
                "take_profit_pct": self.take_profit_pct * 100,
                "continuous_learning": learning_status,
                "live_data_enabled": self.live_data_enabled,
            }

    def get_recent_signals(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trading signals."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM trading_signals 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get recent signals: {e}")
            return []

    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT * FROM trade_executions 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """,
                    (limit,),
                )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            return []

    def get_recent_executions(
        self, symbol: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent trade executions - alias for get_recent_trades with optional symbol filter."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol:
                    cursor = conn.execute(
                        """
                        SELECT * FROM trade_executions 
                        WHERE symbol = ?
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """,
                        (symbol, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM trade_executions 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """,
                        (limit,),
                    )

                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get recent executions: {e}")
            return []


# Global instance for use in backend
_ai_trading_service: Optional[AIAutoTradingService] = None


def get_ai_trading_service() -> AIAutoTradingService:
    """Get global AI trading service instance."""
    global _ai_trading_service
    if _ai_trading_service is None:
        _ai_trading_service = AIAutoTradingService()
    return _ai_trading_service


if __name__ == "__main__":
    # Demo/test mode
    logging.basicConfig(level=logging.INFO)

    service = AIAutoTradingService(
        symbols=["BTCUSDT", "ETHUSDT"],
        max_position_size=100.0,  # Small amounts for testing
        min_confidence=0.5,
    )

    print("🤖 Starting AI Auto Trading Service demo...")

    if service.start_trading():
        try:
            # Run for 2 minutes in demo mode
            time.sleep(120)

            # Show results
            stats = service.get_performance_stats()
            print("\n📊 Performance Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

            print("\n📈 Recent Signals:")
            signals = service.get_recent_signals(5)
            for signal in signals:
                print(
                    f"  {signal['symbol']} - {signal['action']} (confidence: {signal['confidence']:.2f})"
                )

        finally:
            service.stop_trading()
    else:
        print("❌ Failed to start AI trading service")
