"""
AUTONOMOUS TRADING BOT
=====================

This module connects AI signals to actual trading execution.
Creates the bridge between AI predictions and live trading.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

# Import existing components  
from backend.routes.live_ai_signals import get_live_ai_signals
from backend.utils.exchanges import get_exchange_client
from backend.utils.risk import RiskManager
from backend.database import SessionLocal
from .market_config import (
    get_trading_pairs, get_market_config, get_risk_config,
    get_volume_weighted_pairs, is_layer1_layer2_token,
    get_optimal_market_for_token, MARKET_CONFIGS
)
# Import will be done dynamically to avoid circular imports

logger = logging.getLogger(__name__)


class AutonomousTradingBot:
    """
    Autonomous trading bot that:
    1. Monitors AI signals continuously
    2. Executes trades based on signals
    3. Manages positions and risk
    4. Logs all activities to database
    """
    
    def __init__(self,
                 balance: float = 10000.0,
                 risk_per_trade: float = 0.01,  # 1% risk per trade
                 min_confidence: float = 0.4,
                 dry_run: bool = True,
                 enabled_markets: Optional[List[str]] = None):
        
        self.balance = balance
        self.risk_per_trade = risk_per_trade
        self.min_confidence = min_confidence
        self.dry_run = dry_run
        
        # Multi-market configuration
        self.enabled_markets = enabled_markets or ["SPOT", "FUTURES"]
        self.market_balances = {market: balance / len(self.enabled_markets) 
                               for market in self.enabled_markets}
        
        # Initialize components
        self.risk_manager = RiskManager()
        self.binance_client = get_exchange_client("binance")
        
        # Trading state per market
        self.positions: Dict[str, Dict[str, Dict]] = {market: {} for market in self.enabled_markets}
        self.daily_trades: Dict[str, int] = {market: 0 for market in self.enabled_markets}
        self.last_signal_check = datetime.now()
        self.running = False
        
        # Load trading pairs for each market
        self.trading_pairs = {
            market: get_volume_weighted_pairs(market, 50) 
            for market in self.enabled_markets
        }
        
        logger.info(f"Multi-market trading bot initialized - Markets: {self.enabled_markets}, Dry run: {dry_run}")
    
    async def start(self):
        """Start the autonomous trading loop"""
        self.running = True
        logger.info("🚀 Starting autonomous trading bot...")
        
        try:
            while self.running:
                await self._trading_cycle()
                await asyncio.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Trading bot error: {e}")
        finally:
            logger.info("Trading bot stopped")
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        
    async def _trading_cycle(self):
        """Single trading cycle - check signals and execute trades"""
        try:
            # 1. Get fresh AI signals
            signals = await get_live_ai_signals(limit=10, profile="mixed")
            
            if not signals:
                logger.debug("No AI signals received")
                return
                
            logger.info(f"Received {len(signals)} AI signals")
            
            # 2. Process each signal
            for signal in signals:
                await self._process_signal(signal)
                
            # 3. Check existing positions for exit conditions
            await self._manage_positions()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _process_signal(self, signal: Dict[str, Any]):
        """Process a single AI signal across multiple markets and execute optimal trade"""
        try:
            symbol = signal['symbol']
            side = signal['side']  # 'buy' or 'sell'
            confidence = signal['confidence']
            
            # Skip low confidence signals
            if confidence < self.min_confidence:
                logger.debug(f"Signal {symbol} confidence {confidence} below threshold")
                return
            
            # Extract base token to check if it's Layer 1/2
            base_token = symbol.replace("USDC", "").replace("USDT", "").replace("BUSD", "")
            if not is_layer1_layer2_token(base_token):
                logger.debug(f"Token {base_token} not in Layer 1/2 list")
                return
            
            # Find optimal market for this signal
            optimal_market = get_optimal_market_for_token(base_token, "volume")
            if optimal_market not in self.enabled_markets:
                optimal_market = self.enabled_markets[0]  # Fallback to first enabled market
            
            # Check if we already have position in this symbol across any market
            has_position = any(symbol in positions for positions in self.positions.values())
            if has_position:
                logger.debug(f"Already have position in {symbol} across markets")
                return
            
            # Check daily trade limits for this market
            risk_config = get_risk_config(optimal_market)
            if self.daily_trades[optimal_market] >= risk_config['max_daily_trades']:
                logger.warning(f"Daily trade limit reached for {optimal_market}")
                return
            
            # Get current price for the optimal market pair
            market_symbol = self._get_market_symbol(symbol, optimal_market)
            current_price = await self._get_current_price(market_symbol, optimal_market)
            if not current_price:
                logger.warning(f"Could not get price for {market_symbol} on {optimal_market}")
                return
            
            # Calculate position size based on market-specific risk
            position_size = self._calculate_position_size(
                current_price, confidence, optimal_market
            )
            
            logger.info(f"📊 Trade calc for {market_symbol}: price=${current_price}, size={position_size}, notional=${position_size * current_price:.2f}")
            
            # Validate trade with risk manager
            stop_loss_price = self._calculate_stop_loss(current_price, side, optimal_market)
            
            is_valid, reason = self.risk_manager.validate_order(
                balance=self.market_balances[optimal_market],
                qty=position_size, 
                price=current_price,
                stop_loss=stop_loss_price
            )
            
            if not is_valid:
                logger.warning(f"Trade rejected by risk manager for {optimal_market}: {reason}")
                return
            
            # Execute trade in optimal market
            await self._execute_trade(
                market_symbol, side, position_size, current_price, 
                confidence, signal, optimal_market
            )
            
            # Increment daily trade count
            self.daily_trades[optimal_market] += 1
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _get_market_symbol(self, symbol: str, market_type: str) -> str:
        """Convert symbol to market-specific format"""
        base_token = symbol.replace("USDC", "").replace("USDT", "").replace("BUSD", "")
        
        if market_type == "FUTURES":
            return f"{base_token}USDC"  # USDC-M futures
        elif market_type in ["MARGIN", "CROSS_MARGIN"]:
            # Prefer USDC, fallback to USDT
            return f"{base_token}USDC" if "USDC" in MARKET_CONFIGS[market_type]["base_currencies"] else f"{base_token}USDT"
        else:  # SPOT
            return f"{base_token}USDC"
    
    def _calculate_position_size(self, price: float, confidence: float, market_type: str) -> float:
        """Calculate position size based on risk, confidence, and market type"""
        risk_config = get_risk_config(market_type)
        market_config = get_market_config(market_type)
        
        # Market balance for this specific market
        available_balance = self.market_balances[market_type]
        
        # Risk amount = balance * max_position_size * confidence_multiplier
        confidence_multiplier = min(confidence * 1.5, 1.0)  # Scale confidence
        max_position_percent = risk_config['max_position_size']
        risk_amount = available_balance * max_position_percent * confidence_multiplier
        
        # For leveraged markets, account for leverage
        leverage = market_config.get('leverage', 1)
        effective_risk = risk_amount / leverage if leverage > 1 else risk_amount
        
        # Position size = risk_amount / (price * stop_loss_percent)
        stop_loss_percent = risk_config['stop_loss']
        position_size = effective_risk / (price * stop_loss_percent)
        
        # Ensure minimum order size (Binance usually ~$10)
        min_notional = 10.0
        if position_size * price < min_notional:
            position_size = min_notional / price
            
        return round(position_size, 6)
    
    def _calculate_stop_loss(self, price: float, side: str, market_type: str) -> float:
        """Calculate stop loss price based on market type"""
        risk_config = get_risk_config(market_type)
        stop_loss_percent = risk_config['stop_loss']
        
        if side == 'buy':
            return price * (1 - stop_loss_percent)
        else:  # sell
            return price * (1 + stop_loss_percent)
    
    def _calculate_take_profit(self, price: float, side: str, market_type: str) -> float:
        """Calculate take profit price based on market type"""
        risk_config = get_risk_config(market_type)
        take_profit_percent = risk_config['take_profit']
        
        if side == 'buy':
            return price * (1 + take_profit_percent)
        else:  # sell
            return price * (1 - take_profit_percent)
    
    async def _get_current_price(self, symbol: str, market_type: str = "SPOT") -> Optional[float]:
        """Get current market price for symbol"""
        try:
            # Use Binance API to get current price
            from backend.routes.external_data import binance_ohlcv
            
            data = await binance_ohlcv(symbol, limit=1)
            candles = data.get('candles', [])
            
            if candles:
                return float(candles[-1]['close'])
                
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
        
        return None
    
    async def _execute_trade(self, symbol: str, side: str, qty: float, 
                           price: float, confidence: float, original_signal: Dict, 
                           market_type: str):
        """Execute the actual trade in specified market"""
        try:
            if self.dry_run:
                # Simulate trade execution
                logger.info(f"🔥 DRY RUN TRADE [{market_type}]: {side.upper()} {qty} {symbol} @ {price} (confidence: {confidence})")
                
                # Create mock order result
                order_result = {
                    'symbol': symbol,
                    'side': side.upper(),
                    'type': 'MARKET',
                    'quantity': str(qty),
                    'price': str(price),
                    'status': 'FILLED',
                    'orderId': f'DRYRUN_{int(datetime.now().timestamp())}'
                }
            else:
                # Real trade execution
                logger.info(f"🚀 LIVE TRADE [{market_type}]: {side.upper()} {qty} {symbol} @ {price} (confidence: {confidence})")
                
                order_result = self.binance_client.create_order(
                    symbol=symbol,
                    side=side.upper(),
                    qty=qty,
                    order_type='MARKET'
                )
            
            # Track position in specific market
            self.positions[market_type][symbol] = {
                'side': side,
                'qty': qty,
                'entry_price': price,
                'entry_time': datetime.now(),
                'confidence': confidence,
                'market_type': market_type,
                'stop_loss': self._calculate_stop_loss(price, side, market_type),
                'take_profit': self._calculate_take_profit(price, side, market_type),
                'order_id': order_result.get('orderId'),
                'original_signal': original_signal
            }
            
            # Save to database
            await self._save_trade_to_db(symbol, side, qty, price, confidence, original_signal, market_type)
            
            logger.info(f"✅ Trade executed successfully: {symbol} {side} {qty} @ {price}")
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
    
    async def _save_trade_to_db(self, symbol: str, side: str, qty: float, 
                              price: float, confidence: float, signal: Dict, market_type: str):
        """Save trade to database using direct SQL (avoiding ORM complexity)"""
        try:
            # Use direct database connection to avoid import issues
            import sqlite3
            import os
            
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trades.db")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Insert trade record
            cursor.execute("""
                INSERT INTO trades (symbol, side, entry_price, qty, timestamp, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                side,
                price,
                qty,
                datetime.now().isoformat(),
                "open",
                json.dumps({
                    'confidence': confidence,
                    'market_type': market_type,
                    'ai_signal': signal,
                    'bot_executed': True
                })
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Trade saved to database: {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save trade to database: {e}")
    
    async def _manage_positions(self):
        """Check existing positions for exit conditions across all markets"""
        positions_to_close = []
        
        for market_type in self.enabled_markets:
            for symbol, position in self.positions[market_type].items():
                try:
                    current_price = await self._get_current_price(symbol, market_type)
                    if not current_price:
                        continue
                    
                    # Check stop loss and take profit
                    if self._should_close_position(position, current_price):
                        positions_to_close.append((market_type, symbol, position, current_price))
                        
                except Exception as e:
                    logger.error(f"Error managing position {symbol} in {market_type}: {e}")
        
        # Close positions that meet exit criteria
        for market_type, symbol, position, current_price in positions_to_close:
            await self._close_position(market_type, symbol, position, current_price)
    
    def _should_close_position(self, position: Dict, current_price: float) -> bool:
        """Determine if position should be closed"""
        entry_price = position['entry_price']
        side = position['side']
        stop_loss = position['stop_loss']
        take_profit = position.get('take_profit')
        entry_time = position['entry_time']
        
        # Stop loss check
        if side == 'buy' and current_price <= stop_loss:
            logger.info(f"Stop loss triggered for BUY position: {current_price} <= {stop_loss}")
            return True
        elif side == 'sell' and current_price >= stop_loss:
            logger.info(f"Stop loss triggered for SELL position: {current_price} >= {stop_loss}")
            return True
        
        # Take profit check
        if take_profit:
            if side == 'buy' and current_price >= take_profit:
                logger.info(f"Take profit triggered for BUY position: {current_price} >= {take_profit}")
                return True
            elif side == 'sell' and current_price <= take_profit:
                logger.info(f"Take profit triggered for SELL position: {current_price} <= {take_profit}")
                return True
        
        # Time-based exit (hold for max 2 hours for leveraged markets, 4 hours for spot)
        market_type = position.get('market_type', 'SPOT')
        max_hold_hours = 2 if market_type in ['FUTURES', 'MARGIN', 'CROSS_MARGIN'] else 4
        
        if datetime.now() - entry_time > timedelta(hours=max_hold_hours):
            logger.info(f"Time-based exit triggered for {market_type} position after {max_hold_hours}h")
            return True
        
        return False
    
    async def _close_position(self, market_type: str, symbol: str, position: Dict, current_price: float):
        """Close an open position"""
        try:
            side = 'sell' if position['side'] == 'buy' else 'buy'  # Opposite side to close
            qty = position['qty']
            
            if self.dry_run:
                logger.info(f"🔥 DRY RUN CLOSE [{market_type}]: {side.upper()} {qty} {symbol} @ {current_price}")
            else:
                logger.info(f"🚀 LIVE CLOSE [{market_type}]: {side.upper()} {qty} {symbol} @ {current_price}")
                
                # Note: Real implementation would need to handle different market types
                # For futures/margin, different API endpoints would be used
                order_result = self.binance_client.create_order(
                    symbol=symbol,
                    side=side.upper(),
                    qty=qty,
                    order_type='MARKET'
                )
            
            # Calculate P&L (accounting for leverage in futures/margin)
            entry_price = position['entry_price']
            leverage = get_market_config(market_type).get('leverage', 1)
            
            if position['side'] == 'buy':
                pnl = (current_price - entry_price) * qty * leverage
            else:
                pnl = (entry_price - current_price) * qty * leverage
            
            logger.info(f"💰 Position closed [{market_type}] - P&L: ${pnl:.2f} (leverage: {leverage}x)")
            
            # Update market-specific balance
            self.market_balances[market_type] += pnl
            self.balance = sum(self.market_balances.values())
            
            # Remove from positions
            del self.positions[market_type][symbol]
            
            # Update database
            await self._update_trade_in_db(symbol, current_price, pnl)
            
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
    
    async def _update_trade_in_db(self, symbol: str, exit_price: float, pnl: float):
        """Update trade in database with exit details"""
        try:
            import sqlite3
            import os
            
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trades.db")
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Find and update the most recent open trade for this symbol
            cursor.execute(
                "UPDATE trades SET exit_price = ?, pnl = ?, status = 'closed' WHERE symbol = ? AND status = 'open'",
                (exit_price, pnl, symbol)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"💾 Trade updated in database: {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to update trade in database: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'running': self.running,
            'balance': self.balance,
            'positions': len(self.positions),
            'position_details': self.positions,
            'dry_run': self.dry_run,
            'last_check': self.last_signal_check.isoformat()
        }


# Global bot instance
trading_bot = None


def get_trading_bot() -> AutonomousTradingBot:
    """Get or create the global trading bot instance"""
    global trading_bot
    if trading_bot is None:
        trading_bot = AutonomousTradingBot(dry_run=True)  # Start in dry run mode
    return trading_bot


async def start_trading_bot(dry_run: bool = True):
    """Start the autonomous trading bot"""
    bot = get_trading_bot()
    bot.dry_run = dry_run
    await bot.start()


def stop_trading_bot():
    """Stop the autonomous trading bot"""
    global trading_bot
    if trading_bot:
        trading_bot.stop()