#!/usr/bin/env python3
"""
ATR Position Sizer - Risk-Based Position Sizing (No Hardcoded Sizes)

Calculates position size based on:
- Account equity
- Risk per trade (% of equity)
- ATR-based stop distance
- Dynamic leverage caps (regime-aware)
- Spread/slippage compensation

Formula:
    risk_usd = equity * risk_pct
    stop_distance_usd = atr * atr_multiplier
    raw_qty = risk_usd / stop_distance_usd
    final_qty = raw_qty / (1 + spread_pct + slippage_pct)  # Fee compensation

Leverage Caps (Dynamic):
- CHOP regime: 2-3x
- TREND regime: 4-6x
- Never exceed env MAX_LEVERAGE

Redis Keys:
- quantum:equity:current (HASH: equity)
- quantum:market_state:{symbol} (HASH: atr, regime, spread_bps)
"""

import os
import sys
import time
import json
import logging
from typing import Optional, Tuple, Dict
import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class ATRSizerConfig:
    """ATR sizer configuration"""
    
    def __init__(self):
        # Risk per trade (% of equity)
        self.RISK_PCT_BASE = float(os.getenv('SIZER_RISK_PCT_BASE', '0.005'))  # 0.5%
        self.RISK_PCT_TREND = float(os.getenv('SIZER_RISK_PCT_TREND', '0.007'))  # 0.7% in trends
        self.RISK_PCT_CHOP = float(os.getenv('SIZER_RISK_PCT_CHOP', '0.003'))  # 0.3% in chop
        
        # ATR stop multipliers
        self.ATR_MULT_STOP = float(os.getenv('SIZER_ATR_MULT_STOP', '1.8'))  # 1.8 * ATR for stop
        self.ATR_MULT_TP = float(os.getenv('SIZER_ATR_MULT_TP', '1.2'))  # 1.2 * ATR for first TP
        self.ATR_MULT_TRAIL = float(os.getenv('SIZER_ATR_MULT_TRAIL', '1.0'))  # 1.0 * ATR for trailing
        
        # Leverage caps
        self.MAX_LEVERAGE = float(os.getenv('SIZER_MAX_LEVERAGE', '6.0'))
        self.LEV_CHOP_MIN = float(os.getenv('SIZER_LEV_CHOP_MIN', '2.0'))
        self.LEV_CHOP_MAX = float(os.getenv('SIZER_LEV_CHOP_MAX', '3.0'))
        self.LEV_TREND_MIN = float(os.getenv('SIZER_LEV_TREND_MIN', '3.0'))
        self.LEV_TREND_MAX = float(os.getenv('SIZER_LEV_TREND_MAX', '6.0'))
        
        # Fee/slippage compensation
        self.FEE_PCT = float(os.getenv('SIZER_FEE_PCT', '0.0006'))  # 0.06% taker fee
        self.SLIPPAGE_BASE_PCT = float(os.getenv('SIZER_SLIPPAGE_BASE_PCT', '0.0005'))  # 0.05% base slippage
        
        # Min/max position sizes (safety)
        self.MIN_NOTIONAL_USD = float(os.getenv('SIZER_MIN_NOTIONAL_USD', '10'))
        self.MAX_NOTIONAL_USD = float(os.getenv('SIZER_MAX_NOTIONAL_USD', '5000'))
        
        # Redis
        self.REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        
        logger.info(f"ATRSizer Config:")
        logger.info(f"  Risk Base: {self.RISK_PCT_BASE*100:.2f}%")
        logger.info(f"  Risk Trend: {self.RISK_PCT_TREND*100:.2f}%")
        logger.info(f"  Risk Chop: {self.RISK_PCT_CHOP*100:.2f}%")
        logger.info(f"  ATR Stop Mult: {self.ATR_MULT_STOP}x")
        logger.info(f"  Max Leverage: {self.MAX_LEVERAGE}x")


class ATRPositionSizer:
    """
    ATR-based position sizer with dynamic risk and leverage
    """
    
    def __init__(self, config: ATRSizerConfig):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        logger.info("ATRPositionSizer initialized")
    
    def _get_equity(self) -> float:
        """Get current account equity"""
        equity_key = "quantum:equity:current"
        data = self.redis.hgetall(equity_key)
        
        if not data:
            logger.error("No equity data - cannot size position (fail-closed)")
            return 0.0
        
        equity = float(data.get('equity', '0'))
        age = time.time() - float(data.get('last_update_ts', '0'))
        
        if age > 300:
            logger.warning(f"Equity data stale (age={age:.0f}s)")
        
        return equity
    
    def _get_market_state(self, symbol: str) -> Dict:
        """Get market state (ATR, regime, spread) for symbol"""
        market_key = f"quantum:market_state:{symbol}"
        data = self.redis.hgetall(market_key)
        
        if not data:
            logger.warning(f"{symbol}: No market state data (fail-closed)")
            return {}
        
        age = time.time() - float(data.get('last_update_ts', '0'))
        if age > 120:
            logger.warning(f"{symbol}: Market state stale (age={age:.0f}s)")
        
        return {
            'atr': float(data.get('atr', '0')),
            'regime': data.get('regime', 'BASE'),
            'spread_bps': float(data.get('spread_bps', '0')),
            'price': float(data.get('price', '0')),
            'atr_pct': float(data.get('atr_pct', '0'))
        }
    
    def _get_risk_pct(self, regime: str) -> float:
        """Get risk % based on regime"""
        if regime == 'TREND_STRONG':
            return self.config.RISK_PCT_TREND
        elif regime == 'CHOP':
            return self.config.RISK_PCT_CHOP
        else:
            return self.config.RISK_PCT_BASE
    
    def _get_leverage_cap(self, regime: str, confidence: float) -> float:
        """Get dynamic leverage cap based on regime and confidence"""
        if regime == 'CHOP':
            # Low leverage in choppy markets
            base = self.config.LEV_CHOP_MIN
            max_lev = self.config.LEV_CHOP_MAX
        elif regime == 'TREND_STRONG':
            # Higher leverage in trending markets
            base = self.config.LEV_TREND_MIN
            max_lev = self.config.LEV_TREND_MAX
        else:
            # Base regime
            base = (self.config.LEV_CHOP_MAX + self.config.LEV_TREND_MIN) / 2
            max_lev = self.config.LEV_TREND_MIN
        
        # Scale with confidence (0.5-1.0 → base to max)
        conf_factor = max(0.0, min(1.0, (confidence - 0.5) / 0.5))
        leverage = base + (max_lev - base) * conf_factor
        
        # Never exceed global max
        return min(leverage, self.config.MAX_LEVERAGE)
    
    def calculate_position(
        self,
        symbol: str,
        side: str,
        confidence: float,
        atr_override: Optional[float] = None,
        price_override: Optional[float] = None
    ) -> Dict:
        """
        Calculate position size with ATR-based stop
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            confidence: Signal confidence [0-1]
            atr_override: Override ATR value (for testing)
            price_override: Override price (for testing)
        
        Returns:
            {
                'qty': float,
                'notional_usd': float,
                'leverage': float,
                'stop_loss': float,
                'take_profit': float,
                'stop_distance_usd': float,
                'risk_usd': float,
                'risk_pct': float,
                'reasoning': str
            }
        """
        
        # Get equity
        equity = self._get_equity()
        if equity <= 0:
            logger.error(f"{symbol}: No equity - cannot size (fail-closed)")
            return {'qty': 0.0, 'reasoning': 'no_equity'}
        
        # Get market state
        market = self._get_market_state(symbol)
        if not market or market.get('atr', 0) <= 0:
            logger.error(f"{symbol}: No ATR data - cannot size (fail-closed)")
            return {'qty': 0.0, 'reasoning': 'no_atr'}
        
        atr = atr_override if atr_override else market['atr']
        price = price_override if price_override else market['price']
        regime = market['regime']
        spread_bps = market['spread_bps']
        
        if price <= 0:
            logger.error(f"{symbol}: No price - cannot size (fail-closed)")
            return {'qty': 0.0, 'reasoning': 'no_price'}
        
        # Calculate risk
        risk_pct = self._get_risk_pct(regime)
        risk_usd = equity * risk_pct
        
        # Calculate stop distance (ATR-based)
        stop_distance_usd = atr * self.config.ATR_MULT_STOP
        
        # Calculate raw quantity
        if stop_distance_usd <= 0:
            logger.error(f"{symbol}: Stop distance zero - cannot size")
            return {'qty': 0.0, 'reasoning': 'zero_stop_distance'}
        
        raw_qty = risk_usd / stop_distance_usd
        
        # Compensate for fees + slippage
        spread_pct = spread_bps / 10000
        slippage_pct = self.config.SLIPPAGE_BASE_PCT + (spread_pct * 0.5)  # Slippage scales with spread
        total_cost_pct = self.config.FEE_PCT + slippage_pct
        
        final_qty = raw_qty / (1 + total_cost_pct)
        
        # Calculate notional
        notional_usd = final_qty * price
        
        # Apply min/max notional limits
        if notional_usd < self.config.MIN_NOTIONAL_USD:
            logger.warning(f"{symbol}: Notional ${notional_usd:.2f} < min ${self.config.MIN_NOTIONAL_USD}")
            return {'qty': 0.0, 'reasoning': 'below_min_notional'}
        
        if notional_usd > self.config.MAX_NOTIONAL_USD:
            # Scale down to max
            scale = self.config.MAX_NOTIONAL_USD / notional_usd
            final_qty *= scale
            notional_usd = final_qty * price
            logger.warning(f"{symbol}: Scaled down to max notional ${self.config.MAX_NOTIONAL_USD}")
        
        # Calculate leverage cap
        leverage_cap = self._get_leverage_cap(regime, confidence)
        
        # Calculate actual leverage (notional / equity)
        actual_leverage = notional_usd / equity
        
        if actual_leverage > leverage_cap:
            # Scale down to leverage cap
            scale = leverage_cap / actual_leverage
            final_qty *= scale
            notional_usd = final_qty * price
            actual_leverage = leverage_cap
            logger.info(f"{symbol}: Scaled down to leverage cap {leverage_cap:.1f}x")
        
        # Calculate stop loss and take profit prices
        if side.upper() in ['BUY', 'LONG']:
            stop_loss = price - stop_distance_usd
            take_profit = price + (atr * self.config.ATR_MULT_TP)
        else:  # SELL/SHORT
            stop_loss = price + stop_distance_usd
            take_profit = price - (atr * self.config.ATR_MULT_TP)
        
        reasoning = (
            f"equity=${equity:.0f} risk={risk_pct*100:.2f}% "
            f"atr={atr:.4f} stop_dist={stop_distance_usd:.2f} "
            f"regime={regime} lev={actual_leverage:.1f}x "
            f"fees={total_cost_pct*100:.2f}%"
        )
        
        logger.info(f"SIZER_ATR {symbol}: {reasoning} → qty={final_qty:.4f} notional=${notional_usd:.2f}")
        
        return {
            'qty': final_qty,
            'notional_usd': notional_usd,
            'leverage': actual_leverage,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_distance_usd': stop_distance_usd,
            'risk_usd': risk_usd,
            'risk_pct': risk_pct,
            'atr': atr,
            'price': price,
            'regime': regime,
            'confidence': confidence,
            'reasoning': reasoning
        }


def main():
    """Test ATR sizer"""
    config = ATRSizerConfig()
    sizer = ATRPositionSizer(config)
    
    # Test calculation
    result = sizer.calculate_position(
        symbol='BTCUSDT',
        side='BUY',
        confidence=0.7,
        atr_override=100.0,  # $100 ATR
        price_override=50000.0  # $50k BTC
    )
    
    if result['qty'] > 0:
        logger.info(f"Position: {result['qty']:.4f} BTC @ ${result['price']:.2f}")
        logger.info(f"  Notional: ${result['notional_usd']:.2f}")
        logger.info(f"  Leverage: {result['leverage']:.2f}x")
        logger.info(f"  Stop Loss: ${result['stop_loss']:.2f}")
        logger.info(f"  Take Profit: ${result['take_profit']:.2f}")
    else:
        logger.warning(f"Cannot size: {result['reasoning']}")


if __name__ == '__main__':
    main()
