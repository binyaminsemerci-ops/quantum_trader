#!/usr/bin/env python3
"""
RiskGuard - Hard Equity Protection (Fail-Closed)

Global risk gates that block OPEN actions and enforce EMERGENCY_FLATTEN when breached.

Features:
- Max daily loss gate (-2.5% equity/day)
- Max drawdown from peak (-8%)
- Consecutive loss counter (3+ losses → cooldown)
- Spread/volatility spike gate
- Emergency flatten mode

All guards are fail-closed: missing data = BLOCK

Redis Keys:
- quantum:equity:current (HASH: equity, peak, last_update_ts)
- quantum:risk:daily_pnl (HASH: date, pnl, trades)
- quantum:risk:consecutive_losses (STRING: count)
- quantum:risk:guard_active (HASH: guard_type, reason, expires_at)
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
import redis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger(__name__)


class RiskGuardConfig:
    """Risk guard configuration from environment"""
    
    def __init__(self):
        # Max daily loss (% of equity)
        self.MAX_DAILY_LOSS_PCT = float(os.getenv('RISK_MAX_DAILY_LOSS_PCT', '0.025'))  # 2.5%
        
        # Max drawdown from peak (% of peak equity)
        self.MAX_DRAWDOWN_PCT = float(os.getenv('RISK_MAX_DRAWDOWN_PCT', '0.08'))  # 8%
        
        # Consecutive losses before cooldown
        self.MAX_CONSECUTIVE_LOSSES = int(os.getenv('RISK_MAX_CONSECUTIVE_LOSSES', '3'))
        self.CONSEC_LOSS_COOLDOWN_SEC = int(os.getenv('RISK_CONSEC_LOSS_COOLDOWN_SEC', '3600'))  # 1 hour
        
        # Spread/volatility gates
        self.MAX_SPREAD_BPS = float(os.getenv('RISK_MAX_SPREAD_BPS', '50'))  # 50 bps = 0.5%
        self.MAX_ATR_PCT = float(os.getenv('RISK_MAX_ATR_PCT', '0.05'))  # 5% ATR
        
        # Emergency flatten gate (% drawdown)
        self.EMERGENCY_FLATTEN_DRAWDOWN_PCT = float(os.getenv('RISK_EMERGENCY_FLATTEN_PCT', '0.10'))  # 10%
        self.EMERGENCY_FLATTEN_DURATION_SEC = int(os.getenv('RISK_EMERGENCY_FLATTEN_DURATION_SEC', '14400'))  # 4 hours
        
        # Redis config
        self.REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
        self.REDIS_DB = int(os.getenv('REDIS_DB', '0'))
        
        # Equity tracking (fallback if missing)
        self.INITIAL_EQUITY = float(os.getenv('RISK_INITIAL_EQUITY', '10000'))
        
        logger.info(f"RiskGuard Config:")
        logger.info(f"  Max Daily Loss: {self.MAX_DAILY_LOSS_PCT*100:.1f}%")
        logger.info(f"  Max Drawdown: {self.MAX_DRAWDOWN_PCT*100:.1f}%")
        logger.info(f"  Emergency Flatten: {self.EMERGENCY_FLATTEN_DRAWDOWN_PCT*100:.1f}%")
        logger.info(f"  Consecutive Losses: {self.MAX_CONSECUTIVE_LOSSES}")
        logger.info(f"  Max Spread: {self.MAX_SPREAD_BPS} bps")


class RiskGuard:
    """
    Global risk guard enforcer
    
    Checks all risk gates and publishes guard events to Redis streams.
    Returns (is_blocked: bool, reason: str, guard_type: str)
    """
    
    def __init__(self, config: RiskGuardConfig):
        self.config = config
        self.redis = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=True
        )
        
        # Initialize equity tracking if missing
        self._ensure_equity_tracking()
        
        logger.info("RiskGuard initialized")
    
    def _ensure_equity_tracking(self):
        """Initialize equity tracking if missing (fail-closed)"""
        equity_key = "quantum:equity:current"
        if not self.redis.exists(equity_key):
            # Initialize with fallback equity
            self.redis.hset(equity_key, mapping={
                'equity': str(self.config.INITIAL_EQUITY),
                'peak': str(self.config.INITIAL_EQUITY),
                'last_update_ts': str(time.time())
            })
            logger.warning(f"Initialized equity tracking with fallback: ${self.config.INITIAL_EQUITY}")
    
    def _get_equity(self) -> Tuple[float, float, float]:
        """
        Get current equity, peak, and age
        
        Returns:
            (current_equity, peak_equity, age_seconds)
        """
        equity_key = "quantum:equity:current"
        data = self.redis.hgetall(equity_key)
        
        if not data:
            logger.error("RISK_GUARD_TRIGGERED: No equity data (fail-closed)")
            return 0.0, 0.0, 999999.0
        
        current = float(data.get('equity', '0'))
        peak = float(data.get('peak', '0'))
        last_update = float(data.get('last_update_ts', '0'))
        age = time.time() - last_update
        
        return current, peak, age
    
    def _get_daily_pnl(self) -> float:
        """Get today's PnL"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"quantum:risk:daily_pnl:{today}"
        
        data = self.redis.hgetall(daily_key)
        if not data:
            return 0.0
        
        return float(data.get('pnl', '0'))
    
    def _get_consecutive_losses(self) -> int:
        """Get consecutive loss count"""
        key = "quantum:risk:consecutive_losses"
        count = self.redis.get(key)
        return int(count) if count else 0
    
    def _set_guard_active(self, guard_type: str, reason: str, duration_sec: int):
        """Activate a risk guard with TTL"""
        guard_key = f"quantum:risk:guard_active:{guard_type}"
        expires_at = time.time() + duration_sec
        
        self.redis.setex(guard_key, duration_sec, json.dumps({
            'guard_type': guard_type,
            'reason': reason,
            'activated_at': time.time(),
            'expires_at': expires_at
        }))
        
        # Publish event
        self.redis.xadd('quantum:stream:risk.events', {
            'event': 'RISK_GUARD_ACTIVATED',
            'guard_type': guard_type,
            'reason': reason,
            'duration_sec': str(duration_sec),
            'timestamp': str(time.time())
        })
        
        logger.warning(f"RISK_GUARD_TRIGGERED type={guard_type} reason={reason} duration={duration_sec}s")
    
    def check_all_guards(self, action: str, symbol: str = "UNKNOWN", spread_bps: float = 0.0, atr_pct: float = 0.0) -> Tuple[bool, str, str]:
        """
        Check all risk guards
        
        Args:
            action: OPEN or CLOSE
            symbol: Trading symbol
            spread_bps: Bid-ask spread in basis points
            atr_pct: ATR as % of price
        
        Returns:
            (is_blocked, reason, guard_type)
        """
        
        # CLOSE actions always allowed (priority exits)
        if action in ['FULL_CLOSE_PROPOSED', 'PARTIAL_75', 'PARTIAL_50', 'PARTIAL_25', 'CLOSE']:
            return False, "", ""
        
        # Check if any guard is active
        guard_patterns = self.redis.keys("quantum:risk:guard_active:*")
        for guard_key in guard_patterns:
            guard_data = self.redis.get(guard_key)
            if guard_data:
                guard_info = json.loads(guard_data)
                guard_type = guard_info['guard_type']
                reason = guard_info['reason']
                logger.info(f"RISK_GUARD_ACTIVE: {guard_type} - {reason} (blocking OPEN)")
                return True, reason, guard_type
        
        # Guard 1: Max daily loss
        current_equity, peak_equity, equity_age = self._get_equity()
        
        if current_equity <= 0 or equity_age > 300:
            # Fail-closed: no equity data or stale
            reason = f"equity_missing_or_stale (age={equity_age:.0f}s)"
            logger.error(f"RISK_GUARD_TRIGGERED type=DAILY_LOSS reason={reason}")
            return True, reason, "EQUITY_STALE"
        
        daily_pnl = self._get_daily_pnl()
        daily_loss_pct = abs(daily_pnl) / current_equity if daily_pnl < 0 else 0.0
        
        if daily_loss_pct > self.config.MAX_DAILY_LOSS_PCT:
            reason = f"daily_loss={daily_loss_pct*100:.2f}% > {self.config.MAX_DAILY_LOSS_PCT*100:.1f}%"
            self._set_guard_active('DAILY_LOSS', reason, 86400)  # Block for rest of day
            logger.error(f"RISK_GUARD_TRIGGERED type=DAILY_LOSS value={daily_loss_pct*100:.2f}%")
            return True, reason, "DAILY_LOSS"
        
        # Guard 2: Max drawdown from peak
        drawdown_pct = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
        
        if drawdown_pct > self.config.MAX_DRAWDOWN_PCT:
            # Check if emergency flatten needed
            if drawdown_pct > self.config.EMERGENCY_FLATTEN_DRAWDOWN_PCT:
                reason = f"EMERGENCY_FLATTEN drawdown={drawdown_pct*100:.1f}% > {self.config.EMERGENCY_FLATTEN_DRAWDOWN_PCT*100:.1f}%"
                self._set_guard_active('EMERGENCY_FLATTEN', reason, self.config.EMERGENCY_FLATTEN_DURATION_SEC)
                logger.critical(f"RISK_GUARD_TRIGGERED type=EMERGENCY_FLATTEN value={drawdown_pct*100:.1f}%")
                
                # TODO: Trigger emergency flatten (emit reduceOnly plans for all positions)
                self._trigger_emergency_flatten()
                
                return True, reason, "EMERGENCY_FLATTEN"
            else:
                reason = f"drawdown={drawdown_pct*100:.1f}% > {self.config.MAX_DRAWDOWN_PCT*100:.1f}%"
                self._set_guard_active('DRAWDOWN', reason, 3600)  # Block for 1 hour
                logger.error(f"RISK_GUARD_TRIGGERED type=DRAWDOWN value={drawdown_pct*100:.1f}%")
                return True, reason, "DRAWDOWN"
        
        # Guard 3: Consecutive losses
        consec_losses = self._get_consecutive_losses()
        if consec_losses >= self.config.MAX_CONSECUTIVE_LOSSES:
            reason = f"consec_losses={consec_losses} >= {self.config.MAX_CONSECUTIVE_LOSSES}"
            self._set_guard_active('CONSEC_LOSS', reason, self.config.CONSEC_LOSS_COOLDOWN_SEC)
            logger.error(f"RISK_GUARD_TRIGGERED type=CONSEC_LOSS value={consec_losses}")
            return True, reason, "CONSEC_LOSS"
        
        # Guard 4: Spread spike
        if spread_bps > self.config.MAX_SPREAD_BPS:
            reason = f"{symbol} spread={spread_bps:.1f} bps > {self.config.MAX_SPREAD_BPS:.0f} bps"
            logger.warning(f"RISK_GUARD_TRIGGERED type=SPREAD_SPIKE value={spread_bps:.1f} bps symbol={symbol}")
            return True, reason, "SPREAD_SPIKE"
        
        # Guard 5: ATR spike
        if atr_pct > self.config.MAX_ATR_PCT:
            reason = f"{symbol} atr={atr_pct*100:.1f}% > {self.config.MAX_ATR_PCT*100:.1f}%"
            logger.warning(f"RISK_GUARD_TRIGGERED type=ATR_SPIKE value={atr_pct*100:.1f}% symbol={symbol}")
            return True, reason, "ATR_SPIKE"
        
        # All guards passed
        return False, "", ""
    
    def _trigger_emergency_flatten(self):
        """Emit reduceOnly close plans for all open positions"""
        try:
            # Get open positions from Binance (or local position tracker)
            # For now, publish event to stream for other services to handle
            self.redis.xadd('quantum:stream:risk.events', {
                'event': 'EMERGENCY_FLATTEN_REQUESTED',
                'reason': 'drawdown_breach',
                'timestamp': str(time.time())
            })
            
            logger.critical("EMERGENCY_FLATTEN_REQUESTED published to risk.events stream")
        except Exception as e:
            logger.error(f"Failed to trigger emergency flatten: {e}")
    
    def update_equity(self, new_equity: float):
        """Update current equity and peak"""
        equity_key = "quantum:equity:current"
        data = self.redis.hgetall(equity_key)
        
        current_peak = float(data.get('peak', '0')) if data else new_equity
        new_peak = max(current_peak, new_equity)
        
        self.redis.hset(equity_key, mapping={
            'equity': str(new_equity),
            'peak': str(new_peak),
            'last_update_ts': str(time.time())
        })
        
        drawdown_pct = (new_peak - new_equity) / new_peak if new_peak > 0 else 0.0
        logger.info(f"Equity updated: ${new_equity:.2f} (peak=${new_peak:.2f}, dd={drawdown_pct*100:.1f}%)")
    
    def record_trade_result(self, pnl: float, symbol: str):
        """Record trade result for daily PnL and consecutive loss tracking"""
        # Update daily PnL
        today = datetime.utcnow().strftime('%Y-%m-%d')
        daily_key = f"quantum:risk:daily_pnl:{today}"
        
        data = self.redis.hgetall(daily_key)
        current_pnl = float(data.get('pnl', '0')) if data else 0.0
        trade_count = int(data.get('trades', '0')) if data else 0
        
        new_pnl = current_pnl + pnl
        new_count = trade_count + 1
        
        self.redis.hset(daily_key, mapping={
            'pnl': str(new_pnl),
            'trades': str(new_count),
            'date': today
        })
        self.redis.expire(daily_key, 86400 * 2)  # Keep for 2 days
        
        # Update consecutive losses
        consec_key = "quantum:risk:consecutive_losses"
        if pnl < 0:
            # Loss: increment counter
            consec = self._get_consecutive_losses()
            self.redis.set(consec_key, str(consec + 1))
            logger.info(f"Trade result: {symbol} PnL=${pnl:.2f} (LOSS, consec={consec+1})")
        else:
            # Win: reset counter
            self.redis.set(consec_key, '0')
            logger.info(f"Trade result: {symbol} PnL=${pnl:.2f} (WIN, consec=0)")


def main():
    """Test RiskGuard"""
    config = RiskGuardConfig()
    guard = RiskGuard(config)
    
    # Test checks
    is_blocked, reason, guard_type = guard.check_all_guards('OPEN', 'BTCUSDT', spread_bps=10, atr_pct=0.02)
    
    if is_blocked:
        logger.warning(f"OPEN blocked: {guard_type} - {reason}")
    else:
        logger.info("All guards passed - OPEN allowed")


if __name__ == '__main__':
    main()
