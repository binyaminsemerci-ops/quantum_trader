#!/usr/bin/env python3
"""
GovernerAgent - Risk Management & Capital Allocation Layer
----------------------------------------------------------
Final decision maker that converts ensemble predictions into safe position sizes.

Responsibilities:
- Position sizing (Kelly Criterion / Fixed Fractional)
- Risk management (max drawdown, exposure limits)
- Capital allocation across symbols
- Trade frequency control (cooldown periods)
- Emergency circuit breakers

Input: Meta agent predictions (action, confidence)
Output: Trade allocation (symbol, size, risk_amount, approved)
"""
import os
import json
import time
import redis
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

# ---------- LOGGER ----------
def safe_log(msg):
    """Simple logger for governer agent"""
    print(msg, flush=True)
    try:
        logfile = Path("/var/log/quantum/governer-agent.log")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a", encoding="utf-8") as f:
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{ts} | {msg}\n")
    except Exception:
        pass

# ---------- DATA CLASSES ----------
@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size_pct: float = 0.10  # Max 10% of balance per position
    max_total_exposure_pct: float = 0.50  # Max 50% total exposure
    max_drawdown_pct: float = 0.15  # Circuit breaker at 15% drawdown
    min_confidence_threshold: float = 0.65  # Only trade if confidence > 65%
    kelly_fraction: float = 0.25  # Use 25% of full Kelly (safer)
    cooldown_after_loss_minutes: int = 60  # Wait 1 hour after loss
    max_daily_trades: int = 200  # Max trades per day (TESTNET: raised for testing)
    emergency_stop: bool = False  # Manual kill switch

@dataclass
class PositionAllocation:
    """Position allocation decision"""
    symbol: str
    action: str  # BUY, SELL, HOLD
    approved: bool
    position_size_usd: float
    position_size_pct: float
    risk_amount_usd: float
    confidence: float
    reason: str  # Why approved/rejected
    kelly_optimal: float = 0.0
    timestamp: str = ""

# ---------- GOVERNER AGENT ----------
class GovernerAgent:
    """
    Risk management and capital allocation agent.
    
    Converts ensemble predictions into safe, profitable position sizes.
    Implements multiple risk controls and circuit breakers.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None, state_file: Optional[str] = None):
        self.name = "Governer-Agent"
        self.config = config or RiskConfig()
        self.redis = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        self.is_testnet = any(
            val.lower() in {"true", "1", "yes", "y", "on"}
            for val in [
                os.getenv("USE_BINANCE_TESTNET", ""),
                os.getenv("BINANCE_TESTNET", ""),
                os.getenv("TRADING_MODE", "")
            ]
            if val
        )
        self.daily_limit = int(os.getenv("GOVERNOR_MAX_DAILY_TRADES", self.config.max_daily_trades))
        if self.is_testnet:
            self.daily_limit = max(self.daily_limit, 1_000_000)
        self._last_daily_log_ts = 0.0
        
        # State file for tracking trades, losses, cooldowns
        self.state_file = Path(state_file or "/app/data/governer_state.json")
        self.state = self._load_state()
        
        # Portfolio tracking
        self.initial_balance = self.state.get("initial_balance", 10000.0)
        self.current_balance = self.state.get("current_balance", self.initial_balance)
        self.peak_balance = self.state.get("peak_balance", self.initial_balance)
        
        # Trade history (for cooldowns and limits)
        self.trade_history = self.state.get("trade_history", [])
        self.daily_trade_count = 0
        self.last_trade_date = self.state.get("last_trade_date", "")
        
        # Active positions (symbol → size)
        self.active_positions = self.state.get("active_positions", {})
        
        safe_log(f"[{self.name}] Initialized with balance=${self.current_balance:.2f}, "
                f"risk_config={asdict(self.config)}")
    
    def _load_state(self) -> dict:
        """Load persisted state from disk"""
        try:
            if self.state_file.exists():
                with open(self.state_file, "r") as f:
                    return json.load(f)
        except Exception as e:
            safe_log(f"[{self.name}] State load error: {e}")
        return {}
    
    def _save_state(self):
        """Persist state to disk"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "initial_balance": self.initial_balance,
                "current_balance": self.current_balance,
                "peak_balance": self.peak_balance,
                "trade_history": self.trade_history[-100:],  # Keep last 100 trades
                "last_trade_date": self.last_trade_date,
                "active_positions": self.active_positions,
                "last_updated": datetime.utcnow().isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            safe_log(f"[{self.name}] State save error: {e}")
    
    def _daily_key(self) -> str:
        return f"quantum:governor:daily_trades:{datetime.now().strftime('%Y%m%d')}"

    def _ttl_to_next_midnight(self) -> int:
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).date()
        expire_at = datetime.combine(tomorrow, datetime.min.time()) + timedelta(hours=1)
        return max(3600, int((expire_at - now).total_seconds()))

    def _get_daily_count(self) -> int:
        try:
            val = self.redis.get(self._daily_key())
            count = int(val) if val is not None else 0
            self.daily_trade_count = count
            return count
        except Exception as e:
            safe_log(f"[{self.name}] Redis daily count read failed: {e}")
            return self.daily_trade_count

    def _increment_daily_count(self) -> int:
        try:
            key = self._daily_key()
            pipe = self.redis.pipeline()
            pipe.incr(key)
            ttl = self.redis.ttl(key)
            if ttl is None or ttl < 0:
                pipe.expire(key, self._ttl_to_next_midnight())
            result = pipe.execute()
            count = int(result[0]) if result else self._get_daily_count()
            self.daily_trade_count = count
            return count
        except Exception as e:
            safe_log(f"[{self.name}] Redis daily count increment failed: {e}")
            return self.daily_trade_count

    def _log_daily_count(self):
        now = time.time()
        if now - self._last_daily_log_ts >= 60:
            count = self._get_daily_count()
            safe_log(
                f"[{self.name}] DAILY_COUNT key={self._daily_key()} count={count} limit={self.daily_limit} mode={'TESTNET' if self.is_testnet else 'LIVE'}"
            )
            self._last_daily_log_ts = now
    
    def _check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Check all circuit breakers and emergency stops.
        
        Returns: (is_safe, reason)
        """
        # Emergency stop
        if self.config.emergency_stop:
            return False, "EMERGENCY_STOP_ACTIVE"
        
        # Max drawdown check
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if current_drawdown > self.config.max_drawdown_pct:
            return False, f"MAX_DRAWDOWN_EXCEEDED ({current_drawdown:.1%} > {self.config.max_drawdown_pct:.1%})"
        
        # Daily trade limit (Redis-backed)
        self._log_daily_count()
        current_count = self._get_daily_count()
        if current_count >= self.daily_limit:
            return False, f"DAILY_TRADE_LIMIT_REACHED ({current_count}/{self.daily_limit})"
        
        # Total exposure check
        total_exposure = sum(self.active_positions.values())
        if total_exposure > self.current_balance * self.config.max_total_exposure_pct:
            return False, f"MAX_EXPOSURE_EXCEEDED ({total_exposure:.0f} > {self.current_balance * self.config.max_total_exposure_pct:.0f})"
        
        return True, "OK"
    
    def _check_cooldown(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if symbol is in cooldown period after recent loss.
        
        Returns: (can_trade, reason)
        """
        cooldown_minutes = self.config.cooldown_after_loss_minutes
        cutoff_time = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
        
        # Check recent losses for this symbol
        recent_losses = [
            t for t in self.trade_history[-50:]
            if t.get("symbol") == symbol 
            and t.get("pnl", 0) < 0
            and datetime.fromisoformat(t.get("timestamp", "2000-01-01")) > cutoff_time
        ]
        
        if recent_losses:
            last_loss = recent_losses[-1]
            time_since = datetime.utcnow() - datetime.fromisoformat(last_loss["timestamp"])
            remaining = cooldown_minutes - (time_since.total_seconds() / 60)
            return False, f"COOLDOWN_ACTIVE (loss {int(remaining)}min ago)"
        
        return True, "OK"
    
    def _calculate_kelly_position(
        self, 
        confidence: float, 
        win_rate: float = 0.55,
        avg_win: float = 1.5,
        avg_loss: float = 1.0
    ) -> float:
        """
        Calculate Kelly Criterion position size.
        
        Kelly % = (p * b - q) / b
        Where:
            p = win probability
            b = win/loss ratio
            q = 1 - p (loss probability)
        
        Args:
            confidence: Model confidence (0-1)
            win_rate: Historical win rate (default 55%)
            avg_win: Average win size relative to risk
            avg_loss: Average loss size relative to risk
        
        Returns:
            Optimal position size as fraction of balance (0-1)
        """
        # Use confidence as proxy for win probability
        p = confidence
        q = 1 - p
        b = avg_win / avg_loss
        
        # Kelly formula
        kelly_pct = (p * b - q) / b
        
        # Apply safety fraction (use only 25% of full Kelly by default)
        kelly_safe = kelly_pct * self.config.kelly_fraction
        
        # Clamp to reasonable range (0% - max_position_size)
        kelly_safe = max(0.0, min(kelly_safe, self.config.max_position_size_pct))
        
        return kelly_safe
    
    def allocate_position(
        self,
        symbol: str,
        action: str,
        confidence: float,
        balance: Optional[float] = None,
        meta_override: bool = False
    ) -> PositionAllocation:
        """
        Main decision function: Approve or reject trade with position sizing.
        
        Args:
            symbol: Trading pair
            action: BUY, SELL, or HOLD from meta agent
            confidence: Meta agent confidence (0-1)
            balance: Current account balance (uses tracked balance if None)
            meta_override: Whether meta agent overrode ensemble
        
        Returns:
            PositionAllocation with approval decision and sizing
        """
        timestamp = datetime.utcnow().isoformat()
        
        # Update balance if provided
        if balance is not None:
            self.current_balance = balance
            self.peak_balance = max(self.peak_balance, balance)
        
        # Default: HOLD (no trade)
        if action == "HOLD":
            return PositionAllocation(
                symbol=symbol,
                action="HOLD",
                approved=False,
                position_size_usd=0.0,
                position_size_pct=0.0,
                risk_amount_usd=0.0,
                confidence=confidence,
                reason="HOLD_SIGNAL",
                timestamp=timestamp
            )
        
        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            safe_log(f"[{self.name}] {symbol} REJECTED: Low confidence {confidence:.3f} < {self.config.min_confidence_threshold:.3f}")
            return PositionAllocation(
                symbol=symbol,
                action=action,
                approved=False,
                position_size_usd=0.0,
                position_size_pct=0.0,
                risk_amount_usd=0.0,
                confidence=confidence,
                reason=f"LOW_CONFIDENCE ({confidence:.3f} < {self.config.min_confidence_threshold:.3f})",
                timestamp=timestamp
            )
        
        # Check circuit breakers
        is_safe, reason = self._check_circuit_breakers()
        if not is_safe:
            safe_log(f"[{self.name}] {symbol} REJECTED: Circuit breaker - {reason}")
            return PositionAllocation(
                symbol=symbol,
                action=action,
                approved=False,
                position_size_usd=0.0,
                position_size_pct=0.0,
                risk_amount_usd=0.0,
                confidence=confidence,
                reason=reason,
                timestamp=timestamp
            )
        
        # Check cooldown
        can_trade, cooldown_reason = self._check_cooldown(symbol)
        if not can_trade:
            safe_log(f"[{self.name}] {symbol} REJECTED: {cooldown_reason}")
            return PositionAllocation(
                symbol=symbol,
                action=action,
                approved=False,
                position_size_usd=0.0,
                position_size_pct=0.0,
                risk_amount_usd=0.0,
                confidence=confidence,
                reason=cooldown_reason,
                timestamp=timestamp
            )
        
        # Calculate Kelly-optimal position size
        kelly_pct = self._calculate_kelly_position(confidence)
        position_size_usd = self.current_balance * kelly_pct
        risk_amount_usd = position_size_usd * 0.02  # Risk 2% per trade (stop loss)
        
        # Log and return approval
        safe_log(
            f"[{self.name}] {symbol} APPROVED: {action} | "
            f"Size=${position_size_usd:.2f} ({kelly_pct:.1%}) | "
            f"Risk=${risk_amount_usd:.2f} | "
            f"Conf={confidence:.3f} | "
            f"Meta={meta_override}"
        )

        # P0.6: Daily trades committed after trade.intent publish (router). Do not increment here.
        # self._increment_daily_count()  # DISABLED: Moved to router publish path

        return PositionAllocation(
            symbol=symbol,
            action=action,
            approved=True,
            position_size_usd=position_size_usd,
            position_size_pct=kelly_pct,
            risk_amount_usd=risk_amount_usd,
            confidence=confidence,
            reason="APPROVED",
            kelly_optimal=kelly_pct,
            timestamp=timestamp
        )
    
    def record_trade_result(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: float,
        position_size: float,
        pnl: float
    ):
        """
        Record trade outcome for learning and risk adjustment.
        
        Args:
            symbol: Trading pair
            action: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            position_size: Position size in USD
            pnl: Profit/Loss in USD
        """
        trade = {
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": position_size,
            "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.trade_history.append(trade)
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        
        self._save_state()
        
        safe_log(
            f"[{self.name}] Trade recorded: {symbol} {action} | "
            f"PnL=${pnl:.2f} | Balance=${self.current_balance:.2f}"
        )
    
    def get_stats(self) -> Dict:
        """Get current risk statistics"""
        total_exposure = sum(self.active_positions.values())
        current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        
        recent_trades = self.trade_history[-20:]
        win_count = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        win_rate = win_count / len(recent_trades) if recent_trades else 0.0
        
        return {
            "balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown_pct": current_drawdown,
            "total_exposure_usd": total_exposure,
            "exposure_pct": total_exposure / self.current_balance if self.current_balance > 0 else 0,
            "daily_trades": self._get_daily_count(),
            "recent_win_rate": win_rate,
            "total_trades": len(self.trade_history),
            "active_positions": len(self.active_positions)
        }
