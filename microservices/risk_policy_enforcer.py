"""
RISK POLICY ENFORCER — QUANTUM TRADER (FULL LIVE)

Maps RISK_POLICY_FULL_LIVE.md to executable runtime enforcement.

Authority: Systemisk — overstyrer RL, model, og menneskelig discretion.
Scope: Live trading med ekte kapital.
"""

import logging
import time
from enum import Enum
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import redis

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Global system state (§ GLOBAL META-METRIC)"""
    GO = "GO"  # All gates pass, trading allowed
    PAUSED = "PAUSED"  # Temporary block, auto-recovery possible
    NO_GO = "NO_GO"  # Critical failure, manual intervention required


class FailureType(Enum):
    """Failure classification (§ 8.1)"""
    INFRASTRUCTURE = "INFRASTRUCTURE"  # Hard stop
    LEARNING_PLANE = "LEARNING_PLANE"  # Pause
    MARKET_REGIME = "MARKET_REGIME"  # No new trades
    MODEL_ANOMALY = "MODEL_ANOMALY"  # Rollback/freeze


@dataclass
class RiskMetrics:
    """Runtime metrics collected for policy enforcement"""
    # LAYER 0
    redis_available: bool
    rl_feedback_heartbeat_age: Optional[float]  # seconds
    rl_trainer_heartbeat_age: Optional[float]
    
    # LAYER 1
    daily_pnl: float
    rolling_drawdown_pct: float
    consecutive_losses: int
    effective_leverage: float
    
    # LAYER 2
    symbol_in_whitelist: bool
    realized_volatility: Optional[float]
    spread_bps: Optional[float]
    
    # Meta
    system_state: SystemState
    failure_reason: Optional[str] = None


@dataclass
class RiskLimits:
    """Static risk limits from policy"""
    # LAYER 0
    heartbeat_max_age_sec: int = 30
    
    # LAYER 1
    max_leverage: float = 10.0
    daily_loss_limit: float = -1000.0  # USD
    rolling_drawdown_max_pct: float = 15.0
    rolling_drawdown_window_days: int = 30
    max_consecutive_losses: int = 5
    loss_streak_cooldown_minutes: int = 60
    
    # LAYER 2
    vol_min: float = 0.005  # 0.5%
    vol_max: float = 0.10  # 10%
    max_spread_bps: float = 10.0  # 0.1%
    symbol_whitelist: List[str] = None
    
    def __post_init__(self):
        if self.symbol_whitelist is None:
            self.symbol_whitelist = ["BTCUSDT", "ETHUSDT"]


class RiskPolicyEnforcer:
    """
    Central risk enforcement engine.
    
    Evaluates all layers (0-2) and computes SYSTEM_STATE.
    """
    
    def __init__(
        self,
        redis_client: redis.Redis,
        limits: Optional[RiskLimits] = None
    ):
        self.redis = redis_client
        self.limits = limits or RiskLimits()
        
        # Redis keys
        self.rl_feedback_heartbeat_key = "quantum:svc:rl_feedback_v2:heartbeat"
        self.rl_trainer_heartbeat_key = "quantum:svc:rl_trainer:heartbeat"
        self.kill_switch_key = "quantum:global:kill_switch"
        
        # State tracking
        self.daily_trades: List[Dict] = []  # Reset at day boundary
        self.equity_curve: List[float] = []  # For drawdown calculation
        self.last_state_check: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        
        logger.info(f"RiskPolicyEnforcer initialized with limits: {self.limits}")
    
    # ============================================================
    # LAYER 0 — SYSTEMIC SAFETY (§ 4)
    # ============================================================
    
    def _check_layer0_infrastructure(self) -> Tuple[bool, Optional[str]]:
        """
        § 4.1 Infrastruktur Kill-Switch
        § 4.2 Execution Integrity
        
        Returns: (passed, failure_reason)
        """
        # Redis availability
        try:
            self.redis.ping()
        except Exception as e:
            return False, f"Redis unavailable: {e}"
        
        # Kill-switch check
        kill_switch = self.redis.get(self.kill_switch_key)
        if kill_switch and kill_switch.decode() == "1":
            return False, "Kill-switch activated"
        
        # RL Feedback V2 heartbeat
        try:
            feedback_hb = self.redis.get(self.rl_feedback_heartbeat_key)
            if not feedback_hb:
                return False, "RL Feedback V2 heartbeat missing"
            
            feedback_age = time.time() - float(feedback_hb.decode())
            if feedback_age > self.limits.heartbeat_max_age_sec:
                return False, f"RL Feedback V2 heartbeat stale ({feedback_age:.1f}s)"
        except Exception as e:
            return False, f"RL Feedback V2 heartbeat error: {e}"
        
        # RL Trainer heartbeat
        try:
            trainer_hb = self.redis.get(self.rl_trainer_heartbeat_key)
            if not trainer_hb:
                return False, "RL Trainer heartbeat missing"
            
            trainer_age = time.time() - float(trainer_hb.decode())
            if trainer_age > self.limits.heartbeat_max_age_sec:
                return False, f"RL Trainer heartbeat stale ({trainer_age:.1f}s)"
        except Exception as e:
            return False, f"RL Trainer heartbeat error: {e}"
        
        return True, None
    
    # ============================================================
    # LAYER 1 — CAPITAL PROTECTION (§ 5)
    # ============================================================
    
    def _check_layer1_capital_protection(self) -> Tuple[bool, Optional[str]]:
        """
        § 5.1 Max Leverage
        § 5.2 Daily Loss Limit
        § 5.3 Rolling Drawdown
        § 5.4 Loss Streak Breaker
        
        Returns: (passed, failure_reason)
        """
        # § 5.2 Daily loss limit
        daily_pnl = self._get_daily_pnl()
        if daily_pnl < self.limits.daily_loss_limit:
            return False, f"Daily loss limit breached: {daily_pnl:.2f} < {self.limits.daily_loss_limit}"
        
        # § 5.3 Rolling drawdown
        drawdown_pct = self._get_rolling_drawdown()
        if drawdown_pct > self.limits.rolling_drawdown_max_pct:
            return False, f"Rolling drawdown exceeded: {drawdown_pct:.2f}% > {self.limits.rolling_drawdown_max_pct}%"
        
        # § 5.4 Loss streak breaker
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            remaining = (self.cooldown_until - datetime.utcnow()).total_seconds()
            return False, f"Loss streak cooldown active ({remaining:.0f}s remaining)"
        
        return True, None
    
    def _get_daily_pnl(self) -> float:
        """Calculate PnL for current day"""
        # Reset at day boundary
        now = datetime.utcnow()
        self.daily_trades = [
            t for t in self.daily_trades
            if datetime.fromisoformat(t['timestamp']).date() == now.date()
        ]
        return sum(t['pnl'] for t in self.daily_trades)
    
    def _get_rolling_drawdown(self) -> float:
        """Calculate max drawdown over rolling window"""
        if not self.equity_curve:
            return 0.0
        
        # Keep only last N days
        window_start = datetime.utcnow() - timedelta(days=self.limits.rolling_drawdown_window_days)
        self.equity_curve = [
            e for e in self.equity_curve
            if datetime.fromisoformat(e['timestamp']) > window_start
        ]
        
        if not self.equity_curve:
            return 0.0
        
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = max(equity_values)
        trough = min(equity_values)
        
        if peak == 0:
            return 0.0
        
        return ((peak - trough) / peak) * 100.0
    
    def check_leverage(self, requested_leverage: float) -> Tuple[bool, float]:
        """
        § 5.1 Max leverage hard cap
        
        Returns: (allowed, clamped_leverage)
        """
        if requested_leverage > self.limits.max_leverage:
            logger.warning(
                f"Leverage clamped: {requested_leverage:.2f} -> {self.limits.max_leverage}"
            )
            return False, self.limits.max_leverage
        return True, requested_leverage
    
    def record_trade_outcome(self, pnl: float, symbol: str):
        """Record trade for Layer 1 tracking"""
        now = datetime.utcnow()
        
        self.daily_trades.append({
            'timestamp': now.isoformat(),
            'pnl': pnl,
            'symbol': symbol
        })
        
        # Update consecutive losses
        if pnl < 0:
            # Count recent consecutive losses
            consecutive = 0
            for trade in reversed(self.daily_trades):
                if trade['pnl'] < 0:
                    consecutive += 1
                else:
                    break
            
            # § 5.4 Loss streak breaker
            if consecutive >= self.limits.max_consecutive_losses:
                self.cooldown_until = now + timedelta(
                    minutes=self.limits.loss_streak_cooldown_minutes
                )
                logger.warning(
                    f"Loss streak breaker triggered: {consecutive} consecutive losses. "
                    f"Cooldown until {self.cooldown_until.isoformat()}"
                )
    
    def update_equity(self, equity: float):
        """Update equity curve for drawdown calculation"""
        self.equity_curve.append({
            'timestamp': datetime.utcnow().isoformat(),
            'equity': equity
        })
    
    # ============================================================
    # LAYER 2 — MARKET & REGIME GATING (§ 6)
    # ============================================================
    
    def _check_layer2_market_gating(
        self,
        symbol: str,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        § 6.1 Volatility Gate
        § 6.2 Liquidity Gate
        § 6.3 Symbol Whitelist
        
        Returns: (passed, failure_reason)
        """
        # § 6.3 Symbol whitelist
        if symbol not in self.limits.symbol_whitelist:
            return False, f"Symbol {symbol} not in whitelist"
        
        # § 6.1 Volatility gate
        if volatility is not None:
            if volatility < self.limits.vol_min:
                return False, f"Volatility too low: {volatility:.4f} < {self.limits.vol_min}"
            if volatility > self.limits.vol_max:
                return False, f"Volatility too high: {volatility:.4f} > {self.limits.vol_max}"
        
        # § 6.2 Liquidity gate
        if spread_bps is not None:
            if spread_bps > self.limits.max_spread_bps:
                return False, f"Spread too wide: {spread_bps:.2f} > {self.limits.max_spread_bps}"
        
        return True, None
    
    # ============================================================
    # GLOBAL SYSTEM STATE (§ GLOBAL META-METRIC)
    # ============================================================
    
    def compute_system_state(
        self,
        symbol: Optional[str] = None,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> RiskMetrics:
        """
        Compute global SYSTEM_STATE by evaluating all layers.
        
        This is the single source of truth for execution decisions.
        """
        self.last_state_check = datetime.utcnow()
        
        # Initialize metrics
        metrics = RiskMetrics(
            redis_available=False,
            rl_feedback_heartbeat_age=None,
            rl_trainer_heartbeat_age=None,
            daily_pnl=self._get_daily_pnl(),
            rolling_drawdown_pct=self._get_rolling_drawdown(),
            consecutive_losses=self._count_consecutive_losses(),
            effective_leverage=0.0,
            symbol_in_whitelist=(symbol in self.limits.symbol_whitelist) if symbol else False,
            realized_volatility=volatility,
            spread_bps=spread_bps,
            system_state=SystemState.NO_GO
        )
        
        # LAYER 0: Infrastructure (§ 4)
        layer0_pass, layer0_reason = self._check_layer0_infrastructure()
        if not layer0_pass:
            metrics.system_state = SystemState.NO_GO
            metrics.failure_reason = f"LAYER 0 FAIL: {layer0_reason}"
            logger.error(metrics.failure_reason)
            return metrics
        
        metrics.redis_available = True
        
        # Get heartbeat ages
        try:
            fb_hb = self.redis.get(self.rl_feedback_heartbeat_key)
            if fb_hb:
                metrics.rl_feedback_heartbeat_age = time.time() - float(fb_hb.decode())
            
            tr_hb = self.redis.get(self.rl_trainer_heartbeat_key)
            if tr_hb:
                metrics.rl_trainer_heartbeat_age = time.time() - float(tr_hb.decode())
        except Exception:
            pass
        
        # LAYER 1: Capital Protection (§ 5)
        layer1_pass, layer1_reason = self._check_layer1_capital_protection()
        if not layer1_pass:
            metrics.system_state = SystemState.PAUSED
            metrics.failure_reason = f"LAYER 1 FAIL: {layer1_reason}"
            logger.warning(metrics.failure_reason)
            return metrics
        
        # LAYER 2: Market Gating (§ 6)
        if symbol:
            layer2_pass, layer2_reason = self._check_layer2_market_gating(
                symbol=symbol,
                volatility=volatility,
                spread_bps=spread_bps
            )
            if not layer2_pass:
                metrics.system_state = SystemState.PAUSED
                metrics.failure_reason = f"LAYER 2 FAIL: {layer2_reason}"
                logger.warning(metrics.failure_reason)
                return metrics
        
        # All layers pass
        metrics.system_state = SystemState.GO
        metrics.failure_reason = None
        logger.debug("All risk layers passed. SYSTEM_STATE = GO")
        
        return metrics
    
    def _count_consecutive_losses(self) -> int:
        """Count consecutive losses from recent trades"""
        count = 0
        for trade in reversed(self.daily_trades):
            if trade['pnl'] < 0:
                count += 1
            else:
                break
        return count
    
    # ============================================================
    # EXECUTION GATE (§ 4.2)
    # ============================================================
    
    def allow_trade(
        self,
        symbol: str,
        requested_leverage: float,
        volatility: Optional[float] = None,
        spread_bps: Optional[float] = None
    ) -> Tuple[bool, SystemState, Optional[str]]:
        """
        Final execution gate.
        
        Returns: (allowed, system_state, reason)
        
        Usage:
            allowed, state, reason = enforcer.allow_trade(...)
            if not allowed:
                logger.error(f"Trade blocked: {reason}")
                return ABORT_EXECUTION
        """
        # Compute system state
        metrics = self.compute_system_state(
            symbol=symbol,
            volatility=volatility,
            spread_bps=spread_bps
        )
        
        if metrics.system_state != SystemState.GO:
            return False, metrics.system_state, metrics.failure_reason
        
        # Check leverage (Layer 1 enforcement)
        leverage_ok, _ = self.check_leverage(requested_leverage)
        if not leverage_ok:
            return False, SystemState.PAUSED, f"Leverage exceeds cap: {requested_leverage} > {self.limits.max_leverage}"
        
        return True, SystemState.GO, None
    
    # ============================================================
    # FAILURE HANDLER (§ 8)
    # ============================================================
    
    def handle_failure(self, failure_type: FailureType, details: str):
        """
        Central failure handler (§ 8.1)
        
        Failure classification:
        - INFRASTRUCTURE → HARD STOP
        - LEARNING_PLANE → PAUSE
        - MARKET_REGIME → NO NEW TRADES
        - MODEL_ANOMALY → ROLLBACK/FREEZE
        """
        logger.error(f"Failure detected: {failure_type.value} - {details}")
        
        if failure_type == FailureType.INFRASTRUCTURE:
            # Hard stop: activate kill-switch
            self.redis.set(self.kill_switch_key, "1")
            logger.critical("HARD STOP: Kill-switch activated due to infrastructure failure")
        
        elif failure_type == FailureType.LEARNING_PLANE:
            # Pause: set cooldown
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=30)
            logger.warning(f"PAUSE: Learning plane failure. Cooldown until {self.cooldown_until}")
        
        elif failure_type == FailureType.MARKET_REGIME:
            # No new trades: handled by Layer 2 gates
            logger.warning("NO NEW TRADES: Market regime violation")
        
        elif failure_type == FailureType.MODEL_ANOMALY:
            # Rollback/freeze: set extended cooldown
            self.cooldown_until = datetime.utcnow() + timedelta(hours=2)
            logger.warning(f"MODEL ANOMALY: Extended cooldown until {self.cooldown_until}")


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_enforcer(redis_url: str = "redis://localhost:6379") -> RiskPolicyEnforcer:
    """Factory function for risk enforcer"""
    redis_client = redis.from_url(redis_url)
    return RiskPolicyEnforcer(redis_client)


def log_risk_metrics(metrics: RiskMetrics):
    """Pretty-print risk metrics for monitoring"""
    logger.info("=" * 60)
    logger.info("RISK METRICS SNAPSHOT")
    logger.info("=" * 60)
    logger.info(f"SYSTEM_STATE: {metrics.system_state.value}")
    logger.info(f"Failure Reason: {metrics.failure_reason or 'None'}")
    logger.info("")
    logger.info("LAYER 0 — Infrastructure:")
    logger.info(f"  Redis: {'✓' if metrics.redis_available else '✗'}")
    logger.info(f"  RL Feedback HB age: {metrics.rl_feedback_heartbeat_age:.1f}s" if metrics.rl_feedback_heartbeat_age else "  RL Feedback HB age: N/A")
    logger.info(f"  RL Trainer HB age: {metrics.rl_trainer_heartbeat_age:.1f}s" if metrics.rl_trainer_heartbeat_age else "  RL Trainer HB age: N/A")
    logger.info("")
    logger.info("LAYER 1 — Capital:")
    logger.info(f"  Daily PnL: ${metrics.daily_pnl:.2f}")
    logger.info(f"  Rolling Drawdown: {metrics.rolling_drawdown_pct:.2f}%")
    logger.info(f"  Consecutive Losses: {metrics.consecutive_losses}")
    logger.info("")
    logger.info("LAYER 2 — Market:")
    logger.info(f"  Symbol Whitelist: {'✓' if metrics.symbol_in_whitelist else '✗'}")
    logger.info(f"  Volatility: {metrics.realized_volatility:.4f}" if metrics.realized_volatility else "  Volatility: N/A")
    logger.info(f"  Spread: {metrics.spread_bps:.2f} bps" if metrics.spread_bps else "  Spread: N/A")
    logger.info("=" * 60)
