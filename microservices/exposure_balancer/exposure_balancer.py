"""
Adaptive Exposure Balancer - Phase 4P
Real-time portfolio risk management and exposure balancing

Monitors and adjusts:
- Total margin utilization (max 85%)
- Per-symbol exposure (max 15%)
- Diversification (min 5 symbols)
- Cross-exchange divergence risk
- Position sizing based on confidence

Actions:
- Reduce overexposed positions
- Close high-risk trades
- Hedge divergence risk
- Rebalance symbol weights

Integration:
- Phase 4O+ (ILFv2, RL Agent)
- Phase 4M+ (Cross-Exchange Intelligence)
- Auto Executor (position management)
"""

import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from redis import Redis
except ImportError:
    Redis = None

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExposureMetrics:
    """Current portfolio exposure metrics"""
    total_margin_used: float
    total_margin_available: float
    margin_utilization: float  # 0-1
    symbol_count: int
    avg_confidence: float
    cross_divergence: float
    positions: Dict[str, float]  # symbol -> margin USD
    symbol_exposures: Dict[str, float]  # symbol -> percentage


@dataclass
class BalanceAction:
    """Action to take for rebalancing"""
    action_type: str  # reduce, close, expand, hedge, rebalance
    symbol: Optional[str]
    target_size: Optional[float]
    reason: str
    priority: int  # 1=critical, 2=important, 3=optimization
    timestamp: float


class ExposureBalancer:
    """
    Real-time adaptive exposure balancer
    
    Continuously monitors portfolio risk and adjusts positions to maintain:
    - Margin utilization < 85%
    - Per-symbol exposure < 15%
    - Minimum 5 symbols (diversification)
    - Low cross-exchange divergence exposure
    """
    
    def __init__(self, redis_client: Optional[Redis] = None, config: Optional[Dict] = None):
        """
        Initialize Exposure Balancer
        
        Args:
            redis_client: Redis client for data and commands
            config: Configuration overrides
        """
        self.redis = redis_client
        self.config = config or {}
        
        # Risk limits
        self.max_margin_util = self.config.get("max_margin_util", 0.85)  # 85%
        self.max_symbol_exposure = self.config.get("max_symbol_exposure", 0.15)  # 15%
        self.min_diversification = self.config.get("min_diversification", 5)
        self.divergence_threshold = self.config.get("divergence_threshold", 0.03)  # 3%
        
        # Rebalance settings
        self.rebalance_interval = self.config.get("rebalance_interval", 10)  # seconds
        self.aggressive_mode = self.config.get("aggressive_mode", False)
        
        # State tracking
        self.last_adjust = time.time()
        self.last_rebalance = time.time()
        self.portfolio: Dict[str, float] = {}
        self.total_margin_used = 0.0
        self.total_margin_available = 100000.0  # Default, updated from Redis
        
        # Statistics
        self.actions_taken = 0
        self.actions_by_type: Dict[str, int] = {}
        self.last_metrics: Optional[ExposureMetrics] = None
        
        logger.info(
            f"[Exposure-Balancer] Initialized | "
            f"Max Margin: {self.max_margin_util*100:.0f}% | "
            f"Max Symbol: {self.max_symbol_exposure*100:.0f}% | "
            f"Min Symbols: {self.min_diversification}"
        )
    
    def update_portfolio(self) -> ExposureMetrics:
        """
        Update portfolio state from Redis
        
        Returns:
            ExposureMetrics with current state
        """
        if not self.redis:
            logger.warning("[Exposure-Balancer] No Redis client, cannot update portfolio")
            return self._empty_metrics()
        
        try:
            # Get open positions (symbol -> margin USD)
            positions_raw = self.redis.hgetall("quantum:positions:open") or {}
            self.portfolio = {
                symbol.decode() if isinstance(symbol, bytes) else symbol: float(margin)
                for symbol, margin in positions_raw.items()
            }
            
            # Calculate total margin used
            self.total_margin_used = sum(abs(margin) for margin in self.portfolio.values())
            
            # Get total margin available
            margin_total_str = self.redis.get("quantum:margin:total")
            if margin_total_str:
                self.total_margin_available = float(margin_total_str)
            
            # Calculate margin utilization
            margin_util = (
                self.total_margin_used / self.total_margin_available
                if self.total_margin_available > 0
                else 0.0
            )
            
            # Get average confidence
            confidence_str = self.redis.get("quantum:meta:confidence")
            avg_confidence = float(confidence_str) if confidence_str else 0.5
            
            # Get cross-exchange divergence
            divergence_str = self.redis.get("quantum:cross:divergence")
            cross_divergence = float(divergence_str) if divergence_str else 0.0
            
            # Calculate per-symbol exposures
            symbol_exposures = {}
            if self.total_margin_used > 0:
                for symbol, margin in self.portfolio.items():
                    symbol_exposures[symbol] = abs(margin) / self.total_margin_used
            
            # Build metrics
            metrics = ExposureMetrics(
                total_margin_used=self.total_margin_used,
                total_margin_available=self.total_margin_available,
                margin_utilization=margin_util,
                symbol_count=len(self.portfolio),
                avg_confidence=avg_confidence,
                cross_divergence=cross_divergence,
                positions=self.portfolio.copy(),
                symbol_exposures=symbol_exposures
            )
            
            self.last_metrics = metrics
            
            logger.debug(
                f"[Exposure-Balancer] Portfolio updated | "
                f"Symbols: {metrics.symbol_count} | "
                f"Margin: {margin_util*100:.1f}% | "
                f"Confidence: {avg_confidence:.2f} | "
                f"Divergence: {cross_divergence:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"[Exposure-Balancer] Error updating portfolio: {e}")
            return self._empty_metrics()
    
    def _empty_metrics(self) -> ExposureMetrics:
        """Return empty metrics for error cases"""
        return ExposureMetrics(
            total_margin_used=0.0,
            total_margin_available=100000.0,
            margin_utilization=0.0,
            symbol_count=0,
            avg_confidence=0.5,
            cross_divergence=0.0,
            positions={},
            symbol_exposures={}
        )
    
    def assess_risk(self, metrics: ExposureMetrics) -> List[BalanceAction]:
        """
        Assess portfolio risk and generate rebalancing actions
        
        Args:
            metrics: Current exposure metrics
        
        Returns:
            List of BalanceAction to take (sorted by priority)
        """
        actions = []
        
        # Check 1: Margin overload (CRITICAL)
        if metrics.margin_utilization > self.max_margin_util:
            actions.append(BalanceAction(
                action_type="reduce_margin",
                symbol=None,
                target_size=None,
                reason=f"Margin overload: {metrics.margin_utilization*100:.1f}% > {self.max_margin_util*100:.0f}%",
                priority=1,
                timestamp=time.time()
            ))
            
            # Alert
            self._send_alert("margin_overload", {
                "utilization": metrics.margin_utilization,
                "threshold": self.max_margin_util
            })
        
        # Check 2: Per-symbol overexposure (IMPORTANT)
        for symbol, exposure in metrics.symbol_exposures.items():
            if exposure > self.max_symbol_exposure:
                target = metrics.total_margin_used * self.max_symbol_exposure
                actions.append(BalanceAction(
                    action_type="reduce",
                    symbol=symbol,
                    target_size=target,
                    reason=f"Symbol overexposure: {exposure*100:.1f}% > {self.max_symbol_exposure*100:.0f}%",
                    priority=2,
                    timestamp=time.time()
                ))
        
        # Check 3: Underdiversification (IMPORTANT)
        if metrics.symbol_count < self.min_diversification and metrics.symbol_count > 0:
            actions.append(BalanceAction(
                action_type="expand",
                symbol=None,
                target_size=None,
                reason=f"Underdiversified: {metrics.symbol_count} < {self.min_diversification} symbols",
                priority=2,
                timestamp=time.time()
            ))
        
        # Check 4: High cross-exchange divergence (IMPORTANT)
        if metrics.cross_divergence > self.divergence_threshold:
            actions.append(BalanceAction(
                action_type="hedge",
                symbol=None,
                target_size=None,
                reason=f"High divergence: {metrics.cross_divergence*100:.2f}% > {self.divergence_threshold*100:.1f}%",
                priority=2,
                timestamp=time.time()
            ))
            
            # Alert
            self._send_alert("high_divergence", {
                "divergence": metrics.cross_divergence,
                "threshold": self.divergence_threshold
            })
        
        # Check 5: Symbol weight imbalance (OPTIMIZATION)
        if metrics.symbol_count > 0:
            avg_exposure = 1.0 / metrics.symbol_count
            for symbol, exposure in metrics.symbol_exposures.items():
                ratio = exposure / avg_exposure
                if ratio > 1.5:  # 50% above average
                    target = metrics.total_margin_used * avg_exposure
                    actions.append(BalanceAction(
                        action_type="rebalance",
                        symbol=symbol,
                        target_size=target,
                        reason=f"Symbol overweight: {ratio:.1f}x average",
                        priority=3,
                        timestamp=time.time()
                    ))
        
        # Sort by priority (1=highest)
        actions.sort(key=lambda a: a.priority)
        
        return actions
    
    def execute_action(self, action: BalanceAction):
        """
        Execute a rebalancing action
        
        Args:
            action: BalanceAction to execute
        """
        if not self.redis:
            logger.warning("[Exposure-Balancer] No Redis client, cannot execute action")
            return
        
        try:
            # Build command for executor
            command = {
                "timestamp": action.timestamp,
                "action": action.action_type,
                "reason": action.reason,
                "priority": action.priority
            }
            
            if action.symbol:
                command["symbol"] = action.symbol
            
            if action.target_size is not None:
                command["target_size"] = action.target_size
            
            # Send to executor commands stream
            self.redis.xadd(
                "quantum:stream:executor.commands",
                command,
                maxlen=500
            )
            
            # Update statistics
            self.actions_taken += 1
            self.actions_by_type[action.action_type] = (
                self.actions_by_type.get(action.action_type, 0) + 1
            )
            
            logger.info(
                f"[Exposure-Balancer] Action executed: {action.action_type} | "
                f"Symbol: {action.symbol or 'N/A'} | "
                f"Reason: {action.reason}"
            )
            
        except Exception as e:
            logger.error(f"[Exposure-Balancer] Error executing action: {e}")
    
    def _send_alert(self, alert_type: str, data: Dict):
        """Send alert to Redis stream"""
        if not self.redis:
            return
        
        try:
            alert = {
                "timestamp": time.time(),
                "type": alert_type,
                **data
            }
            
            self.redis.xadd(
                "quantum:stream:exposure.alerts",
                alert,
                maxlen=500
            )
            
            logger.warning(f"[Exposure-Balancer] ALERT: {alert_type} - {data}")
            
        except Exception as e:
            logger.error(f"[Exposure-Balancer] Error sending alert: {e}")
    
    def rebalance(self):
        """
        Main rebalancing logic - called periodically
        
        Steps:
        1. Update portfolio state
        2. Assess risks
        3. Execute high-priority actions immediately
        4. Queue lower-priority actions for next cycle
        """
        now = time.time()
        
        # Update portfolio
        metrics = self.update_portfolio()
        
        # Assess risk
        actions = self.assess_risk(metrics)
        
        if not actions:
            logger.debug("[Exposure-Balancer] No actions needed")
            return
        
        # Execute critical actions (priority 1) immediately
        critical_actions = [a for a in actions if a.priority == 1]
        for action in critical_actions:
            self.execute_action(action)
        
        # Execute important actions (priority 2) if interval elapsed
        if now - self.last_rebalance >= self.rebalance_interval:
            important_actions = [a for a in actions if a.priority == 2]
            for action in important_actions:
                self.execute_action(action)
            
            self.last_rebalance = now
        
        # Execute optimization actions (priority 3) less frequently
        if now - self.last_adjust >= self.rebalance_interval * 3:
            optimization_actions = [a for a in actions if a.priority == 3]
            for action in optimization_actions:
                self.execute_action(action)
            
            self.last_adjust = now
    
    def get_statistics(self) -> Dict:
        """Get balancer statistics"""
        return {
            "actions_taken": self.actions_taken,
            "actions_by_type": self.actions_by_type.copy(),
            "last_metrics": {
                "margin_utilization": round(self.last_metrics.margin_utilization, 3) if self.last_metrics else 0.0,
                "symbol_count": self.last_metrics.symbol_count if self.last_metrics else 0,
                "avg_confidence": round(self.last_metrics.avg_confidence, 2) if self.last_metrics else 0.0,
                "cross_divergence": round(self.last_metrics.cross_divergence, 4) if self.last_metrics else 0.0
            } if self.last_metrics else {},
            "limits": {
                "max_margin_util": self.max_margin_util,
                "max_symbol_exposure": self.max_symbol_exposure,
                "min_diversification": self.min_diversification,
                "divergence_threshold": self.divergence_threshold
            }
        }


# Global singleton
_exposure_balancer: Optional[ExposureBalancer] = None


def get_exposure_balancer(
    redis_client: Optional[Redis] = None,
    config: Optional[Dict] = None
) -> ExposureBalancer:
    """Get or create global ExposureBalancer instance"""
    global _exposure_balancer
    
    if _exposure_balancer is None:
        _exposure_balancer = ExposureBalancer(redis_client=redis_client, config=config)
        logger.info("[Exposure-Balancer] Global balancer initialized")
    
    return _exposure_balancer
