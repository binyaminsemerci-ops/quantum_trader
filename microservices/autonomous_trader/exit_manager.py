"""
Exit Manager - Monitors positions and decides when to exit
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import httpx

from .position_tracker import Position
from .funding_rate_filter import fetch_funding_rate

logger = logging.getLogger(__name__)

# Exit threshold: 0.5% daily bleed (0.00167 per 8h period)
# More conservative than entry filter (0.1% per 8h)
FUNDING_BLEED_THRESHOLD = 0.00167  # ~0.5% per day


@dataclass
class ExitDecision:
    """Exit decision for a position"""
    symbol: str
    action: str  # PARTIAL_CLOSE, CLOSE, HOLD
    percentage: float  # 0.0-1.0
    reason: str
    hold_score: int
    exit_score: int
    factors: Dict


class ExitManager:
    """
    Monitors positions and decides when to exit
    
    Uses:
    - AI Engine exit evaluator (Phase 3D)
    - Stop loss triggers
    - Take profit triggers
    - Time-based exits
    """
    
    def __init__(
        self,
        redis_client,
        ai_engine_url: str = "http://127.0.0.1:8001",
        use_ai_exits: bool = True
    ):
        self.redis = redis_client
        self.ai_engine_url = ai_engine_url
        self.use_ai_exits = use_ai_exits
        # Increased timeout: AI Engine can take 10-30s when busy with predictions
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Statistics
        self.evaluations_count = 0
        self.exits_triggered = 0
        
        logger.info(f"[ExitManager] Initialized (AI exits: {use_ai_exits})")
    
    async def evaluate_position(self, position: Position) -> ExitDecision:
        """
        Evaluate position for exit
        
        Priority:
        1. Stop loss check (immediate)
        2. Take profit check (immediate)
        3. AI exit evaluation (dynamic)
        """
        self.evaluations_count += 1
        
        # Check stop loss
        sl_triggered = self._check_stop_loss(position)
        if sl_triggered:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason="stop_loss_triggered",
                hold_score=0,
                exit_score=10,
                factors={"sl_triggered": True}
            )
        
        # Check take profit
        tp_triggered = self._check_take_profit(position)
        if tp_triggered:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason="take_profit_triggered",
                hold_score=0,
                exit_score=10,
                factors={"tp_triggered": True}
            )
        
        # Check funding fee bleed (before emergency SL)
        funding_bleed = await self._check_funding_bleed(position)
        if funding_bleed:
            return funding_bleed
        
        # Emergency stop loss for losing positions (R < -1.5)
        if position.R_net < -1.5:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"emergency_stop_loss (R={position.R_net:.2f})",
                hold_score=0,
                exit_score=10,
                factors={"emergency_sl": True, "R_net": position.R_net}
            )
        
        # AI exit evaluation (for winners)
        if self.use_ai_exits and position.R_net > 0.5:
            ai_decision = await self._call_ai_exit_evaluator(position)
            if ai_decision:
                return ai_decision
        
        # AI exit evaluation (for losers too, but not emergency)
        # Let AI decide if we should hold or cut losses at R < -1.5
        if self.use_ai_exits and -1.5 <= position.R_net <= 0.5:
            ai_decision = await self._call_ai_exit_evaluator(position)
            if ai_decision:
                return ai_decision
        
        # Default: hold
        return ExitDecision(
            symbol=position.symbol,
            action="HOLD",
            percentage=0.0,
            reason="no_exit_conditions",
            hold_score=5,
            exit_score=0,
            factors={}
        )
    
    def _check_stop_loss(self, position: Position) -> bool:
        """Check if stop loss triggered"""
        if position.stop_loss == 0:
            return False
        
        if position.side == "LONG":
            return position.current_price <= position.stop_loss
        else:
            return position.current_price >= position.stop_loss
    
    def _check_take_profit(self, position: Position) -> bool:
        """Check if take profit triggered"""
        if position.take_profit == 0:
            return False
        
        if position.side == "LONG":
            return position.current_price >= position.take_profit
        else:
            return position.current_price <= position.take_profit
    
    async def _check_funding_bleed(self, position: Position) -> Optional[ExitDecision]:
        """
        Check if position is bleeding excessive funding fees
        
        Closes positions that:
        - Pay funding fees (rate > 0 for LONG, rate < 0 for SHORT)
        - Exceed threshold: 0.5% daily bleed (~0.00167 per 8h)
        
        Returns:
            ExitDecision if bleeding excessive fees, None otherwise
        """
        try:
            # Fetch current funding rate
            funding_rate = await fetch_funding_rate(position.symbol)
            
            # Calculate notional value and fee per period
            notional = position.position_qty * position.current_price
            
            # Determine if we're PAYING fees
            is_paying = False
            if position.side == "LONG" and funding_rate > 0:
                is_paying = True
            elif position.side == "SHORT" and funding_rate < 0:
                is_paying = True
            
            if not is_paying:
                return None  # We're receiving funding, keep position
            
            # Calculate fee cost per 8h period
            fee_per_period = notional * abs(funding_rate)
            
            # Check if exceeds threshold
            if abs(funding_rate) > FUNDING_BLEED_THRESHOLD:
                daily_bleed = fee_per_period * 3  # 3 periods per day
                margin = notional / position.leverage
                bleed_percent = (daily_bleed / margin) * 100
                
                logger.warning(
                    f"[💸 FUNDING BLEED] {position.symbol}: "
                    f"Rate={funding_rate:.5f} ({funding_rate*100:.3f}%), "
                    f"Fee={fee_per_period:.2f} USDT/8h, "
                    f"Daily={daily_bleed:.2f} USDT ({bleed_percent:.1f}% of margin)"
                )
                
                return ExitDecision(
                    symbol=position.symbol,
                    action="CLOSE",
                    percentage=1.0,
                    reason=f"excessive_funding_bleed ({funding_rate*100:.3f}%/8h)",
                    hold_score=0,
                    exit_score=9,
                    factors={
                        "funding_bleed": True,
                        "funding_rate": funding_rate,
                        "fee_per_period": fee_per_period,
                        "daily_bleed": daily_bleed,
                        "bleed_percent": bleed_percent
                    }
                )
            
            return None
        
        except Exception as e:
            logger.debug(f"[ExitManager] Funding check failed for {position.symbol}: {e}")
            return None
    
    async def _call_ai_exit_evaluator(self, position: Position) -> Optional[ExitDecision]:
        """
        Call AI Engine exit evaluator API
        
        Returns:
            ExitDecision if AI available, None otherwise
        """
        try:
            payload = {
                "symbol": position.symbol,
                "side": position.side,
                "entry_price": position.entry_price,
                "current_price": position.current_price,
                "position_qty": position.position_qty,
                "entry_timestamp": position.entry_timestamp,
                "age_sec": position.age_sec,
                "R_net": position.R_net,
                "R_history": position.R_history,
                "entry_regime": position.entry_regime,
                "entry_confidence": position.entry_confidence,
                "peak_price": position.peak_price
            }
            
            response = await self.http_client.post(
                f"{self.ai_engine_url}/api/v1/evaluate-exit",
                json=payload
            )
            
            if response.status_code != 200:
                logger.warning(f"[ExitManager] AI Engine returned {response.status_code}")
                return None
            
            data = response.json()
            
            return ExitDecision(
                symbol=position.symbol,
                action=data.get("action", "HOLD"),
                percentage=data.get("percentage", 0.0),
                reason=data.get("reason", "ai_evaluation"),
                hold_score=data.get("hold_score", 0),
                exit_score=data.get("exit_score", 0),
                factors=data.get("factors", {})
            )
        
        except httpx.TimeoutException:
            logger.warning("[ExitManager] AI Engine timeout - using R-threshold fallback")
            return self._get_fallback_exit_decision(position)
        except Exception as e:
            logger.error(f"[ExitManager] AI Engine call failed: {e} - using R-threshold fallback")
            return self._get_fallback_exit_decision(position)
    
    def _get_fallback_exit_decision(self, position: Position) -> ExitDecision:
        """
        Fallback exit decision when AI Engine is unavailable.
        Uses R-threshold logic with fee protection (simulates AI Engine logic).
        NOTE: entry_timestamp from Redis snapshots is unreliable, so age-based
        conditions are de-prioritized.
        """
        R = position.R_net
        age_hours = position.age_sec / 3600
        
        # Emergency exits for big winners
        if R >= 5:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"fallback_extreme_profit (R={R:.2f})",
                hold_score=0,
                exit_score=10,
                factors={"fallback": True, "R_net": R}
            )
        
        # Take profit on good winners  
        if R >= 3:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"fallback_high_profit (R={R:.2f})",
                hold_score=2,
                exit_score=8,
                factors={"fallback": True, "R_net": R}
            )
        
        # Partial close on decent winners
        if R >= 2:
            return ExitDecision(
                symbol=position.symbol,
                action="PARTIAL_CLOSE",
                percentage=0.5,
                reason=f"fallback_decent_profit (R={R:.2f})",
                hold_score=4,
                exit_score=6,
                factors={"fallback": True, "R_net": R}
            )
        
        # === FEE PROTECTION (matches AI Engine) ===
        # Close positions with R >= 0.5 to secure profit before fees erode
        # AI Engine triggers FEE_PROTECTION at this level
        if R >= 0.5:
            return ExitDecision(
                symbol=position.symbol,
                action="CLOSE",
                percentage=1.0,
                reason=f"fallback_fee_protection (R={R:.2f})",
                hold_score=3,
                exit_score=8,
                factors={"fallback": True, "R_net": R}
            )
        
        # Default: hold
        return ExitDecision(
            symbol=position.symbol,
            action="HOLD",
            percentage=0.0,
            reason="fallback_hold",
            hold_score=5,
            exit_score=0,
            factors={"fallback": True, "R_net": R}
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
