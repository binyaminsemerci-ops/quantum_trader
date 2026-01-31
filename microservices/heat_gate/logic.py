"""
P2.6 Portfolio Heat Gate - Core Logic Module
=============================================
Deterministic heat calculation and moderation policy.
FAIL-CLOSED: missing/stale inputs => block or NOOP (configurable via HEAT_FAIL_MODE).
"""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple


def clamp01(value: float) -> float:
    """Clamp value to [0, 1] range."""
    return max(0.0, min(1.0, value))


class HeatGateLogic:
    """Portfolio heat gate logic with deterministic formulas."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize heat gate logic with configuration.
        
        Args:
            config: Dictionary with all ENV values (targets, weights, thresholds, etc.)
        """
        self.config = config
        
        # Targets
        self.gross_target = float(config.get("GROSS_TARGET_USD", 5000))
        self.dd_target = float(config.get("DD_TARGET_USD", 250))
        self.burst_target = float(config.get("BURST_TARGET_USD", 250))
        self.fee_target = float(config.get("FEE_TARGET_USD", 50))
        self.churn_target = float(config.get("CHURN_TARGET", 1.0))
        
        # Weights
        self.w_exp = float(config.get("W_EXP", 1.0))
        self.w_dd = float(config.get("W_DD", 1.0))
        self.w_burst = float(config.get("W_BURST", 1.0))
        self.w_fee = float(config.get("W_FEE", 0.5))
        self.w_churn = float(config.get("W_CHURN", 0.5))
        self.weight_sum = self.w_exp + self.w_dd + self.w_burst + self.w_fee + self.w_churn
        
        # Thresholds
        self.t_warm = float(config.get("T_WARM", 0.45))
        self.t_hot = float(config.get("T_HOT", 0.70))
        
        # Partials
        self.warm_partial = float(config.get("WARM_PARTIAL", 0.50))
        self.hot_partial = float(config.get("HOT_PARTIAL", 0.25))
        
        # Hold-close (shadow-only marker)
        self.enable_hold_close = config.get("ENABLE_HOLD_CLOSE", "false").lower() == "true"
        self.hold_kill_max = float(config.get("HOLD_KILL_MAX", 0.30))
        
        # Staleness
        self.stale_sec = int(config.get("STALE_SEC", 600))
    
    def compute_heat_score(self, portfolio_state: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compute portfolio heat score from state.
        
        Args:
            portfolio_state: Dict with exposure/dd/burst/fee/churn values
            
        Returns:
            Tuple of (heat_score, component_dict)
        """
        gross_exp = float(portfolio_state.get("gross_exposure_usd", 0))
        dd_ewma = float(portfolio_state.get("dd_ewma_usd", 0))
        burst_ewma = float(portfolio_state.get("lossburst_ewma_usd", 0))
        fee_ewma = float(portfolio_state.get("fee_burden_ewma_usd", 0))
        churn_ewma = float(portfolio_state.get("churn_ewma", 0))
        
        # Component scores (0..1)
        exposure01 = clamp01(gross_exp / self.gross_target) if self.gross_target > 0 else 0.0
        dd01 = clamp01(dd_ewma / self.dd_target) if self.dd_target > 0 else 0.0
        burst01 = clamp01(burst_ewma / self.burst_target) if self.burst_target > 0 else 0.0
        fee01 = clamp01(fee_ewma / self.fee_target) if self.fee_target > 0 else 0.0
        churn01 = clamp01(churn_ewma / self.churn_target) if self.churn_target > 0 else 0.0
        
        # Weighted sum
        heat_raw = (
            self.w_exp * exposure01 +
            self.w_dd * dd01 +
            self.w_burst * burst01 +
            self.w_fee * fee01 +
            self.w_churn * churn01
        )
        
        # Normalized
        heat_score = clamp01(heat_raw / self.weight_sum) if self.weight_sum > 0 else 0.0
        
        components = {
            "exposure01": round(exposure01, 4),
            "dd01": round(dd01, 4),
            "burst01": round(burst01, 4),
            "fee01": round(fee01, 4),
            "churn01": round(churn01, 4),
        }
        
        return heat_score, components
    
    def determine_heat_level(self, heat_score: float) -> str:
        """Determine heat level (cold/warm/hot) from score."""
        if heat_score < self.t_warm:
            return "cold"
        elif heat_score < self.t_hot:
            return "warm"
        else:
            return "hot"
    
    def apply_moderation_policy(
        self,
        in_action: str,
        heat_level: str,
        kill_score: Optional[float] = None
    ) -> Tuple[str, str, Optional[float]]:
        """
        Apply moderation policy to input action.
        
        Args:
            in_action: Original action from harvest proposal
            heat_level: cold/warm/hot/unknown
            kill_score: Optional kill_score from harvest proposal
            
        Returns:
            Tuple of (heat_action, out_action, recommended_partial)
            - heat_action: "NONE"|"DOWNGRADE_FULL_TO_PARTIAL"|"HOLD_CLOSE"
            - out_action: moderated action or original
            - recommended_partial: 0.25/0.50/0.75 or None
        """
        # Only moderate FULL_CLOSE_PROPOSED
        if in_action != "FULL_CLOSE_PROPOSED":
            return "NONE", in_action, None
        
        # Unknown heat => fail-open
        if heat_level == "unknown":
            return "NONE", in_action, None
        
        # Cold => no action needed
        if heat_level == "cold":
            return "NONE", in_action, None
        
        # Warm => downgrade to 50% partial
        if heat_level == "warm":
            return "DOWNGRADE_FULL_TO_PARTIAL", "PARTIAL_50_PROPOSED", self.warm_partial
        
        # Hot => check for hold-close or downgrade to 25% partial
        if heat_level == "hot":
            # HOLD_CLOSE is shadow-only marker for analytics
            if self.enable_hold_close and kill_score is not None and kill_score < self.hold_kill_max:
                return "HOLD_CLOSE", "HOLD_CLOSE_SHADOW", None
            else:
                return "DOWNGRADE_FULL_TO_PARTIAL", "PARTIAL_25_PROPOSED", self.hot_partial
        
        # Fallback (should not reach)
        return "NONE", in_action, None
    
    def process_harvest_proposal(
        self,
        proposal: Dict[str, Any],
        portfolio_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a harvest proposal and generate heat decision.
        
        Args:
            proposal: Harvest proposal message from stream
            portfolio_state: Portfolio state from Redis (or None if missing)
            
        Returns:
            Heat decision dict ready for stream output
        """
        ts_now = int(time.time())
        symbol = proposal.get("symbol", "UNKNOWN")
        plan_id = proposal.get("plan_id", "unknown")
        in_action = proposal.get("action", "UNKNOWN")
        kill_score = proposal.get("kill_score")
        
        # Convert kill_score to float if present
        if kill_score is not None:
            try:
                kill_score = float(kill_score)
            except (ValueError, TypeError):
                kill_score = None
        
        # Fail-open: check if portfolio state is missing or stale
        if portfolio_state is None or not portfolio_state:
            return self._build_failopen_decision(
                ts_now, symbol, plan_id, in_action, "missing_inputs"
            )
        
        # Check staleness (try both ts_epoch and timestamp for compatibility)
        state_ts = portfolio_state.get("ts_epoch") or portfolio_state.get("timestamp")
        if state_ts is None:
            return self._build_failopen_decision(
                ts_now, symbol, plan_id, in_action, "missing_ts"
            )
        
        try:
            state_ts = int(state_ts)
            age_sec = ts_now - state_ts
            if age_sec > self.stale_sec or age_sec < 0:
                return self._build_failopen_decision(
                    ts_now, symbol, plan_id, in_action, "stale_inputs", age_sec
                )
        except (ValueError, TypeError):
            return self._build_failopen_decision(
                ts_now, symbol, plan_id, in_action, "invalid_ts"
            )
        
        # Compute heat score
        try:
            heat_score, components = self.compute_heat_score(portfolio_state)
            heat_level = self.determine_heat_level(heat_score)
            
            # Apply moderation policy
            heat_action, out_action, recommended_partial = self.apply_moderation_policy(
                in_action, heat_level, kill_score
            )
            
            # Build debug JSON (keep small)
            debug_data = {
                "components": components,
                "thresholds": {"warm": self.t_warm, "hot": self.t_hot}
            }
            debug_json = json.dumps(debug_data, separators=(',', ':'))[:400]
            
            # Get operation mode from env (default shadow)
            heat_mode = os.getenv("HEAT_MODE", "shadow").lower()
            
            return {
                "ts_epoch": ts_now,
                "symbol": symbol,
                "plan_id": plan_id,
                "in_action": in_action,
                "out_action": out_action,
                "action": out_action,
                "heat_level": heat_level,
                "heat_score": round(heat_score, 4),
                "heat_action": heat_action,
                "recommended_partial": recommended_partial if recommended_partial else "",
                "reason": "ok",
                "inputs_age_sec": age_sec,
                "mode": heat_mode,
                "debug_json": debug_json
            }
        
        except Exception as e:
            return self._build_failopen_decision(
                ts_now, symbol, plan_id, in_action, f"compute_error:{str(e)[:50]}"
            )
    
    def _build_failopen_decision(
        self,
        ts_now: int,
        symbol: str,
        plan_id: str,
        in_action: str,
        reason: str,
        age_sec: Optional[int] = None
    ) -> Dict[str, Any]:
        """Build fail decision (FAIL-CLOSED or FAIL-OPEN based on HEAT_FAIL_MODE env)."""
        fail_mode = os.getenv("HEAT_FAIL_MODE", "CLOSED").upper()
        heat_mode = os.getenv("HEAT_MODE", "shadow").lower()
        
        if fail_mode == "CLOSED":
            # FAIL-CLOSED: block execution when inputs missing/stale
            out_action = "NOOP"
            log_prefix = "FAIL-CLOSED"
        else:
            # FAIL-OPEN: allow execution (legacy behavior)
            out_action = in_action
            log_prefix = "FAIL-OPEN"
        
        return {
            "ts_epoch": ts_now,
            "symbol": symbol,
            "plan_id": plan_id,
            "in_action": in_action,
            "out_action": out_action,
            "action": out_action,
            "heat_level": "unknown",
            "heat_score": 0.0,
            "heat_action": "NONE",
            "recommended_partial": "",
            "reason": f"{log_prefix}({reason})",
            "inputs_age_sec": age_sec if age_sec is not None else "",
            "mode": heat_mode,
            "debug_json": json.dumps({"fail_mode": fail_mode})
        }
