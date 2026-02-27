"""
engine/evaluator.py
Pure exit evaluation logic — no Redis I/O, no side effects.
All inputs explicit, all outputs deterministic.

Decision priority order (Phase 0 spec §3):
 1. R_net <= -R_stop_effective         → FULL_CLOSE  (stop hit)
 2. Trailing pullback from max_R_seen  → FULL_CLOSE  (trailing)
 3. R_net >= r_target_base             → FULL_CLOSE  (target hit)
 4. R_net >= partial_75_r, stage < 3   → PARTIAL_75
 5. R_net >= partial_50_r, stage < 2   → PARTIAL_50
 6. R_net >= partial_25_r, stage < 1   → PARTIAL_25
 7. Otherwise                          → HOLD
"""

from dataclasses import dataclass
from typing import Tuple
from engine.config import HarvestV2Config
from engine.state import SymbolState


VOL_FACTOR_MAP = {
    "LOW_VOL":  "vol_factor_low",
    "MID_VOL":  "vol_factor_mid",
    "HIGH_VOL": "vol_factor_high",
}


@dataclass
class EvalResult:
    decision: str
    R_net: float
    R_stop: float
    R_target: float
    regime: str
    vol_factor: float
    emit_reason: str    # "DECISION_CHANGE" | "R_STEP" | "SUPPRESSED"


class ExitEvaluator:
    """
    Stateless evaluator.
    Receives SymbolState (already updated with current ATR push)
    and returns EvalResult.
    """

    def evaluate(
        self,
        pos_unrealized_pnl: float,
        pos_entry_risk_usdt: float,
        state: SymbolState,
        heat: float,
        cfg: HarvestV2Config,
    ) -> Tuple[str, EvalResult]:
        """
        Returns (decision, EvalResult).
        Does NOT mutate state — caller is responsible for state updates.
        """
        # ── R_net ──────────────────────────────────────────────────────
        R_net = pos_unrealized_pnl / pos_entry_risk_usdt

        # ── Regime + vol factor ─────────────────────────────────────────
        regime = state.detect_regime()
        vol_factor = getattr(cfg, VOL_FACTOR_MAP[regime])

        # ── Dynamic stop (vol only — heat does NOT affect stop) ─────────
        R_stop = cfg.r_stop_base * vol_factor

        # ── Dynamic target (vol + heat) ─────────────────────────────────
        heat_adj   = 1.0 - heat * cfg.heat_sensitivity
        R_target   = cfg.r_target_base * vol_factor * heat_adj

        # ── Update trailing peak (before decision check) ─────────────────
        # NOTE: caller calls state.update_max_R(R_net) AFTER getting result
        # to allow evaluator to remain pure. We pass current max_R_seen.

        # ── Decision cascade (priority order) ────────────────────────────
        decision = self._decide(R_net, R_stop, R_target, state, cfg)

        # ── Emission guard ────────────────────────────────────────────────
        if state.should_emit(decision, R_net, cfg.r_emit_step):
            reason = (
                "DECISION_CHANGE" if decision != state.last_decision
                else "R_STEP"
            )
        else:
            decision = "HOLD_SUPPRESSED"
            reason   = "SUPPRESSED"

        result = EvalResult(
            decision=decision,
            R_net=R_net,
            R_stop=R_stop,
            R_target=R_target,
            regime=regime,
            vol_factor=vol_factor,
            emit_reason=reason,
        )
        return decision, result

    # ------------------------------------------------------------------ #

    @staticmethod
    def _decide(
        R_net: float,
        R_stop: float,
        R_target: float,
        state: SymbolState,
        cfg: HarvestV2Config,
    ) -> str:
        # 1. Hard stop
        if R_net <= -R_stop:
            return "FULL_CLOSE"

        # 2. Trailing stop
        if state.trailing_triggered(R_net, cfg.trailing_step):
            return "FULL_CLOSE"

        # 3. Target
        if R_net >= R_target:
            return "FULL_CLOSE"

        # 4–6. Partials (monotonic — never go back a stage)
        if R_net >= cfg.partial_75_r and state.partial_stage < 3:
            return "PARTIAL_75"

        if R_net >= cfg.partial_50_r and state.partial_stage < 2:
            return "PARTIAL_50"

        if R_net >= cfg.partial_25_r and state.partial_stage < 1:
            return "PARTIAL_25"

        # 7. Hold
        return "HOLD"
