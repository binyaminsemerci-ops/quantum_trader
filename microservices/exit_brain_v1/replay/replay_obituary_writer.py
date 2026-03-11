"""
ReplayObituaryWriter — Builds structured post-mortems for exit decisions.

Phase 5 replay component. Shadow-only.

Reads from: Phase 1-4 shadow streams via ReplayLoader
Writes to: quantum:stream:exit.obituary.shadow via ShadowPublisher
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from ..models.trade_exit_obituary import TradeExitObituary
from ..models.action_candidate import VALID_ACTIONS
from .replay_loader import ReplayLoader
from .outcome_reconstructor import OutcomeReconstructor, OutcomePathResult

logger = logging.getLogger(__name__)

# Epsilon to avoid division by zero
_EPS = 1e-10

# Default evaluation horizon (4 hours)
DEFAULT_HORIZON_SEC = 14400.0

# Best-exit-window scan: rolling window width (5 minutes)
BEST_WINDOW_WIDTH_SEC = 300.0


class ReplayObituaryWriter:
    """
    Builds TradeExitObituary records by:
    1. Loading decision traces and upstream snapshots
    2. Reconstructing post-decision price/PnL path
    3. Computing quality scores (regret, preservation, opportunity capture)
    4. Publishing obituary to shadow stream

    Shadow-only. Never writes to execution paths.
    """

    def __init__(
        self,
        redis_client,
        publisher,
        horizon_seconds: float = DEFAULT_HORIZON_SEC,
    ) -> None:
        """
        Args:
            redis_client: Synchronous redis.Redis instance.
            publisher: ShadowPublisher instance.
            horizon_seconds: Post-decision evaluation horizon.
        """
        self._loader = ReplayLoader(redis_client)
        self._reconstructor = OutcomeReconstructor(redis_client)
        self._publisher = publisher
        self._horizon_seconds = horizon_seconds

    def build_obituaries_for_window(
        self,
        start_ts: float,
        end_ts: float,
    ) -> List[TradeExitObituary]:
        """
        Build obituaries for all decisions in a time window.

        Args:
            start_ts: Window start (epoch).
            end_ts: Window end (epoch).

        Returns:
            List of built obituaries. Also published to shadow stream.
        """
        traces = self._loader.load_decision_traces(start_ts, end_ts)
        policies = self._loader.load_policy_decisions(start_ts, end_ts)

        # Index policies by decision_id for fast lookup
        policy_by_id: Dict[str, Dict] = {}
        for p in policies:
            did = p.get("decision_id", "")
            if did:
                policy_by_id[did] = p

        obituaries: List[TradeExitObituary] = []
        for trace in traces:
            decision_id = trace.get("source_decision_id", "")
            policy = policy_by_id.get(decision_id)
            if not policy:
                logger.warning(
                    "[ObituaryWriter] No policy found for decision_id=%s, skipping",
                    decision_id,
                )
                continue

            obit = self.build_trade_exit_obituary(trace, policy)
            if obit is not None:
                obituaries.append(obit)

        return obituaries

    def build_trade_exit_obituary(
        self,
        trace: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Optional[TradeExitObituary]:
        """
        Build a single obituary from a decision trace and policy decision.

        Args:
            trace: Decoded decision trace dict.
            policy: Decoded policy decision dict.

        Returns:
            TradeExitObituary, or None if reconstruction fails critically.
        """
        now = time.time()
        symbol = trace.get("symbol", "")
        position_id = trace.get("position_id", "")
        quality_flags: List[str] = []

        if not symbol or not position_id:
            logger.error("[ObituaryWriter] Missing symbol or position_id in trace")
            return None

        # Load upstream snapshots near decision time
        decision_ts = self._safe_float(trace.get("trace_timestamp", 0))
        if decision_ts <= 0:
            quality_flags.append("MISSING_DECISION_TIMESTAMP")
            decision_ts = now - self._horizon_seconds  # fallback

        snapshots = self._loader.load_snapshots_for_decision(decision_ts, symbol)

        # Extract belief/hazard/utility snapshots
        belief_snapshot = self._extract_belief_snapshot(snapshots.get("belief"))
        hazard_snapshot = self._extract_hazard_snapshot(snapshots.get("hazard"))
        utility_snapshot = self._extract_utility_snapshot(snapshots.get("utility"))

        if not belief_snapshot:
            quality_flags.append("MISSING_UPSTREAM_BELIEF")
        if not hazard_snapshot:
            quality_flags.append("MISSING_UPSTREAM_HAZARD")

        # Extract state info for reconstruction
        state = snapshots.get("state") or {}
        entry_price = self._safe_float(state.get("entry_price", 0))
        side = state.get("side", "LONG")
        quantity = self._safe_float(state.get("quantity", 1))
        open_ts = self._safe_float(state.get("open_timestamp", 0))
        upstream_qflags = self._parse_quality_flags(state.get("data_quality_flags", ""))

        # Reconstruct post-decision outcome
        outcome = self._reconstructor.reconstruct(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            decision_timestamp=decision_ts,
            horizon_seconds=self._horizon_seconds,
            quantity=quantity,
        )
        quality_flags.extend(outcome.quality_flags)

        # Compute best exit window
        best_start, best_end, best_pnl, best_reason = self.estimate_best_exit_window(
            outcome
        )

        # Extract recommendation from policy
        chosen = policy.get("chosen_action", "HOLD")
        chosen_utility = self._safe_float(policy.get("chosen_action_utility", 0))
        chosen_confidence = self._safe_float(policy.get("decision_confidence", 0))
        policy_passed = self._safe_bool(policy.get("policy_passed", False))
        reason_codes = self._parse_json_list(policy.get("reason_codes", "[]"))

        # Compute scores
        regret = self.compute_regret_score(outcome, best_pnl)
        preservation = self.compute_preservation_score(outcome)
        opportunity = self.compute_opportunity_capture_score(outcome, best_pnl)

        # Build obituary
        lifecycle = (now - open_ts) if open_ts > 0 else 0.0
        obit = TradeExitObituary(
            position_id=position_id,
            symbol=symbol,
            obituary_id=str(uuid.uuid4()),
            obituary_timestamp=now,
            open_timestamp=open_ts if open_ts > 0 else decision_ts,
            lifecycle_duration_seconds=max(0.0, lifecycle),
            peak_unrealized_pnl=outcome.peak_pnl,
            peak_unrealized_pnl_pct=(
                outcome.peak_pnl / max(entry_price * quantity, _EPS)
                if entry_price > 0 else 0.0
            ),
            max_drawdown_after_peak=outcome.max_drawdown_after_peak,
            best_exit_window_start=best_start,
            best_exit_window_end=best_end,
            best_exit_window_pnl=best_pnl,
            best_exit_window_reason=best_reason,
            recommended_action_at_decision=chosen,
            recommended_action_utility=chosen_utility,
            recommended_action_confidence=chosen_confidence,
            policy_passed=policy_passed,
            policy_reason_codes=reason_codes,
            belief_snapshot=belief_snapshot,
            hazard_snapshot=hazard_snapshot,
            utility_snapshot=utility_snapshot,
            decision_trace_ref=trace.get("trace_id", ""),
            source_decision_id=trace.get("source_decision_id", ""),
            post_decision_price_path=outcome.price_path[:100],  # Cap for storage
            post_decision_pnl_path=outcome.pnl_path[:100],
            evaluation_horizon_seconds=self._horizon_seconds,
            regret_score=regret,
            preservation_score=preservation,
            opportunity_capture_score=opportunity,
            quality_flags=quality_flags,
            upstream_quality_flags=upstream_qflags,
        )

        # Publish
        self.publish_obituary_shadow(obit)
        return obit

    def load_decision_trace(
        self,
        trace_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a specific decision trace by ID."""
        return self._loader.load_decision_trace_by_id(trace_id)

    def load_upstream_snapshots(
        self,
        decision_timestamp: float,
        symbol: str,
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Load upstream snapshots closest to a decision timestamp."""
        return self._loader.load_snapshots_for_decision(decision_timestamp, symbol)

    def reconstruct_post_decision_path(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        decision_timestamp: float,
        quantity: float = 1.0,
    ) -> OutcomePathResult:
        """Reconstruct price/PnL path after a decision point."""
        return self._reconstructor.reconstruct(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            decision_timestamp=decision_timestamp,
            horizon_seconds=self._horizon_seconds,
            quantity=quantity,
        )

    def estimate_best_exit_window(
        self,
        outcome: OutcomePathResult,
    ) -> tuple:
        """
        Find the optimal exit window within the post-decision path.

        Scans for the rolling window where average PnL was highest.

        Returns:
            (window_start_ts, window_end_ts, window_pnl, reason)
        """
        if len(outcome.pnl_path) < 2 or len(outcome.timestamps) < 2:
            return (0.0, 0.0, 0.0, "insufficient_data")

        # Best single point (maximum PnL)
        best_pnl = outcome.peak_pnl
        best_ts = outcome.peak_pnl_timestamp

        # v1: Best exit = peak PnL moment with ±BEST_WINDOW_WIDTH_SEC/2
        half_w = BEST_WINDOW_WIDTH_SEC / 2.0
        best_start = max(best_ts - half_w, outcome.timestamps[0])
        best_end = min(best_ts + half_w, outcome.timestamps[-1])
        reason = "peak_pnl_scan"

        return (best_start, best_end, best_pnl, reason)

    def compute_regret_score(
        self,
        outcome: OutcomePathResult,
        best_possible_pnl: float,
    ) -> float:
        """
        Regret = (best_possible - actual) / max(|best_possible|, eps).

        Returns:
            [0, 1] — 0 = no regret, 1 = maximum regret.
        """
        actual = outcome.final_pnl
        if abs(best_possible_pnl) < _EPS:
            return 0.0
        regret = (best_possible_pnl - actual) / max(abs(best_possible_pnl), _EPS)
        return max(0.0, min(1.0, regret))

    def compute_preservation_score(
        self,
        outcome: OutcomePathResult,
    ) -> float:
        """
        Preservation = 1 - (max_drawdown_after_peak / max(peak_pnl, eps)).

        Returns:
            [0, 1] — 1 = perfect capital preservation, 0 = total giveback.
        """
        peak = outcome.peak_pnl
        if peak <= _EPS:
            return 1.0  # No gains to preserve
        dd = outcome.max_drawdown_after_peak
        score = 1.0 - (dd / max(peak, _EPS))
        return max(0.0, min(1.0, score))

    def compute_opportunity_capture_score(
        self,
        outcome: OutcomePathResult,
        best_possible_pnl: float,
    ) -> float:
        """
        Opportunity capture = actual / max(best_possible, eps).

        Returns:
            [0, 1] — 1 = captured all upside, 0 = captured nothing.
        """
        actual = outcome.final_pnl
        if best_possible_pnl <= _EPS:
            return 1.0 if actual >= 0 else 0.0
        score = actual / max(best_possible_pnl, _EPS)
        return max(0.0, min(1.0, score))

    def publish_obituary_shadow(self, obituary: TradeExitObituary) -> Optional[str]:
        """Publish obituary to shadow stream via ShadowPublisher."""
        errors = obituary.validate()
        if errors:
            logger.warning(
                "[ObituaryWriter] Obituary validation warnings for %s: %s",
                obituary.position_id, errors,
            )
            if "INCOMPLETE" not in ",".join(obituary.quality_flags):
                obituary.quality_flags.append("INCOMPLETE_OBITUARY")
        return self._publisher.publish_obituary(obituary)

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _safe_float(val: Any) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    @staticmethod
    def _parse_json_list(val: Any) -> List[str]:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            import json
            try:
                parsed = json.loads(val)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return [v.strip() for v in val.split(",") if v.strip()]
        return []

    @staticmethod
    def _parse_quality_flags(val: Any) -> List[str]:
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        return []

    @staticmethod
    def _extract_belief_snapshot(belief: Optional[Dict]) -> Dict[str, float]:
        if not belief:
            return {}
        keys = [
            "exit_pressure", "hold_conviction", "directional_edge",
            "uncertainty_total", "data_completeness",
        ]
        result: Dict[str, float] = {}
        for k in keys:
            v = belief.get(k)
            if v is not None:
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    pass
        return result

    @staticmethod
    def _extract_hazard_snapshot(hazard: Optional[Dict]) -> Dict[str, float]:
        if not hazard:
            return {}
        keys = [
            "drawdown_hazard", "reversal_hazard", "volatility_hazard",
            "time_decay_hazard", "regime_hazard", "ensemble_hazard",
            "composite_hazard",
        ]
        result: Dict[str, float] = {}
        for k in keys:
            v = hazard.get(k)
            if v is not None:
                try:
                    result[k] = float(v)
                except (ValueError, TypeError):
                    pass
        return result

    @staticmethod
    def _extract_utility_snapshot(utility: Optional[Dict]) -> List[Dict]:
        if not utility:
            return []
        # Utility stream stores all_candidates as JSON
        raw = utility.get("all_candidates")
        if isinstance(raw, list):
            return raw[:3]  # Top 3
        if isinstance(raw, str):
            import json
            try:
                parsed = json.loads(raw)
                return parsed[:3] if isinstance(parsed, list) else []
            except (json.JSONDecodeError, ValueError):
                return []
        return []
