"""models: internal data models for exit_management_agent (PATCH-1 / PATCH-7A / PATCH-7B).

All dataclasses in this module are pure value objects — no Redis I/O,
no side effects.  Tests can construct them without any infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PositionSnapshot:
    """
    Parsed from a quantum:position:{symbol} Redis hash.

    Field origin (quantum:position:{symbol} hash):
        symbol          — from key name
        side            — "LONG" | "SHORT"
        quantity        — "quantity" field (note: NOT "position_qty")
        entry_price     — "entry_price" field
        mark_price      — derived: entry_price ± unrealized_pnl/quantity
                          (overridden by quantum:ticker:{symbol}.markPrice if available)
        leverage        — "leverage" field  (≥ 1.0)
        stop_loss       — "stop_loss" field (0.0 = not set)
        take_profit     — "take_profit" field (0.0 = not set)
        unrealized_pnl  — "unrealized_pnl" field (USD)
        entry_risk_usdt — "entry_risk_usdt" field (0.0 = unknown, harvest_brain hasn't run yet)
        sync_timestamp  — "sync_timestamp" field (unix epoch of last hash update)
    """

    symbol: str
    side: str            # "LONG" | "SHORT"
    quantity: float      # always > 0 for open positions after parsing
    entry_price: float
    mark_price: float    # current estimated price
    leverage: float      # ≥ 1.0
    stop_loss: float     # 0.0 = not set
    take_profit: float   # 0.0 = not set
    unrealized_pnl: float
    entry_risk_usdt: float  # 0.0 = not yet computed
    sync_timestamp: float   # unix epoch
    opened_at: Optional[float] = None  # unix epoch when position was opened

    @property
    def is_long(self) -> bool:
        return self.side.upper() in ("LONG", "BUY")

    @property
    def is_short(self) -> bool:
        return self.side.upper() in ("SHORT", "SELL")


@dataclass
class PerceptionResult:
    """
    Computed observations about a single position.
    Fed into DecisionEngine.decide().

    Notes:
        peak_price   — best price observed since this agent process started.
                       Lower bound: resets on service restart.
        age_sec      — wall-clock time since agent first observed this symbol.
                       Also a lower bound; resets on restart. Used for TIME_STOP.
        r_effective_t1  — leverage-scaled partial-harvest trigger.
        r_effective_lock — leverage-scaled break-even / trail tighten trigger.
    """

    snapshot: PositionSnapshot
    R_net: float
    peak_price: float
    age_sec: float
    distance_to_sl_pct: float   # > 0 = safe buffer; < 0 = SL already breached. 0 if no SL.
    giveback_pct: float         # [0.0–1.0] fraction of peak profit given back.
    r_effective_t1: float
    r_effective_lock: float


@dataclass
class ExitDecision:
    """
    Shadow exit decision produced by DecisionEngine.

    NEVER written to application/execution streams in PATCH-1.
    Serialised and written to quantum:stream:exit.audit only.

    dry_run is always True in PATCH-1; audit.py enforces this with a RuntimeError
    if it detects dry_run=False.
    """

    snapshot: PositionSnapshot
    action: str    # HOLD | MOVE_TO_BREAKEVEN | TIGHTEN_TRAIL | PARTIAL_CLOSE_25 | FULL_CLOSE | TIME_STOP_EXIT
    reason: str
    urgency: str   # LOW | MEDIUM | HIGH | EMERGENCY
    R_net: float
    confidence: float         # 0.0–1.0
    suggested_sl: Optional[float]          # For SL-modifying actions; None for HOLD/FULL_CLOSE
    suggested_qty_fraction: Optional[float]  # For PARTIAL_CLOSE_25; None otherwise
    dry_run: bool              # Always True in PATCH-1
    # PATCH-7A: formula scoring state attached when ScoringEngine ran.
    # None when a hard guard (drawdown stop / SL breach / time stop) bypassed scoring.
    score_state: Optional["ExitScoreState"] = None
    # PATCH-7B: Qwen3 layer result — present only when scoring_mode="ai" and
    # formula_action is in _ALLOWED_ACTIONS (TIGHTEN_TRAIL/MOVE_TO_BREAKEVEN skip).
    # None on hard-guard decisions, formula/shadow mode, or skipped actions.
    qwen3_result: Optional["Qwen3LayerResult"] = None

    @property
    def is_actionable(self) -> bool:
        """True if any action other than HOLD is recommended."""
        return self.action != "HOLD"


@dataclass(frozen=True)
class ExitScoreState:
    """
    [PATCH-7A] Output of ScoringEngine.score() for one position.

    Contains all five dimension scores, the composite exit_score, and the
    formula engine's recommended action/urgency/confidence/reason.

    Attached to ExitDecision.score_state when scoring is active.
    score_state=None means the decision came from a hard guard (emergency
    drawdown, SL breach, time stop) that bypassed scoring entirely.

    Dimension scores are all in [0.0, 1.0].  Higher = stronger exit pressure.
    """

    # ── Position context (pass-through for AI contract in PATCH-7B) ──────────
    symbol: str
    side: str
    R_net: float
    age_sec: float
    age_fraction: float          # age_sec / max_hold_sec, clamped [0, 1]
    giveback_pct: float
    distance_to_sl_pct: float
    peak_price: float
    mark_price: float
    entry_price: float
    leverage: float
    r_effective_t1: float
    r_effective_lock: float

    # ── Five dimension scores ─────────────────────────────────────────────────
    d_r_loss: float       # D1: loss pressure relative to emergency threshold
    d_r_gain: float       # D2: depth into profit target zone
    d_giveback: float     # D3: giveback fraction, zero-gated below break-even
    d_time: float         # D4: convex time-decay pressure
    d_sl_proximity: float # D5: closeness to SL within 5% buffer window

    # ── Composite score ───────────────────────────────────────────────────────
    exit_score: float     # weighted sum in [0.0, 1.0]

    # ── Formula recommendation ────────────────────────────────────────────────
    formula_action: str
    formula_urgency: str
    formula_confidence: float
    formula_reason: str


@dataclass(frozen=True)
class Qwen3LayerResult:
    """
    [PATCH-7B] Output of Qwen3Layer.evaluate() for one position.

    action      — one of: HOLD | PARTIAL_CLOSE_25 | FULL_CLOSE | TIME_STOP_EXIT
    confidence  — model-reported confidence, clamped to [0.0, 1.0]
    reason      — model-reported reason string, truncated to 200 chars
    fallback    — True when model output was rejected and formula_action was used
    latency_ms  — wall-clock HTTP round-trip in milliseconds
    raw         — first 500 chars of raw model response body (for audit/debug)
    """

    action: str
    confidence: float
    reason: str
    fallback: bool
    latency_ms: float
    raw: str


@dataclass(frozen=True)
class DecisionSnapshot:
    """
    [PATCH-8A] Complete point-in-time snapshot of every signal present at the
    moment a decision was written to the audit stream.

    Persisted to quantum:hash:exit.decision:{decision_id} with a configurable
    TTL.  The decision_id is also added to
    quantum:set:exit.pending_decisions:{symbol} so outcome labelling jobs can
    find all open (not-yet-resolved) decisions for a symbol in O(1).

    All float fields are left as float (not pre-formatted strings) so readers
    can do arithmetic without parsing.  The audit stream record still stores
    them as formatted strings to preserve existing behaviour.
    """

    decision_id: str        # uuid4 string — unique per audit write
    ts_epoch: int           # unix timestamp at time of write
    # ── position context ────────────────────────────────────────────────────
    symbol: str
    side: str               # "LONG" | "SHORT"
    entry_price: float
    mark_price: float
    quantity: float
    unrealized_pnl: float
    # ── formula engine output ────────────────────────────────────────────────
    formula_action: str     # HOLD | PARTIAL_CLOSE_25 | FULL_CLOSE | …
    formula_conf: float     # formula_confidence from ExitScoreState
    # ── Qwen3 layer output ───────────────────────────────────────────────────
    qwen3_action: str       # "" when Qwen3 was not called
    qwen3_conf: float       # 0.0 when Qwen3 was not called
    qwen3_reason: str       # "" when Qwen3 was not called
    qwen3_fallback: bool    # True when Qwen3 output rejected; False when not called
    # ── live (final) decision ────────────────────────────────────────────────
    live_action: str        # action that was acted upon
    live_conf: float        # confidence of the live action
    diverged: bool          # True when formula_action != qwen3_action (and both present)
    exit_score: float       # composite score from ScoringEngine (0.0 if hard-guard)
