"""models: internal data models for exit_management_agent (PATCH-1).

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

    @property
    def is_actionable(self) -> bool:
        """True if any action other than HOLD is recommended."""
        return self.action != "HOLD"
