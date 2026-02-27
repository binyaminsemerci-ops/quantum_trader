"""
risk_kernel_core.py
===================
Single authoritative source for all trade-open SL/TP/sizing decisions.

Every new position MUST call ``compute_trade_plan()`` before any order is
placed.  The returned ``trade_plan`` dict is the ONLY allowed source of
``sl_price``, ``tp_price``, and ``position_size`` for the execution layer.

Rules
-----
* No fallbacks.  If SL or TP cannot be computed → FAIL HARD (raise).
* Wraps ``risk_kernel_stops.compute_proposal()`` for regime/sigma logic.
* Emits a deterministic JSON audit line on every call (logger.info).

Author: quantum_trader risk layer
"""

from __future__ import annotations

import json
import logging
import math
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional

logger = logging.getLogger("risk_kernel_core")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tick_round(price: float, tick_size: float, direction: str) -> float:
    """Round price to nearest tick.  direction='up' or 'down'."""
    if tick_size <= 0:
        return round(price, 8)
    tick = Decimal(str(tick_size))
    p = Decimal(str(price))
    if direction == "up":
        return float((p / tick).to_integral_value(rounding=ROUND_UP) * tick)
    return float((p / tick).to_integral_value(rounding=ROUND_DOWN) * tick)


def _dominant_regime(regime_probs: dict) -> str:
    """Return 'trend' | 'mr' | 'chop' based on highest probability."""
    return max(regime_probs, key=regime_probs.get)


def _validate_trade_plan(plan: dict, symbol: str) -> None:
    """Raise ValueError if any required field is missing or invalid."""
    required = ("entry_price", "sl_price", "tp_price", "position_size",
                "risk_amount", "r_multiple", "regime_label",
                "sigma_value", "volatility_factor", "confidence_score")
    for field in required:
        if plan.get(field) is None:
            raise ValueError(
                f"[RISK_KERNEL_CORE] FATAL: field '{field}' missing in "
                f"trade_plan for {symbol}"
            )
    if not plan["sl_price"] or plan["sl_price"] <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: sl_price={plan['sl_price']} invalid for {symbol}"
        )
    if not plan["tp_price"] or plan["tp_price"] <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: tp_price={plan['tp_price']} invalid for {symbol}"
        )
    is_long = plan.get("side", "BUY").upper() == "BUY"
    entry = plan["entry_price"]
    if is_long:
        if plan["sl_price"] >= entry:
            raise ValueError(
                f"[RISK_KERNEL_CORE] FATAL: LONG sl_price={plan['sl_price']} "
                f">= entry={entry} for {symbol}"
            )
        if plan["tp_price"] <= entry:
            raise ValueError(
                f"[RISK_KERNEL_CORE] FATAL: LONG tp_price={plan['tp_price']} "
                f"<= entry={entry} for {symbol}"
            )
    else:
        if plan["sl_price"] <= entry:
            raise ValueError(
                f"[RISK_KERNEL_CORE] FATAL: SHORT sl_price={plan['sl_price']} "
                f"<= entry={entry} for {symbol}"
            )
        if plan["tp_price"] >= entry:
            raise ValueError(
                f"[RISK_KERNEL_CORE] FATAL: SHORT tp_price={plan['tp_price']} "
                f">= entry={entry} for {symbol}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_trade_plan(
    symbol: str,
    side: str,               # "BUY" or "SELL"
    entry_price: float,
    account_balance: float,
    risk_pct: float,         # e.g. 0.01 for 1 %
    market_state: dict,      # {sigma, mu, ts, regime_probs:{trend,mr,chop}}
    tick_size: float,
    theta: Optional[dict] = None,  # override DEFAULT_THETA_RISK if supplied
) -> dict:
    """
    Compute a fully validated trade plan for a new position.

    Parameters
    ----------
    symbol          : e.g. "BTCUSDT"
    side            : "BUY" (long) or "SELL" (short)
    entry_price     : confirmed fill price
    account_balance : total equity in USDT
    risk_pct        : fraction of balance to risk  (e.g. 0.01 = 1 %)
    market_state    : dict with keys:
                        sigma         – normalised daily volatility  (e.g. 0.02)
                        mu            – expected drift
                        ts            – trend strength scalar
                        regime_probs  – {trend: float, mr: float, chop: float}
    tick_size       : minimum price increment for the symbol
    theta           : optional override for risk parameters (see risk_kernel_stops)

    Returns
    -------
    dict with exactly these 10 keys (all validated, no None values):
        entry_price, sl_price, tp_price, position_size, risk_amount,
        r_multiple, regime_label, sigma_value, volatility_factor,
        confidence_score

    Raises
    ------
    ValueError  if SL/TP cannot be computed or fail directional check.
                NEVER returns silently — callers must not catch blindly.
    """
    if entry_price <= 0:
        raise ValueError(f"[RISK_KERNEL_CORE] entry_price must be > 0, got {entry_price}")
    if account_balance <= 0:
        raise ValueError(f"[RISK_KERNEL_CORE] account_balance must be > 0, got {account_balance}")
    if not (0 < risk_pct <= 0.10):
        raise ValueError(f"[RISK_KERNEL_CORE] risk_pct={risk_pct} out of range (0, 0.10]")

    # ------------------------------------------------------------------
    # 1. Extract market state
    # ------------------------------------------------------------------
    sigma: float = float(market_state.get("sigma") or 0)
    mu: float = float(market_state.get("mu") or 0)
    ts: float = float(market_state.get("ts") or 0)
    regime_probs: dict = market_state.get("regime_probs") or {}

    if sigma <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: sigma={sigma} invalid for {symbol}. "
            "market_state must contain a valid 'sigma' > 0."
        )
    if not regime_probs or not all(k in regime_probs for k in ("trend", "mr", "chop")):
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: regime_probs missing or incomplete for {symbol}. "
            f"Got: {regime_probs}"
        )

    # ------------------------------------------------------------------
    # 2. Regime/sigma kernel — delegate to risk_kernel_stops
    # ------------------------------------------------------------------
    try:
        from ai_engine.risk_kernel_stops import compute_proposal, DEFAULT_THETA_RISK, PositionSnapshot
        effective_theta = theta or DEFAULT_THETA_RISK
        # Synthetic position snapshot at open (no history yet)
        # risk_kernel_stops expects "LONG" / "SHORT", not "BUY" / "SELL"
        _snap_side = "LONG" if side.upper() == "BUY" else "SHORT"
        snap = PositionSnapshot(
            symbol=symbol,
            side=_snap_side,
            entry_price=entry_price,
            current_price=entry_price,
            peak_price=entry_price,
            trough_price=entry_price,
            age_sec=0,
            unrealized_pnl=0.0,
            current_sl=None,
            current_tp=None,
        )
        proposal = compute_proposal(
            symbol=symbol,
            market_state=market_state,
            position=snap,
            theta=effective_theta,
        )
        raw_sl: Optional[float] = proposal.get("proposed_sl")
        raw_tp: Optional[float] = proposal.get("proposed_tp")
        reason_codes: list = proposal.get("reason_codes", [])
        audit_detail: dict = proposal.get("audit", {})
    except ImportError:
        # risk_kernel_stops not importable — compute inline using same math
        logger.warning(
            "[RISK_KERNEL_CORE] risk_kernel_stops not importable, using inline kernel"
        )
        raw_sl, raw_tp, reason_codes, audit_detail = _inline_kernel(
            symbol=symbol, side=side, entry_price=entry_price,
            sigma=sigma, ts=ts, regime_probs=regime_probs, theta=theta,
        )

    # ------------------------------------------------------------------
    # 3. FAIL HARD if kernel returned nothing
    # ------------------------------------------------------------------
    if not raw_sl or raw_sl <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: risk_kernel returned sl={raw_sl} "
            f"for {symbol}. No fallback permitted."
        )
    if not raw_tp or raw_tp <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: risk_kernel returned tp={raw_tp} "
            f"for {symbol}. No fallback permitted."
        )

    # ------------------------------------------------------------------
    # 4. Tick-round (conservative direction)
    # ------------------------------------------------------------------
    is_long = side.upper() == "BUY"
    if is_long:
        final_sl = _tick_round(raw_sl, tick_size, "down")   # below entry
        final_tp = _tick_round(raw_tp, tick_size, "up")     # above entry
    else:
        final_sl = _tick_round(raw_sl, tick_size, "up")     # above entry
        final_tp = _tick_round(raw_tp, tick_size, "down")   # below entry

    # ------------------------------------------------------------------
    # 5. Position sizing (fixed-risk)
    # ------------------------------------------------------------------
    risk_amount = account_balance * risk_pct
    sl_dist = abs(entry_price - final_sl)
    if sl_dist <= 0:
        raise ValueError(
            f"[RISK_KERNEL_CORE] FATAL: sl_dist=0 after tick-rounding for {symbol}"
        )
    position_size = risk_amount / sl_dist   # in base asset units

    # ------------------------------------------------------------------
    # 6. Derived metrics
    # ------------------------------------------------------------------
    tp_dist = abs(final_tp - entry_price)
    r_multiple = tp_dist / sl_dist if sl_dist > 0 else 0.0

    regime_probs_norm = regime_probs
    regime_label = _dominant_regime(regime_probs_norm)
    volatility_factor = sigma / 0.02 if sigma > 0 else 1.0  # normalised to 2% baseline
    # Confidence: blend of trend strength and inverse chop weight
    confidence_score = float(
        0.5 * regime_probs_norm.get("trend", 0)
        + 0.3 * (1 - regime_probs_norm.get("chop", 0))
        + 0.2 * min(ts, 1.0)
    )

    # ------------------------------------------------------------------
    # 7. Assemble and validate plan
    # ------------------------------------------------------------------
    trade_plan = {
        "symbol": symbol,
        "side": side.upper(),
        "entry_price": entry_price,
        "sl_price": final_sl,
        "tp_price": final_tp,
        "position_size": round(position_size, 8),
        "risk_amount": round(risk_amount, 4),
        "r_multiple": round(r_multiple, 3),
        "regime_label": regime_label,
        "sigma_value": sigma,
        "volatility_factor": round(volatility_factor, 4),
        "confidence_score": round(confidence_score, 4),
        # audit extras
        "_reason_codes": reason_codes,
        "_audit_detail": audit_detail,
        "_ts": time.time(),
    }

    _validate_trade_plan(trade_plan, symbol)

    # ------------------------------------------------------------------
    # 8. Deterministic audit log (one line, parseable JSON)
    # ------------------------------------------------------------------
    logger.info(
        "[TRADE_PLAN_AUDIT] %s",
        json.dumps({
            "symbol": symbol,
            "side": side.upper(),
            "entry": entry_price,
            "sl": final_sl,
            "tp": final_tp,
            "size": trade_plan["position_size"],
            "risk_usd": trade_plan["risk_amount"],
            "R": trade_plan["r_multiple"],
            "regime": regime_label,
            "sigma": sigma,
            "vol_factor": trade_plan["volatility_factor"],
            "confidence": trade_plan["confidence_score"],
            "reason_codes": reason_codes,
            "ts": trade_plan["_ts"],
        }, separators=(",", ":")),
    )

    return trade_plan


# ---------------------------------------------------------------------------
# Inline kernel (fallback if risk_kernel_stops import fails)
# ---------------------------------------------------------------------------

def _inline_kernel(
    symbol: str,
    side: str,
    entry_price: float,
    sigma: float,
    ts: float,
    regime_probs: dict,
    theta: Optional[dict],
) -> tuple:
    """
    Reproduces the core math of risk_kernel_stops.compute_proposal()
    for contexts where the module cannot be imported (e.g. unit tests
    running outside the ai_engine package).

    Returns (raw_sl, raw_tp, reason_codes, audit_detail).
    """
    # Default risk parameters (mirrors DEFAULT_THETA_RISK in risk_kernel_stops)
    _default_theta = {
        "sl_min_pct": 0.005,
        "tp_min_pct": 0.010,
        "k_sl": {"trend": 1.8, "mr": 1.2, "chop": 2.2},
        "k_tp": {"trend": 3.5, "mr": 2.0, "chop": 2.5},
        "k_trail": {"trend": 1.2, "mr": 0.8, "chop": 1.5},
        "tp_extend_gain": 0.5,
        "tp_extend_max": 0.3,
        "monotonic_sl": True,
    }
    th = theta if theta else _default_theta

    pi_trend = regime_probs.get("trend", 0.0)
    pi_mr = regime_probs.get("mr", 0.0)
    pi_chop = regime_probs.get("chop", 0.0)

    k_sl = th["k_sl"]
    k_tp = th["k_tp"]

    k_sl_w = k_sl["trend"] * pi_trend + k_sl["mr"] * pi_mr + k_sl["chop"] * pi_chop
    k_tp_w = k_tp["trend"] * pi_trend + k_tp["mr"] * pi_mr + k_tp["chop"] * pi_chop

    sl_pct = max(th["sl_min_pct"], k_sl_w * sigma)
    tp_pct = max(th["tp_min_pct"], k_tp_w * sigma)

    # TP extension in trending + strong trend
    if pi_trend > 0.3 and ts > 0.1:
        extend = min(th["tp_extend_gain"] * pi_trend, th["tp_extend_max"])
        tp_pct = tp_pct * (1 + extend)

    is_long = side.upper() == "BUY"
    raw_sl = entry_price * (1 - sl_pct) if is_long else entry_price * (1 + sl_pct)
    raw_tp = entry_price * (1 + tp_pct) if is_long else entry_price * (1 - tp_pct)

    reason_codes = ["inline_kernel"]
    audit_detail = {
        "k_sl_weighted": round(k_sl_w, 4),
        "k_tp_weighted": round(k_tp_w, 4),
        "sl_pct": round(sl_pct, 6),
        "tp_pct": round(tp_pct, 6),
        "pi_trend": pi_trend,
        "pi_mr": pi_mr,
        "pi_chop": pi_chop,
    }
    return raw_sl, raw_tp, reason_codes, audit_detail
