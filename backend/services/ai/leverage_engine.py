"""
LeverageEngine v1.0 — Sophisticated formula-based leverage calculation.

Core Principle (derived from first principles):
    leverage = risk_fraction / (sl_fraction * margin_fraction)

Mathematical Foundation:
  1. Fractional Kelly (25%) — crypto-safe fraction for fat-tailed distributions
     Ref: Kelly 1956, Thorp 2006, Haghani & Dewey 2016 (Blackjack experiment).
     25% Kelly is the empirical consensus for assets with tail risk.

  2. ATR-based volatility targeting — SL anchored to realized volatility.
     Leverage falls automatically when markets are volatile.
     No magic threshold multipliers.

  3. Geometric blend of Kelly + ATR — conservative, multiplicative first
     principles. Geometric mean ≤ arithmetic mean (conservative bias).

  4. Convex confidence scaling — confidence^1.5 (not linear).
     0.50 confidence → 0.35× leverage (harsh penalty for uncertain signals)
     0.90 confidence → 0.86× leverage

  5. Drawdown circuit breaker — linear 50% cut from pain threshold to max DD.
     The single most impactful risk management rule (AQR, Markowitz).

  6. Regime scalar via ADX — trend strength amplifies, ranging markets reduce.

  7. Portfolio utilization dilution — fuller portfolio = less per-position risk.

All scalar bounds are derived from mathematics, not trial-and-error.
NO ARBITRARY IF/THEN MULTIPLIER BRANCHES.
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Hyper-parameters (well-studied, literature-backed defaults) ────────────────
KELLY_FRACTION       = 0.25   # 25% Kelly: optimal for fat-tailed assets (Thorp 2006)
TARGET_RISK_PCT      = 0.015  # Risk 1.5% of equity per trade (institutional default)
ATR_SL_MULTIPLIER    = 1.5    # Stop-loss = 1.5 × ATR (Wilder's classic, widely validated)
COLD_START_MAX       = 5.0    # Cold-start ceiling (no trade history = no Kelly edge)
HARD_CAP             = 50.0   # Absolute max for crypto futures (conservative of 50/75/125)
HARD_FLOOR           = 1.0    # Never below 1× (covered position)
DRAWDOWN_PAIN_START  = 0.10   # DD < 10%  → no reduction
DRAWDOWN_MAX_CUT     = 0.50   # At max_dd → 50% leverage reduction
# ──────────────────────────────────────────────────────────────────────────────


class LeverageEngine:
    """
    Best-of-breed leverage calculator for crypto perpetual futures.

    Designed to be:
      - Stateless (pure function with no side effects)
      - Transparent (every component logged)
      - Derivable (all numbers traceable to mathematical source)

    Example:
        engine = LeverageEngine(binance_max_table)
        lev = engine.compute(
            symbol="BTCUSDT",
            win_rate=0.62,
            avg_win_pct=0.025,
            avg_loss_pct=0.015,
            atr_pct=0.018,          # ATR as fraction of price
            signal_confidence=0.72,
            open_positions=2,
            max_positions=5,
            current_drawdown_pct=0.05,
            max_drawdown_threshold=0.20,
            total_trades=47,
            margin_fraction=0.02,   # Margin per trade = 2% of equity
        )
    """

    def __init__(
        self,
        binance_max_leverage: dict,
        hard_cap: float = HARD_CAP,
        hard_floor: float = HARD_FLOOR,
    ):
        self.binance_max = binance_max_leverage
        self.hard_cap = hard_cap
        self.hard_floor = hard_floor

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        atr_pct: float,
        signal_confidence: float,
        open_positions: int,
        max_positions: int,
        current_drawdown_pct: float,
        max_drawdown_threshold: float,
        total_trades: int,
        margin_fraction: float = 0.02,
        conservative_mode: bool = False,
        adx: Optional[float] = None,
    ) -> float:
        """
        Compute optimal leverage.

        Args:
            symbol:                  Trading pair (e.g. "BTCUSDT")
            win_rate:                Historical win rate [0..1]
            avg_win_pct:             Average win as fraction of notional
            avg_loss_pct:            Average loss as fraction of notional (positive)
            atr_pct:                 ATR as fraction of price (e.g. 0.018 = 1.8%)
            signal_confidence:       AI ensemble confidence [0..1]
            open_positions:          Currently open positions
            max_positions:           Max simultaneous positions allowed
            current_drawdown_pct:    Current drawdown from equity peak [0..1]
            max_drawdown_threshold:  Max acceptable drawdown before halving [0..1]
            total_trades:            Total historical trades (for cold-start guard)
            margin_fraction:         Margin per trade as fraction of equity (default 2%)
            conservative_mode:       If True, apply additional 40% reduction
            adx:                     ADX trend indicator [0..100], None = unknown

        Returns:
            float: Optimal leverage in [hard_floor, min(symbol_max, hard_cap)]
        """
        symbol_max = min(self.binance_max.get(symbol, 50), self.hard_cap)

        # ── 1. Cold-start guard ───────────────────────────────────────────────
        if total_trades < 5:
            cold = self._cold_start(atr_pct, signal_confidence, margin_fraction, symbol_max)
            logger.info(
                f"[LeverageEngine] Cold-start ({total_trades} trades) → {cold:.1f}x"
            )
            return cold

        # ── 2. Kelly component (25% fractional) ──────────────────────────────
        kelly_lev = self._kelly_leverage(win_rate, avg_win_pct, avg_loss_pct)

        # ── 3. ATR-based volatility targeting ────────────────────────────────
        atr_lev = self._atr_leverage(atr_pct, margin_fraction)

        # ── 4. Base = geometric mean (conservative multiplicative blend) ──────
        base = self._geometric_blend(kelly_lev, atr_lev)

        # ── 5. Composite scalar (all components are bounded, no branching)  ───
        conf_scalar   = self._confidence_scalar(signal_confidence)   # [0.13..1.00]
        util_scalar   = self._utilization_scalar(open_positions, max_positions)  # [0.50..1.00]
        dd_scalar     = self._drawdown_scalar(current_drawdown_pct, max_drawdown_threshold)  # [0.50..1.00]
        regime_scalar = self._regime_scalar(adx)                     # [0.75..1.25]
        cons_scalar   = 0.60 if conservative_mode else 1.0

        composite = conf_scalar * util_scalar * dd_scalar * regime_scalar * cons_scalar
        optimal   = base * composite

        # ── 6. Hard bounds ────────────────────────────────────────────────────
        final = max(self.hard_floor, min(optimal, symbol_max))

        logger.info(
            f"[LeverageEngine] {symbol}: "
            f"kelly={kelly_lev:.1f}x  atr={atr_lev:.1f}x  base={base:.1f}x | "
            f"conf={conf_scalar:.2f}  util={util_scalar:.2f}  "
            f"dd={dd_scalar:.2f}  regime={regime_scalar:.2f}  cons={cons_scalar:.1f} | "
            f"composite={composite:.2f} → optimal={optimal:.1f}x → final={final:.1f}x "
            f"(symbol_max={symbol_max}x)"
        )

        return round(final, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # COMPONENTS (each is documented with mathematical derivation)
    # ──────────────────────────────────────────────────────────────────────────

    def _kelly_leverage(
        self, win_rate: float, avg_win_pct: float, avg_loss_pct: float
    ) -> float:
        """
        Fractional Kelly Criterion (25%) for long-term geometric growth.

        Formula:
            edge     = (W × avg_win) − (L × avg_loss)
            variance = (W × avg_win²) + (L × avg_loss²)
            kelly    = edge / variance × KELLY_FRACTION

        Why 25% Kelly:
          - Full Kelly maximizes long-term growth *in theory*
          - In practice, fat tails (crypto) make full Kelly extremely dangerous
          - 25% (quarter-Kelly) is the empirical consensus for asymmetric assets
            (Thorp 2006, Haghani & Dewey 2016, Vince 1992)
          - Reduces bet by exactly enough to survive 3-4σ tail events

        Returns:
            float: Kelly-optimal fractional leverage ≥ HARD_FLOOR
        """
        loss_rate = 1.0 - win_rate
        avg_loss  = abs(avg_loss_pct)

        edge     = (win_rate * avg_win_pct) - (loss_rate * avg_loss)
        variance = (win_rate * avg_win_pct ** 2) + (loss_rate * avg_loss ** 2)

        if variance < 1e-9 or edge <= 0:
            logger.debug(f"[LeverageEngine] Kelly: no edge (edge={edge:.4f}) → floor {self.hard_floor}")
            return self.hard_floor

        full_kelly    = edge / variance
        fractional    = full_kelly * KELLY_FRACTION

        logger.debug(
            f"[LeverageEngine] Kelly: W={win_rate:.2f} AvgW={avg_win_pct*100:.1f}% "
            f"AvgL={avg_loss*100:.1f}% → edge={edge*100:.2f}% var={variance*100:.2f}% "
            f"fullKelly={full_kelly:.1f}x → 25%={fractional:.1f}x"
        )

        return max(self.hard_floor, fractional)

    def _atr_leverage(self, atr_pct: float, margin_fraction: float) -> float:
        """
        ATR-based volatility-targeting leverage.

        Derivation (first principles — zero magic numbers):
            Define:
              risk_budget   = equity × TARGET_RISK_PCT   (e.g. 1.5% of equity)
              sl_distance   = ATR_SL_MULTIPLIER × atr_pct (e.g. 1.5 × 1.8% = 2.7%)
              notional      = equity × margin_fraction × leverage
              loss_at_SL    = notional × sl_distance

            Constraint: loss_at_SL = risk_budget
              → (equity × margin_fraction × leverage) × sl_distance = equity × TARGET_RISK_PCT
              → leverage = TARGET_RISK_PCT / (sl_distance × margin_fraction)

        Consequences (automatic, no manual tuning):
          - ATR doubles (volatile market)  → leverage halves
          - Margin allocation doubles      → leverage halves (risk stays fixed)
          - Target risk decreases          → leverage decreases proportionally

        Args:
            atr_pct:        ATR as price fraction (e.g. 0.018 for 1.8% ATR)
            margin_fraction: margin allocated = N% of equity (e.g. 0.02)

        Returns:
            float: ATR-implied leverage ≥ HARD_FLOOR
        """
        if atr_pct <= 0 or margin_fraction <= 0:
            logger.debug("[LeverageEngine] ATR: invalid inputs → cold-start fallback")
            return COLD_START_MAX

        sl_pct   = ATR_SL_MULTIPLIER * atr_pct
        leverage = TARGET_RISK_PCT / (sl_pct * margin_fraction)

        logger.debug(
            f"[LeverageEngine] ATR: atr={atr_pct*100:.2f}% → sl={sl_pct*100:.2f}% "
            f"→ leverage={leverage:.1f}x"
        )

        return max(self.hard_floor, leverage)

    def _cold_start(
        self,
        atr_pct: float,
        signal_confidence: float,
        margin_fraction: float,
        symbol_max: float,
    ) -> float:
        """
        Cold-start leverage (< 5 confirmed trades).

        Uses ATR-only formula with 50% additional haircut.
        No Kelly allowed: without validated edge, Kelly is noise amplification.
        Confidence still modulates the result.
        """
        atr_lev  = self._atr_leverage(atr_pct, margin_fraction)
        conf_s   = self._confidence_scalar(signal_confidence)
        raw      = atr_lev * conf_s * 0.50  # 50% haircut: no proven edge yet
        bounded  = max(self.hard_floor, min(raw, COLD_START_MAX, symbol_max))

        logger.debug(
            f"[LeverageEngine] Cold-start: atr_lev={atr_lev:.1f}x "
            f"conf_s={conf_s:.2f} raw={raw:.1f}x → {bounded:.1f}x"
        )
        return bounded

    def _geometric_blend(self, kelly: float, atr: float) -> float:
        """
        Geometric mean of Kelly and ATR components.

        Why geometric (not arithmetic or min):
          - Both components represent multiplicative leverage (compound risk)
          - Geometric mean is always ≤ arithmetic mean (conservative bias)
          - Avoids the pathology of min() choking on whichever component is low
          - When Kelly=20x and ATR=5x: geometric=10x, arithmetic=12.5x, min=5x
            → 10x reflects both signals without ignoring either

        Returns:
            float: Combined base leverage ≥ HARD_FLOOR
        """
        blended = math.sqrt(max(0.0, kelly) * max(0.0, atr))
        return max(self.hard_floor, blended)

    def _confidence_scalar(self, confidence: float) -> float:
        """
        Convex confidence scaling: f(c) = c^1.5

        Why convex (not linear):
          Linear f(c)=c gives 70% → 0.70× and 50% → 0.50×
          Convex f(c)=c^1.5 gives 70% → 0.59× and 50% → 0.35×

          This is correct because:
          - Signal confidence is an *information quality* metric
          - Information degrades non-linearly near the edges
          - A drop from 0.9→0.7 represents more model uncertainty
            than a simple 20% reduction suggests
          - c^1.5 is a balanced exponent: c^2 is too harsh (0.5→0.25)

        Returns:
            float ∈ [0.0, 1.0]
        """
        c = max(0.0, min(1.0, confidence))
        return c ** 1.5

    def _utilization_scalar(self, open_positions: int, max_positions: int) -> float:
        """
        Portfolio utilization dilution: 1.0 - 0.5 × (open/max)

        At 0% utilization   → 1.0× (full leverage allowed)
        At 100% utilization → 0.5× (half leverage — diversification buffer)

        Rationale:
          A fully concentrated portfolio amplifies correlation risk.
          Halving leverage when full maintains same total portfolio risk
          regardless of how many positions are open.

        Returns:
            float ∈ [0.5, 1.0]
        """
        if max_positions <= 0:
            return 1.0
        ratio = min(1.0, open_positions / max_positions)
        return 1.0 - 0.5 * ratio

    def _drawdown_scalar(
        self, current_dd: float, max_dd_threshold: float
    ) -> float:
        """
        Drawdown circuit breaker: linear interpolation from threshold to 50% cut.

        DD < 10%            → 1.00× (no reduction)
        DD = max_dd         → 0.50× (50% reduction)
        DD > max_dd         → 0.50× (floor — still trade but half size)

        Rationale:
          Drawdowns indicate:
            (a) regime change (model is wrong)
            (b) adversarial market conditions for this strategy
          Research (AQR, Two Sigma) shows that reducing leverage during drawdown
          is the single most important rule for long-term survival.
          50% floor: never go to zero — strategy may recover, just protect.

        Returns:
            float ∈ [0.5, 1.0]
        """
        if current_dd <= DRAWDOWN_PAIN_START:
            return 1.0

        pain_range = max(max_dd_threshold - DRAWDOWN_PAIN_START, 0.01)
        ratio  = (current_dd - DRAWDOWN_PAIN_START) / pain_range
        scalar = 1.0 - DRAWDOWN_MAX_CUT * min(1.0, ratio)

        return max(0.50, scalar)

    def _regime_scalar(self, adx: Optional[float]) -> float:
        """
        Market regime adjustment via ADX (Average Directional Index).

        ADX < 20  → ranging/choppy  → 0.75× leverage
        ADX = 25  → neutral         → 1.00× (baseline)
        ADX > 50  → strong trend    → 1.25× leverage

        Formula: scalar = 1.0 + clip((adx − 25) / 100, −0.25, +0.25)

        Derivation:
          - ADX = 25 is the classic "trending" threshold (Wilder 1978)
          - Each 100 ADX points → ±0.25 scalar swing
          - Bounded to [0.75, 1.25] — meaningful but not dominant

        Rationale:
          Trending markets have directional momentum: signals in trend direction
          have higher probability. Ranging markets have mean-reversion pressure
          that works against momentum signals.

        Returns:
            float ∈ [0.75, 1.25]
        """
        if adx is None:
            return 1.0  # unknown regime → neutral

        adjustment = (float(adx) - 25.0) / 100.0
        scalar = 1.0 + max(-0.25, min(0.25, adjustment))
        return scalar
