"""replay_metrics: aggregation engine for exit.replay JSONL records.

PATCH-9 — replay consumer / evaluation engine.

This module is the pure computation core.  It takes a sequence of
ReplayRecord objects (already parsed by replay_schema) and produces typed
aggregation objects that the analysis scripts and DPO exporter can use.

No I/O is performed here — callers handle file loading.  All functions are
pure / side-effect-free and are safe to call from tests without any
filesystem or network access.

Key metrics computed
--------------------
- Divergence rate        Qwen3 vs formula disagreement fraction
- Reward statistics      mean / min / max / std per slice (action, symbol, side)
- Regret distribution    label counts + weighted severity
- Preferred-action freq  how often preferred != live_action (missed signal)
- Win/loss split         reward > 0 = win, < 0 = loss, == 0 = neutral
- Hold-duration band     <5m / 5-60m / 1-4h / >4h
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from ops.offline.replay_schema import ReplayRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(vals: list[float]) -> Optional[float]:
    return statistics.mean(vals) if vals else None


def _stdev(vals: list[float]) -> Optional[float]:
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def _bucket_hold(secs: Optional[int]) -> str:
    """Classify hold duration into a human-readable band."""
    if secs is None:
        return "unknown"
    if secs < 300:
        return "<5m"
    if secs < 3600:
        return "5m-1h"
    if secs < 14400:
        return "1h-4h"
    return ">4h"


def load_jsonl(path) -> list[ReplayRecord]:
    """
    Load all ReplayRecord objects from a single JSONL file path.

    Each line is a JSON object with the same field set as ReplayRecord.
    ``stream_id`` must be present; missing fields fall back to defaults via
    ``from_redis_entry``.

    Never raises on individual malformed lines — they are skipped with a
    count returned separately via the second element.
    """
    import json
    records: list[ReplayRecord] = []
    skipped = 0
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.pop("stream_id", f"jsonl-line-{lineno}")
                rec = ReplayRecord.from_redis_entry(sid, obj)
                records.append(rec)
            except Exception:
                skipped += 1
    return records, skipped


def load_jsonl_dir(dir_path) -> tuple[list[ReplayRecord], int]:
    """
    Load all JSONL files matching ``replay_*.jsonl`` in *dir_path*.

    Returns (records, total_skipped).
    """
    import glob
    import os
    pattern = os.path.join(str(dir_path), "replay_*.jsonl")
    all_records: list[ReplayRecord] = []
    total_skipped = 0
    for path in sorted(glob.glob(pattern)):
        recs, skipped = load_jsonl(path)
        all_records.extend(recs)
        total_skipped += skipped
    return all_records, total_skipped


# ── Sub-aggregates ─────────────────────────────────────────────────────────────

@dataclass
class RewardStats:
    """Descriptive statistics for a collection of reward scalars."""
    count: int
    mean: Optional[float]
    stdev: Optional[float]
    min: Optional[float]
    max: Optional[float]
    win_count: int      # reward > 0
    loss_count: int     # reward < 0
    neutral_count: int  # reward == 0 (or None)

    @property
    def win_rate(self) -> float:
        return _pct(self.win_count, self.count)

    @property
    def loss_rate(self) -> float:
        return _pct(self.loss_count, self.count)

    @classmethod
    def from_values(cls, vals: list[Optional[float]]) -> "RewardStats":
        concrete = [v for v in vals if v is not None]
        wins  = sum(1 for v in concrete if v > 0)
        losses = sum(1 for v in concrete if v < 0)
        neutrals = len(vals) - wins - losses
        return cls(
            count=len(vals),
            mean=round(_mean(concrete), 6) if concrete else None,
            stdev=round(_stdev(concrete), 6) if concrete and len(concrete) >= 2 else None,
            min=round(min(concrete), 6) if concrete else None,
            max=round(max(concrete), 6) if concrete else None,
            win_count=wins,
            loss_count=losses,
            neutral_count=neutrals,
        )


@dataclass
class RegretDistribution:
    """Counts and weighted severity for each regret label."""
    none: int = 0
    late_hold: int = 0
    premature_close: int = 0
    divergence_regret: int = 0
    total: int = 0
    # Weighted regret: sum(regret_score) for non-none records
    weighted_severity: float = 0.0

    @property
    def actionable_rate(self) -> float:
        """Fraction of records with any non-none regret label."""
        return _pct(self.total - self.none, self.total)

    def to_dict(self) -> dict:
        return {
            "none": self.none,
            "late_hold": self.late_hold,
            "premature_close": self.premature_close,
            "divergence_regret": self.divergence_regret,
            "total": self.total,
            "actionable_rate_pct": self.actionable_rate,
            "weighted_severity": round(self.weighted_severity, 4),
        }


# ── Main metrics aggregate ──────────────────────────────────────────────────────

@dataclass
class ReplayMetrics:
    """
    Full aggregation result for a batch of ReplayRecord objects.

    All nested dicts use string keys (action name, symbol, side, etc.) for
    easy JSON serialisation.
    """
    # ── Record counts ──────────────────────────────────────────────────────────
    total_records: int = 0
    records_with_reward: int = 0

    # ── Divergence ────────────────────────────────────────────────────────────
    divergence_count: int = 0          # records where diverged == True
    divergence_rate_pct: float = 0.0

    # ── Qwen3 vs formula agreement ────────────────────────────────────────────
    qwen3_agrees_formula: int = 0
    qwen3_agrees_live: int = 0
    qwen3_overrode_formula: int = 0    # qwen3 != formula (when both available)
    qwen3_available: int = 0           # records where qwen3_action != "null"

    # ── Reward by action (live_action key) ────────────────────────────────────
    reward_by_action: dict[str, RewardStats] = field(default_factory=dict)

    # ── Reward by symbol ──────────────────────────────────────────────────────
    reward_by_symbol: dict[str, RewardStats] = field(default_factory=dict)

    # ── Reward by side (LONG / SHORT) ─────────────────────────────────────────
    reward_by_side: dict[str, RewardStats] = field(default_factory=dict)

    # ── Regret distribution ───────────────────────────────────────────────────
    regret: RegretDistribution = field(default_factory=RegretDistribution)

    # ── Preferred-action analysis ─────────────────────────────────────────────
    preferred_matches_live: int = 0    # preferred_action == live_action
    preferred_differs: int = 0         # missed signal: agent should have done differently
    preferred_action_freq: dict[str, int] = field(default_factory=dict)

    # ── Hold-duration bands ───────────────────────────────────────────────────
    hold_duration_bands: dict[str, int] = field(default_factory=dict)

    # ── Overall reward stats ──────────────────────────────────────────────────
    overall_reward: Optional[RewardStats] = None

    # ── Prompt-improvement signals ────────────────────────────────────────────
    # Synthesised from the aggregation; consumed by analyze_replay.py
    signals: list[str] = field(default_factory=list)


# ── Aggregation ────────────────────────────────────────────────────────────────

def aggregate(records: Sequence[ReplayRecord]) -> ReplayMetrics:
    """
    Compute ReplayMetrics from a sequence of ReplayRecord objects.

    Pure function — no I/O.  Safe to call from tests with synthetic data.
    """
    m = ReplayMetrics()
    m.total_records = len(records)
    if not records:
        return m

    # Accumulators for reward grouping
    by_action: dict[str, list] = {}
    by_symbol: dict[str, list] = {}
    by_side: dict[str, list] = {}
    all_rewards: list[Optional[float]] = []

    for rec in records:
        r = rec.reward  # may be None

        # ── reward availability ──────────────────────────────────────────────
        if r is not None:
            m.records_with_reward += 1
        all_rewards.append(r)

        # ── divergence ──────────────────────────────────────────────────────
        if rec.diverged:
            m.divergence_count += 1

        # ── Qwen3 analysis ───────────────────────────────────────────────────
        if rec.qwen3_action not in ("null", "", "UNKNOWN", None):
            m.qwen3_available += 1
            if rec.qwen3_action == rec.formula_action:
                m.qwen3_agrees_formula += 1
            else:
                m.qwen3_overrode_formula += 1
            if rec.qwen3_action == rec.live_action:
                m.qwen3_agrees_live += 1

        # ── reward by action ─────────────────────────────────────────────────
        action = rec.live_action or "UNKNOWN"
        by_action.setdefault(action, []).append(r)

        # ── reward by symbol ─────────────────────────────────────────────────
        sym = rec.symbol or "UNKNOWN"
        by_symbol.setdefault(sym, []).append(r)

        # ── reward by side ───────────────────────────────────────────────────
        side = rec.side or "UNKNOWN"
        by_side.setdefault(side, []).append(r)

        # ── regret ───────────────────────────────────────────────────────────
        m.regret.total += 1
        label = rec.regret_label or "none"
        if label == "late_hold":
            m.regret.late_hold += 1
        elif label == "premature_close":
            m.regret.premature_close += 1
        elif label == "divergence_regret":
            m.regret.divergence_regret += 1
        else:
            m.regret.none += 1
        if label != "none" and rec.regret_score is not None:
            m.regret.weighted_severity += rec.regret_score

        # ── preferred action ─────────────────────────────────────────────────
        pref = rec.preferred_action or "HOLD"
        m.preferred_action_freq[pref] = m.preferred_action_freq.get(pref, 0) + 1
        if pref == rec.live_action:
            m.preferred_matches_live += 1
        else:
            m.preferred_differs += 1

        # ── hold-duration band ───────────────────────────────────────────────
        band = _bucket_hold(rec.hold_duration_sec)
        m.hold_duration_bands[band] = m.hold_duration_bands.get(band, 0) + 1

    # ── Build RewardStats ─────────────────────────────────────────────────────
    m.overall_reward = RewardStats.from_values(all_rewards)
    m.reward_by_action = {k: RewardStats.from_values(v) for k, v in by_action.items()}
    m.reward_by_symbol = {k: RewardStats.from_values(v) for k, v in by_symbol.items()}
    m.reward_by_side   = {k: RewardStats.from_values(v) for k, v in by_side.items()}

    # ── Divergence rate ───────────────────────────────────────────────────────
    m.divergence_rate_pct = _pct(m.divergence_count, m.total_records)

    # ── Prompt-improvement signals ────────────────────────────────────────────
    m.signals = _derive_signals(m)

    return m


# ── Signal derivation ──────────────────────────────────────────────────────────

_MIN_SAMPLES = 5          # minimum records in a slice before issuing a signal
_HIGH_REGRET_THRESHOLD = 0.30    # actionable regret rate %
_DIVERGENCE_SIGNAL_THRESHOLD = 25.0  # divergence rate %
_HOLD_NEGATIVE_MEAN = -0.2       # mean reward for HOLD below this → signal


def _derive_signals(m: ReplayMetrics) -> list[str]:
    """
    Generate plain-text action signals from aggregated metrics.

    These are advisory — they highlight patterns that warrant human review,
    prompt updates, or policy changes.
    """
    sigs: list[str] = []

    # Divergence rate
    if m.divergence_rate_pct >= _DIVERGENCE_SIGNAL_THRESHOLD:
        sigs.append(
            f"HIGH DIVERGENCE: Qwen3 and formula disagreed on "
            f"{m.divergence_rate_pct:.1f}% of decisions "
            f"({m.divergence_count}/{m.total_records}). "
            "Review prompt alignment with formula rules."
        )

    # HOLD regret
    hold_stats = m.reward_by_action.get("HOLD")
    if hold_stats and hold_stats.count >= _MIN_SAMPLES and hold_stats.mean is not None:
        if hold_stats.mean <= _HOLD_NEGATIVE_MEAN:
            sigs.append(
                f"HOLD DRAG: mean reward for HOLD = {hold_stats.mean:.4f} "
                f"(win_rate={hold_stats.win_rate:.1f}%). "
                "Agent is holding too long — consider tightening FULL_CLOSE triggers in prompt."
            )

    # Premature close rate
    total = m.regret.total
    if total >= _MIN_SAMPLES and m.regret.premature_close / max(total, 1) >= 0.20:
        sigs.append(
            f"PREMATURE CLOSE: {m.regret.premature_close}/{total} records "
            f"({_pct(m.regret.premature_close, total):.1f}%) labelled premature_close. "
            "Agent may be exiting too early — review hold_duration thresholds."
        )

    # Late hold rate
    if total >= _MIN_SAMPLES and m.regret.late_hold / max(total, 1) >= 0.20:
        sigs.append(
            f"LATE HOLD: {m.regret.late_hold}/{total} records "
            f"({_pct(m.regret.late_hold, total):.1f}%) labelled late_hold. "
            "Agent missed exits — consider lowering exit_score threshold in prompt."
        )

    # preferred != live at high rate
    if m.total_records >= _MIN_SAMPLES:
        missed_pct = _pct(m.preferred_differs, m.total_records)
        if missed_pct >= 40.0:
            sigs.append(
                f"MISSED SIGNALS: preferred_action differed from live_action in "
                f"{missed_pct:.1f}% of records "
                f"({m.preferred_differs}/{m.total_records}). "
                "Large gap between what the agent did and what it should have done."
            )

    # Per-symbol losers
    for sym, rs in m.reward_by_symbol.items():
        if rs.count >= _MIN_SAMPLES and rs.mean is not None and rs.mean < -0.3:
            sigs.append(
                f"SYMBOL LOSS: {sym} mean reward = {rs.mean:.4f} over {rs.count} records. "
                "Consider reviewing exit parameters for this symbol."
            )

    # Qwen3 override rate
    if m.qwen3_available >= _MIN_SAMPLES:
        override_pct = _pct(m.qwen3_overrode_formula, m.qwen3_available)
        if override_pct >= 40.0:
            sigs.append(
                f"QWEN3 OVERRIDE: Qwen3 disagreed with formula in "
                f"{override_pct:.1f}% of decisions where Qwen3 was active. "
                "High override rate without better rewards suggests prompt miscalibration."
            )

    if not sigs:
        sigs.append("No significant negative patterns detected in this batch.")

    return sigs
