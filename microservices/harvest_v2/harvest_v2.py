#!/usr/bin/env python3
"""
harvest_v2.py — Exit Engine V2

Modes:
  SHADOW (default): stream_live="" — emits only to harvest.v2.shadow for validation
  LIVE:             stream_live set in Redis quantum:config:harvest_v2 — dual-writes to
                    harvest.v2.shadow AND quantum:stream:apply.plan (live execution)

SAFETY RULES:
  - ATR == 0 → skip
  - Heat missing → 0.0
  - HOLD_SUPPRESSED → never emit
  - Live switch is CONFIG-DRIVEN (Redis) — instant rollback via HDEL stream_live
"""

import json
import os
import sys
import time
import logging
import signal
import uuid

# ── Path fix so sub-packages resolve ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.redis_client import RedisClient
from engine.config import ConfigLoader
from engine.state import StateManager
from engine.evaluator import ExitEvaluator
from engine.metrics import MetricsWriter
from feeds.position_provider import PositionProvider
from feeds.heat_provider import HeatProvider
from feeds.atr_provider import ATRProvider

# ── Logging ───────────────────────────────────────────────────────────────
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("hv2")

# ── Graceful shutdown ─────────────────────────────────────────────────────
_RUNNING = True


def _handle_signal(sig, frame):
    global _RUNNING
    logger.info("[HV2] Signal %s received — shutting down", sig)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)


# ── Live payload builder (apply.plan format) ─────────────────────────────

# Decision → (action, step_name, close_pct_of_current)
_DECISION_MAP = {
    "PARTIAL_25":  ("PARTIAL_CLOSE_PROPOSED", "PARTIAL_25",  25.0),
    "PARTIAL_50":  ("PARTIAL_CLOSE_PROPOSED", "PARTIAL_50",  25.0),  # incremental stage
    "PARTIAL_75":  ("PARTIAL_CLOSE_PROPOSED", "PARTIAL_75",  25.0),  # incremental stage
    "EXIT":        ("FULL_CLOSE_PROPOSED",    "CLOSE_FULL",  100.0),
    "FULL_CLOSE":  ("FULL_CLOSE_PROPOSED",    "CLOSE_FULL",  100.0),
}


def _build_live_payload(pos, result, decision: str) -> dict:
    """Build apply.plan-compatible message for live execution."""
    action, step_name, pct = _DECISION_MAP.get(
        decision, ("FULL_CLOSE_PROPOSED", "CLOSE_FULL", 100.0)
    )
    close_qty  = round(pos.quantity * pct / 100.0, 8)
    close_side = "SELL" if pos.side.upper() in ("LONG", "BUY") else "BUY"
    steps      = json.dumps([{
        "step":  step_name,
        "type":  "market_reduce_only",
        "side":  "close",
        "pct":   pct,
    }])
    return {
        "plan_id":       uuid.uuid4().hex[:16],
        "symbol":        pos.symbol,
        "side":          close_side,
        "action":        action,
        "decision":      "EXECUTE",
        "R_net":         f"{result.R_net:.4f}",
        "reason_codes":  result.emit_reason,
        "steps":         steps,
        "close_qty":     str(close_qty),
        "qty":           str(pos.quantity),
        "price":         "",
        "reduceOnly":    "true",
        "kill_score":    "0.0",
        "source":        "harvest_v2",
        "timestamp":     str(int(time.time())),
    }


# ── Shadow payload builder ─────────────────────────────────────────────────

def _build_shadow_payload(
    pos, result, heat: float, partial_stage: int, max_R_seen
) -> dict:
    """All mandatory fields per Phase 0 spec §6."""
    return {
        "symbol":         pos.symbol,
        "side":           pos.side,
        "R_net":          f"{result.R_net:.4f}",
        "R_stop":         f"{result.R_stop:.4f}",
        "R_target":       f"{result.R_target:.4f}",
        "regime":         result.regime,
        "heat":           f"{heat:.4f}",
        "decision":       result.decision,
        "partial_stage":  str(partial_stage),
        "max_R_seen":     f"{max_R_seen:.4f}" if max_R_seen is not None else "",
        "initial_risk":   f"{pos.entry_risk_usdt:.4f}",
        "unrealized_pnl": f"{pos.unrealized_pnl:.4f}",
        "atr_value":      f"{pos.atr_value:.6f}",
        "vol_factor":     f"{result.vol_factor:.4f}",
        "emit_reason":    result.emit_reason,
        "timestamp":      f"{time.time():.3f}",
        "v2_version":     "2.0.0",
    }


# ── Main loop ─────────────────────────────────────────────────────────────

def main():
    global _RUNNING

    redis          = RedisClient()
    cfg_loader     = ConfigLoader(redis)
    evaluator      = ExitEvaluator()
    heat_provider  = HeatProvider(redis)
    atr_provider   = ATRProvider()
    metrics        = MetricsWriter(redis, cfg_loader.get().metrics_key)

    # State manager initialized with config's atr_window
    cfg = cfg_loader.get()
    state_mgr      = StateManager(redis, atr_window=cfg.atr_window)
    pos_provider   = PositionProvider(redis, max_age_sec=cfg.max_position_age_sec)

    live_mode = bool(cfg.stream_live)
    logger.info(
        "[HV2] HARVEST_V2 STARTING — mode=%s stream_live=%r",
        "LIVE" if live_mode else "SHADOW",
        cfg.stream_live or "(none)",
    )

    metrics.record_start()
    logger.info("[HV2] Startup complete — entering scan loop")

    while _RUNNING:
        tick_start = time.monotonic()

        # --- Reload config (respects refresh interval internally) ----------
        cfg = cfg_loader.get()
        state_mgr.set_atr_window(cfg.atr_window)
        pos_provider.max_age_sec = cfg.max_position_age_sec

        # --- Fetch inputs ---------------------------------------------------
        fetch            = pos_provider.fetch_positions()
        positions        = fetch.positions
        heat             = heat_provider.get_heat()

        # --- Tick accumulators ---------------------------------------------
        n_scanned        = fetch.total_keys
        n_evaluated      = 0
        n_emitted        = 0
        n_skipped_invalid= fetch.skipped_invalid
        n_skip_stale     = fetch.skipped_stale
        n_hold_suppressed= 0

        # --- Per-position evaluation ----------------------------------------
        for pos in positions:
            atr_val = atr_provider.get_atr(pos)
            if atr_val is None:
                n_skipped_invalid += 1
                continue

            n_evaluated += 1
            state = state_mgr.get(pos)

            decision, result = evaluator.evaluate(
                pos_unrealized_pnl  = pos.unrealized_pnl,
                pos_entry_risk_usdt = pos.entry_risk_usdt,
                state               = state,
                heat                = heat,
                cfg                 = cfg,
            )

            # Trailing max update (after pure eval to keep evaluator stateless)
            state.update_max_R(result.R_net)

            # Log per-symbol
            logger.info(
                "[HV2] %s R=%.3f regime=%s stop=%.3f target=%.3f "
                "decision=%s emit=%s reason=%s",
                pos.symbol, result.R_net, result.regime,
                result.R_stop, result.R_target,
                decision,
                "true" if decision not in ("HOLD_SUPPRESSED",) else "false",
                result.emit_reason,
            )

            if decision == "HOLD_SUPPRESSED":
                n_hold_suppressed += 1
                continue

            # --- Update partial_stage BEFORE building payloads ----------------
            if decision == "PARTIAL_25":
                state.partial_stage = max(state.partial_stage, 1)
            elif decision == "PARTIAL_50":
                state.partial_stage = max(state.partial_stage, 2)
            elif decision == "PARTIAL_75":
                state.partial_stage = max(state.partial_stage, 3)

            # --- Always emit to shadow stream (audit trail) -------------------
            shadow_payload = _build_shadow_payload(
                pos, result, heat, state.partial_stage, state.max_R_seen
            )
            redis.xadd(cfg.stream_shadow, shadow_payload)

            # --- Emit to live apply.plan stream if configured ----------------
            if cfg.stream_live:
                live_payload = _build_live_payload(pos, result, decision)
                redis.xadd(cfg.stream_live, live_payload)
                logger.warning(
                    "[HV2_LIVE] EMITTED symbol=%s decision=%s R=%.3f qty=%s → %s",
                    pos.symbol, decision, result.R_net,
                    live_payload["close_qty"], cfg.stream_live,
                )

            n_emitted += 1

            # Persist state + record emission
            state.record_emission(decision, result.R_net)
            state_mgr.save(state)

            metrics.emission(decision, result.regime, result.R_net)

        # --- Tick metrics ---------------------------------------------------
        metrics.tick(
            scanned         = n_scanned,
            evaluated       = n_evaluated,
            emitted         = n_emitted,
            skipped_invalid = n_skipped_invalid,
            hold_suppressed = n_hold_suppressed,
        )

        logger.info(
            "[HV2_TICK] ts=%.3f scanned=%d evaluated=%d emitted=%d "
            "skipped_invalid=%d skipped_stale=%d hold_suppressed=%d heat=%.4f",
            time.time(), n_scanned, n_evaluated, n_emitted,
            n_skipped_invalid, n_skip_stale, n_hold_suppressed, heat,
        )

        # --- Sleep remainder of interval ------------------------------------
        elapsed = time.monotonic() - tick_start
        sleep_for = max(0.0, cfg.scan_interval_sec - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    logger.info("[HV2] HARVEST_V2 STOPPED")


if __name__ == "__main__":
    main()
