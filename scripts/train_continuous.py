import os
import time
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import List

# Add paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Reuse existing training harness
from ai_engine.train_and_save import train_and_save

# Trading universe
from config.trading_universe import (
    LAYER1_COINS,
    LAYER2_COINS,
    DEFAULT_UNIVERSE,
    ALL_COINS,
    apply_quote,
)
from config.config import DEFAULT_QUOTE
from backend.utils.universe import get_l1l2_top_by_volume, get_l1l2_top_by_volume_multi

logging.basicConfig(
    level=os.getenv("QT_LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("train_continuous")


def select_symbols() -> List[str]:
    # Use the same logic as backend for consistency
    universe_profile = os.getenv("QT_UNIVERSE", "l1l2-top")
    max_symbols = int(os.getenv("QT_MAX_SYMBOLS", "100"))
    quotes_env = os.getenv("QT_QUOTES", "USDC,USDT").strip()
    quotes = [q.strip().upper() for q in quotes_env.split(",") if q.strip()]

    try:
        if universe_profile.lower() in ("l1l2-top", "l1l2volume", "l1l2-vol"):
            symbols = get_l1l2_top_by_volume_multi(quotes=quotes, max_symbols=max_symbols)
            logger.info(
                "Selected %d L1/L2+base pairs by 24h volume (quotes=%s)",
                len(symbols), ",".join(quotes)
            )
        else:
            from config.trading_universe import get_trading_universe
            symbols = get_trading_universe(universe_profile)
            # Apply max symbols limit if set
            if max_symbols > 0 and len(symbols) > max_symbols:
                symbols = symbols[:max_symbols]
                logger.info(f"Limited to {max_symbols} symbols (QT_MAX_SYMBOLS)")
    except Exception as e:
        logger.warning(
            f"Failed to load trading universe '{universe_profile}': {e}, using defaults"
        )
        symbols = ["BTCUSDC", "ETHUSDC", "BTCUSDT", "ETHUSDT"]

    return symbols


def run_continuous_training(
    duration_seconds: int,
    interval_seconds: int,
    limit: int,
):
    start_time = datetime.now(timezone.utc)
    end_time = start_time + timedelta(seconds=duration_seconds)

    symbols = select_symbols()
    logger.info(
        "Continuous training started | symbols=%d interval=%ds duration=%ds",
        len(symbols), interval_seconds, duration_seconds,
    )
    logger.info("Training symbols: %s", ", ".join(symbols))

    cycle = 0
    while datetime.now(timezone.utc) < end_time:
        cycle += 1
        cycle_start = datetime.now(timezone.utc)
        logger.info("\n===== Training cycle %d started (%sZ) =====", cycle, cycle_start.strftime('%Y-%m-%dT%H:%M:%S'))
        try:
            # Train and overwrite artifacts to ai_engine/models/
            train_and_save(symbols=symbols, limit=limit)
            logger.info("Cycle %d completed successfully", cycle)
        except Exception as e:
            logger.exception("Cycle %d failed: %s", cycle, e)

        # Sleep until next cycle respecting the interval
        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        sleep_for = max(0, interval_seconds - int(elapsed))
        if sleep_for > 0:
            logger.info("Sleeping %ds until next cycle", sleep_for)
            time.sleep(sleep_for)

    logger.info("Continuous training finished after %d seconds", duration_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run continuous model training")
    parser.add_argument("--duration", type=int, default=int(os.getenv("TRAIN_DURATION_SECONDS", "10800")),
                        help="Total duration in seconds (default 10800 = 3h)")
    parser.add_argument("--interval", type=int, default=int(os.getenv("TRAIN_INTERVAL_SECONDS", "300")),
                        help="Seconds between training cycles (default 300 = 5m)")
    parser.add_argument("--profile", type=str, default=os.getenv("TRAIN_SYMBOLS_PROFILE", "l1l2-top"),
                        help="Symbol profile (deprecated, use QT_UNIVERSE env var)")
    parser.add_argument("--limit", type=int, default=int(os.getenv("TRAIN_LIMIT", "600")),
                        help="Candles to fetch per symbol (default 600)")
    parser.add_argument("--max-symbols", type=int, default=int(os.getenv("TRAIN_MAX_SYMBOLS", "100")),
                        help="Optional cap on number of symbols (deprecated, use QT_MAX_SYMBOLS env var)")

    args = parser.parse_args()

    run_continuous_training(
        duration_seconds=args.duration,
        interval_seconds=args.interval,
        limit=args.limit,
    )
