#!/usr/bin/env python3
"""
Layer 3 — Offline Backtest Runner
=====================================
ISOLATION CONTRACT:
  - Reads only from Parquet files on disk + Redis db=1 (backtest namespace)
  - NEVER reads from or writes to live Redis db=0
  - NEVER touches quantum:stream:apply.plan or any live control key

Architecture:
  1. On startup: auto-runs baseline backtest for TOP_SYMBOLS (last 7 days if data exists)
  2. Persistent: brpop quantum:backtest:queue (db=1) for on-demand backtests
  3. Results: JSON to /opt/quantum/data/backtests/<job_id>.json
  4. Summary: redis hash quantum:backtest:results:<job_id> in db=1
  5. Status: quantum:layer3:backtest:status in db=1 every 30s
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import redis.asyncio as aioredis

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s backtest %(message)s",
)
log = logging.getLogger("backtest")

# ── Config ──────────────────────────────────────────────────────────────
DATA_ROOT     = Path(os.getenv("DATA_ROOT", "/opt/quantum/data"))
BACKTEST_DIR  = DATA_ROOT / "backtests"
OHLCV_DIR     = DATA_ROOT / "ohlcv"
REDIS_HOST    = os.getenv("REDIS_HOST", "localhost")
REDIS_BT_DB   = 1          # ISOLATED: backtest-only Redis DB
QUEUE_KEY     = "quantum:backtest:queue"
STATUS_KEY    = "quantum:layer3:backtest:status"
HEARTBEAT_INTERVAL = 30    # seconds

TOP_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "ADAUSDT", "LINKUSDT", "MATICUSDT",
    "DOTUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT", "INJUSDT",
]

# Strategy parameters
STRATEGIES = {
    "ema_cross": {
        "fast": 8, "slow": 21, "atr_mult_sl": 1.5, "atr_mult_tp": 3.0,
    },
    "rsi_mean_rev": {
        "rsi_period": 14, "oversold": 30, "overbought": 70,
        "atr_mult_sl": 1.0, "atr_mult_tp": 2.0,
    },
    "momentum": {
        "lookback": 20, "threshold_pct": 0.5,
        "atr_mult_sl": 2.0, "atr_mult_tp": 4.0,
    },
}


# ── Data Loading ─────────────────────────────────────────────────────────
def load_parquet_range(symbol: str, days_back: int = 7) -> Optional["pd.DataFrame"]:
    """Load OHLCV Parquet files for the last N days. Returns None if insufficient data."""
    if not HAS_PANDAS:
        return None
    sym_dir = OHLCV_DIR / symbol / "1m"
    if not sym_dir.exists():
        return None

    frames = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    for pq_file in sorted(sym_dir.glob("*.parquet")):
        try:
            date_str = pq_file.stem  # YYYY-MM-DD
            file_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if file_date >= cutoff - timedelta(days=1):
                df = pd.read_parquet(pq_file)
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None
    df = pd.concat(frames).sort_values("open_time").reset_index(drop=True)
    # Filter to exact range
    ts_cutoff = int(cutoff.timestamp() * 1000)
    df = df[df["open_time"] >= ts_cutoff].reset_index(drop=True)
    return df if len(df) >= 100 else None


# ── Indicators ──────────────────────────────────────────────────────────
def _ema(series: "pd.Series", period: int) -> "pd.Series":
    return series.ewm(span=period, adjust=False).mean()

def _atr(df: "pd.DataFrame", period: int = 14) -> "pd.Series":
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _rsi(series: "pd.Series", period: int = 14) -> "pd.Series":
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


# ── Signal Generators ────────────────────────────────────────────────────
def signals_ema_cross(df: "pd.DataFrame", params: dict) -> "pd.Series":
    fast = _ema(df["close"], params["fast"])
    slow = _ema(df["close"], params["slow"])
    sig = pd.Series(0, index=df.index)
    sig[fast > slow] =  1
    sig[fast < slow] = -1
    # Entry only on cross
    prev_sig = sig.shift(1).fillna(0)
    entry = sig != prev_sig
    sig[~entry] = 0
    return sig

def signals_rsi_mean_rev(df: "pd.DataFrame", params: dict) -> "pd.Series":
    rsi = _rsi(df["close"], params["rsi_period"])
    sig = pd.Series(0, index=df.index)
    sig[rsi < params["oversold"]]  =  1
    sig[rsi > params["overbought"]] = -1
    prev_sig = sig.shift(1).fillna(0)
    entry = (sig != 0) & (prev_sig == 0)
    sig[~entry] = 0
    return sig

def signals_momentum(df: "pd.DataFrame", params: dict) -> "pd.Series":
    ret = df["close"].pct_change(params["lookback"]) * 100
    sig = pd.Series(0, index=df.index)
    sig[ret >  params["threshold_pct"]] =  1
    sig[ret < -params["threshold_pct"]] = -1
    prev_sig = sig.shift(1).fillna(0)
    entry = (sig != 0) & (prev_sig != 0) & (sig != prev_sig)
    sig2 = pd.Series(0, index=df.index)
    sig2[sig != 0] = sig[sig != 0]
    sig2[~(sig.shift(1).fillna(0) == 0) & ~(sig != sig.shift(1).fillna(0))] = 0
    return sig


SIGNAL_FNS = {
    "ema_cross":    signals_ema_cross,
    "rsi_mean_rev": signals_rsi_mean_rev,
    "momentum":     signals_momentum,
}


# ── Backtest Engine ──────────────────────────────────────────────────────
def run_backtest(df: "pd.DataFrame", strategy: str, params: dict) -> dict:
    """
    Event-driven candle replay. Returns metrics dict.
    Uses ATR-based SL/TP. Fractional position sizing (1 unit = 1 USDT of notional).
    """
    if not HAS_PANDAS:
        return {"error": "pandas not available"}

    sig_fn = SIGNAL_FNS.get(strategy)
    if sig_fn is None:
        return {"error": f"unknown strategy: {strategy}"}

    signals = sig_fn(df, params)
    atr = _atr(df, 14)

    atr_sl = params.get("atr_mult_sl", 1.5)
    atr_tp = params.get("atr_mult_tp", 3.0)

    trades = []
    equity   = [100.0]  # start with 100 USDT notional
    position = None     # {side, entry_price, sl, tp, entry_idx}
    UNIT_SIZE = 1.0     # 1 USDT per trade (pct basis easier for metrics)

    for i in range(1, len(df)):
        c = df.iloc[i]
        prev_eq = equity[-1]

        # Check existing position for SL/TP
        if position is not None:
            side      = position["side"]
            sl        = position["sl"]
            tp        = position["tp"]
            ep        = position["entry_price"]

            hit_sl = (side ==  1 and c["low"]  <= sl) or \
                     (side == -1 and c["high"] >= sl)
            hit_tp = (side ==  1 and c["high"] >= tp) or \
                     (side == -1 and c["low"]  <= tp)

            if hit_tp:
                pnl_pct = (tp - ep) / ep * side
                equity.append(prev_eq * (1 + pnl_pct))
                trades.append({"pnl_pct": pnl_pct, "result": "win",
                                "bars": i - position["entry_idx"]})
                position = None
                continue
            elif hit_sl:
                pnl_pct = (sl - ep) / ep * side
                equity.append(prev_eq * (1 + pnl_pct))
                trades.append({"pnl_pct": pnl_pct, "result": "loss",
                                "bars": i - position["entry_idx"]})
                position = None
                continue

        equity.append(prev_eq)

        # New entry signal
        if position is None and signals.iloc[i] != 0:
            side = signals.iloc[i]
            ep   = c["close"]
            a    = atr.iloc[i] if atr.iloc[i] > 0 else ep * 0.001
            sl   = ep - side * a * atr_sl
            tp   = ep + side * a * atr_tp
            position = {"side": side, "entry_price": ep,
                        "sl": sl, "tp": tp, "entry_idx": i}

    # Close open position at last bar
    if position is not None:
        ep   = position["entry_price"]
        side = position["side"]
        last = df.iloc[-1]["close"]
        pnl_pct = (last - ep) / ep * side
        equity[-1] = equity[-1] * (1 + pnl_pct)
        trades.append({"pnl_pct": pnl_pct, "result": "open_close",
                       "bars": len(df) - position["entry_idx"]})

    # ── Metrics ─────────────────────────────────────────────────────────
    eq_series = np.array(equity)
    returns   = np.diff(eq_series) / eq_series[:-1]

    n_trades  = len(trades)
    wins      = [t for t in trades if t["pnl_pct"] > 0]
    losses    = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate  = len(wins) / n_trades * 100 if n_trades else 0.0

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses)) or 1e-10
    profit_factor = round(gross_profit / gross_loss, 4)

    total_return = round((eq_series[-1] - eq_series[0]) / eq_series[0] * 100, 4)

    # Sharpe (annualised, 1-min bars: 525600 bars/year)
    sharpe = 0.0
    if len(returns) > 1 and returns.std() > 0:
        sharpe = round(float(returns.mean() / returns.std() * np.sqrt(525_600)), 3)

    # Sortino
    neg_ret = returns[returns < 0]
    sortino = 0.0
    if len(neg_ret) > 1 and neg_ret.std() > 0:
        sortino = round(float(returns.mean() / neg_ret.std() * np.sqrt(525_600)), 3)

    # Max drawdown
    roll_max = np.maximum.accumulate(eq_series)
    dd = (roll_max - eq_series) / roll_max
    max_dd = round(float(dd.max()) * 100, 4)

    avg_bars = round(sum(t["bars"] for t in trades) / n_trades, 1) if n_trades else 0

    return {
        "n_trades":      n_trades,
        "win_rate_pct":  round(win_rate, 2),
        "profit_factor": profit_factor,
        "total_return_pct": total_return,
        "sharpe":        sharpe,
        "sortino":       sortino,
        "max_dd_pct":    max_dd,
        "avg_bars":      avg_bars,
        "bars_total":    len(df),
        "equity_final":  round(float(eq_series[-1]), 4),
    }


# ── Job Processor ─────────────────────────────────────────────────────────
async def process_job(rbt: aioredis.Redis, job: dict) -> dict:
    job_id  = job.get("job_id", str(uuid.uuid4())[:8])
    symbol  = job.get("symbol", "BTCUSDT")
    strategy = job.get("strategy", "ema_cross")
    days_back = int(job.get("days_back", 7))
    params  = job.get("params", STRATEGIES.get(strategy, {}))

    log.info(f"[JOB {job_id}] {symbol} strategy={strategy} days={days_back}")

    t0 = time.time()
    df = load_parquet_range(symbol, days_back)
    if df is None or not HAS_PANDAS:
        result = {"job_id": job_id, "symbol": symbol, "strategy": strategy,
                  "error": "no_data", "ts": int(time.time())}
    else:
        metrics = run_backtest(df, strategy, params)
        result = {
            "job_id":    job_id,
            "symbol":    symbol,
            "strategy":  strategy,
            "days_back": days_back,
            "params":    params,
            "metrics":   metrics,
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "ts":        int(time.time()),
            "ts_human":  datetime.now(timezone.utc).isoformat(),
        }
        log.info(
            f"[JOB {job_id}] {symbol}/{strategy} "
            f"n={metrics['n_trades']} wr={metrics['win_rate_pct']}% "
            f"pf={metrics['profit_factor']} sharpe={metrics['sharpe']} "
            f"dd={metrics['max_dd_pct']}% ret={metrics['total_return_pct']}%"
        )

    # Save to disk
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_file = BACKTEST_DIR / f"{job_id}_{symbol}_{strategy}.json"
    out_file.write_text(json.dumps(result, indent=2))

    # Publish to Redis db=1 (backtest namespace only)
    key = f"quantum:backtest:results:{job_id}"
    flat = {k: str(v) for k, v in result.items() if not isinstance(v, dict)}
    if "metrics" in result:
        for mk, mv in result["metrics"].items():
            flat[f"metrics_{mk}"] = str(mv)
    await rbt.hset(key, mapping=flat)
    await rbt.expire(key, 86400 * 7)  # 7 days TTL

    # Append to leaderboard ZSET by sharpe
    sharpe_val = result.get("metrics", {}).get("sharpe", 0.0)
    try:
        sharpe_val = float(sharpe_val)
    except Exception:
        sharpe_val = 0.0
    lb_key = f"quantum:backtest:leaderboard:{symbol}"
    await rbt.zadd(lb_key, {f"{strategy}:{job_id}": sharpe_val})
    await rbt.zremrangebyrank(lb_key, 0, -51)  # keep top-50

    return result


async def run_auto_baseline(rbt: aioredis.Redis) -> int:
    """Run baseline ema_cross backtest for all symbols that have local data."""
    count = 0
    for sym in TOP_SYMBOLS:
        sym_dir = OHLCV_DIR / sym / "1m"
        if sym_dir.exists() and any(sym_dir.glob("*.parquet")):
            job = {"job_id": f"auto_{sym[:3].lower()}",
                   "symbol": sym, "strategy": "ema_cross", "days_back": 7}
            await process_job(rbt, job)
            count += 1
        await asyncio.sleep(0.05)
    return count


# ── Status Publisher ──────────────────────────────────────────────────────
async def status_publisher(rbt: aioredis.Redis, state: dict):
    while True:
        try:
            await rbt.hset(STATUS_KEY, mapping={
                "ts":            int(time.time()),
                "ts_human":      datetime.now(timezone.utc).isoformat(),
                "jobs_total":    str(state["jobs_total"]),
                "jobs_ok":       str(state["jobs_ok"]),
                "jobs_error":    str(state["jobs_error"]),
                "last_symbol":   state.get("last_symbol", ""),
                "last_strategy": state.get("last_strategy", ""),
                "data_root":     str(DATA_ROOT),
                "pandas_ok":     str(HAS_PANDAS),
            })
        except Exception as e:
            log.warning(f"status publish failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL)


# ── Main ──────────────────────────────────────────────────────────────────
async def main():
    log.info(f"[L3] Backtest Runner starting — data_root={DATA_ROOT} pandas={HAS_PANDAS}")
    log.info(f"[L3] ISOLATION: using Redis db={REDIS_BT_DB} (backtest namespace, NOT live db=0)")

    # Connect to ISOLATED Redis db=1
    rbt = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_BT_DB,
                         decode_responses=True)
    await rbt.ping()
    log.info(f"[L3] Redis db={REDIS_BT_DB} OK")

    state = {"jobs_total": 0, "jobs_ok": 0, "jobs_error": 0}
    asyncio.create_task(status_publisher(rbt, state))

    # Auto-run baseline for symbols with local data
    n = await run_auto_baseline(rbt)
    if n:
        log.info(f"[L3] Auto-baseline complete: {n} symbols backtested")
    else:
        log.info(f"[L3] No local Parquet data yet — waiting for Layer 1 to accumulate candles")
        log.info(f"[L3] Listening for on-demand jobs on {QUEUE_KEY} (db={REDIS_BT_DB})")

    # On-demand job loop
    log.info(f"[L3] Ready — brpop {QUEUE_KEY} (db={REDIS_BT_DB})")
    while True:
        try:
            item = await rbt.brpop(QUEUE_KEY, timeout=30)
            if item is None:
                continue
            _, raw = item
            try:
                job = json.loads(raw)
            except json.JSONDecodeError:
                job = {"symbol": raw.strip(), "strategy": "ema_cross", "days_back": 7}

            state["jobs_total"] += 1
            try:
                result = await process_job(rbt, job)
                state["jobs_ok"] += 1
                state["last_symbol"]   = result.get("symbol", "")
                state["last_strategy"] = result.get("strategy", "")
            except Exception as e:
                log.error(f"Job failed: {e}", exc_info=True)
                state["jobs_error"] += 1

        except Exception as e:
            log.error(f"Queue loop error: {e}", exc_info=True)
            await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())
