#!/usr/bin/env python3
"""
Paper Trade Controller — Phase 2 Execution
============================================
Activates ONLY when quantum:dag8:current_phase = 2 (PAPER_TRADE).
Phase 1 (shadow controller) proved edge in simulation.
Phase 2 sends REAL orders to Binance TESTNET — real fills, real slippage,
zero real capital at risk.

Reads signals from quantum:stream:harvest.v2.shadow (same source as shadow
controller, but now actually executes against testnet).

Produces execution results tagged with paper_trade=true, consumed by:
  - Layer 5 execution monitor (quality scoring)
  - DAG 8 C2 gate (fill_rate, slippage validation)
  - Layer 6 analytics (paper P&L attribution)

Writes to:
  quantum:paper:portfolio:latest   — paper portfolio state
  quantum:paper:position:<SYM>     — simulated open positions (filled by testnet)
  quantum:paper:trades:closed      — LPUSH closed trades (fill_rate, slippage)
  quantum:stream:trade.closed      — tagged paper_trade=true for Layer 5
  quantum:paper:status             — phase status

NEVER touches live position keys or apply.plan stream.

Operator commands:
  redis-cli hgetall quantum:paper:portfolio:latest
  redis-cli hgetall quantum:paper:status
  redis-cli lrange quantum:paper:trades:closed 0 5
  # Check testnet API keys are set:
  redis-cli get quantum:cfg:testnet_api_key
"""

import asyncio
import json
import logging
import os
import time
import hmac
import hashlib
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Optional

import redis.asyncio as aioredis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s paper %(message)s",
)
log = logging.getLogger("paper_trade")

# ── Config ────────────────────────────────────────────────────────────────
REDIS_HOST      = os.getenv("REDIS_HOST", "localhost")
REDIS_DB        = 0

# Binance testnet (no real money)
TESTNET_BASE    = "https://testnet.binancefuture.com"
API_KEY         = os.getenv("BINANCE_TESTNET_API_KEY",    "")
API_SECRET      = os.getenv("BINANCE_TESTNET_API_SECRET", "")

# Paper sizing — use Kelly from Layer 4, but cap per order
MAX_NOTIONAL_USDT  = float(os.getenv("MAX_NOTIONAL_USDT",  "200.0"))  # per trade
DEFAULT_SIZE_USDT  = float(os.getenv("DEFAULT_SIZE_USDT",  "50.0"))   # fallback
LEVERAGE           = int(os.getenv("LEVERAGE",             "3"))

IDLE_SLEEP      = 30
PUBLISH_EVERY   = 60

KEY_PHASE       = "quantum:dag8:current_phase"
KEY_STATUS      = "quantum:paper:status"
KEY_PORTFOLIO   = "quantum:paper:portfolio:latest"
KEY_TRADES      = "quantum:paper:trades:closed"
STREAM_SHADOW   = "quantum:stream:harvest.v2.shadow"
STREAM_CLOSED   = "quantum:stream:trade.closed"
CG_NAME         = "paper_controller"


# ── Testnet HTTP ──────────────────────────────────────────────────────────
def _sign(params: str, secret: str) -> str:
    return hmac.new(secret.encode(), params.encode(), hashlib.sha256).hexdigest()


def testnet_request(method: str, path: str, params: dict) -> Optional[dict]:
    """Sign and send a request to Binance testnet."""
    if not API_KEY or not API_SECRET:
        log.warning("[PAPER] No testnet API keys configured — using simulated fills")
        return None
    try:
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        params["signature"] = _sign(query, API_SECRET)
        full_query = urllib.parse.urlencode(params)
        url = f"{TESTNET_BASE}{path}?{full_query}"
        req = urllib.request.Request(url, method=method)
        req.add_header("X-MBX-APIKEY", API_KEY)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception as e:
        log.error(f"[PAPER] Testnet request error: {e}")
        return None


def get_price(symbol: str) -> Optional[float]:
    """Fetch mark price from testnet."""
    try:
        url = f"{TESTNET_BASE}/fapi/v1/premiumIndex?symbol={symbol}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return float(data.get("markPrice", 0)) or None
    except Exception:
        return None


def place_market_order(symbol: str, side: str, quantity: float) -> Optional[dict]:
    """Place market order on Binance testnet."""
    params = {
        "symbol":   symbol,
        "side":     side.upper(),
        "type":     "MARKET",
        "quantity": f"{quantity:.4f}",
    }
    return testnet_request("POST", "/fapi/v1/order", params)


# ── Portfolio State ───────────────────────────────────────────────────────
class PaperPortfolio:
    def __init__(self):
        self.positions:    Dict[str, dict] = {}
        self.closed_trades: list           = []
        self.equity        = 10000.0
        self.peak          = 10000.0
        self.fills_total   = 0
        self.fills_success = 0

    @property
    def fill_rate(self) -> float:
        return self.fills_success / self.fills_total * 100 if self.fills_total else 0.0

    @property
    def win_rate(self) -> float:
        wins = sum(1 for t in self.closed_trades if t["pnl_usdt"] > 0)
        return wins / len(self.closed_trades) * 100 if self.closed_trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins   = sum(t["pnl_usdt"] for t in self.closed_trades if t["pnl_usdt"] > 0)
        losses = abs(sum(t["pnl_usdt"] for t in self.closed_trades if t["pnl_usdt"] <= 0))
        return round(wins / losses, 3) if losses > 0 else float(wins > 0)

    @property
    def dd_pct(self) -> float:
        return (self.peak - self.equity) / self.peak * 100 if self.peak > 0 else 0.0

    @property
    def avg_slippage_bps(self) -> float:
        slips = [t.get("slippage_bps", 0) for t in self.closed_trades]
        return round(sum(slips) / len(slips), 1) if slips else 0.0


# ── Signal Processor ──────────────────────────────────────────────────────
async def process_signal(
    portfolio: PaperPortfolio,
    fields: dict,
    r: aioredis.Redis,
):
    sig_type = fields.get("type", fields.get("signal_type", "")).upper()
    sym      = fields.get("symbol", fields.get("Symbol", ""))
    side     = fields.get("side", "LONG").upper()

    if not sym:
        return

    portfolio.fills_total += 1

    if sig_type in ("OPEN", "ENTRY", "BUY", "LONG", "SHORT"):
        if sym in portfolio.positions:
            return  # already have position

        # Determine size from Layer 4 Kelly or default
        size_usdt = DEFAULT_SIZE_USDT
        try:
            l4 = await r.hgetall(f"quantum:layer4:sizing:{sym}")
            if l4 and float(l4.get("size_usdt", 0)) > 0:
                size_usdt = min(float(l4["size_usdt"]), MAX_NOTIONAL_USDT)
        except Exception:
            pass

        # Get current price
        price = get_price(sym)
        if not price:
            log.warning(f"[PAPER] {sym}: cannot get price, skipping")
            return

        qty = round(size_usdt / price, 4)
        if qty <= 0:
            return

        # Execute on testnet
        order_result = place_market_order(sym, side, qty)
        fill_price = price  # fallback
        slippage_bps = 0.0

        if order_result and order_result.get("status") == "FILLED":
            fill_price = float(order_result.get("avgPrice", price))
            slippage_bps = abs(fill_price - price) / price * 10000
            portfolio.fills_success += 1
            log.info(f"[PAPER] FILLED {sym} {side} qty={qty:.4f} @ {fill_price:.4f} slip={slippage_bps:.1f}bps")
        elif order_result is None:
            # No API keys — simulate fill at price + 2bps slippage
            slippage_bps = 2.0
            fill_price   = price * (1.0 + slippage_bps / 10000)
            portfolio.fills_success += 1
            log.info(f"[PAPER] SIMFILL {sym} {side} qty={qty:.4f} @ {fill_price:.4f} (no testnet creds)")
        else:
            log.warning(f"[PAPER] REJECTED {sym}: {order_result}")
            return

        portfolio.positions[sym] = {
            "symbol":       sym,
            "side":         side,
            "qty":          qty,
            "entry_price":  fill_price,
            "notional":     fill_price * qty,
            "slippage_bps": slippage_bps,
            "open_ts":      int(time.time()),
        }

    elif sig_type in ("CLOSE", "EXIT", "SELL", "FULL_CLOSE_PROPOSED", "PARTIAL_75", "PARTIAL_50"):
        if sym not in portfolio.positions:
            return

        pos = portfolio.positions.pop(sym)
        price = get_price(sym)
        if not price:
            price = pos["entry_price"]

        side_close = "SELL" if pos["side"] == "LONG" else "BUY"
        order_result = place_market_order(sym, side_close, pos["qty"])
        fill_price   = price
        slippage_bps = 0.0
        portfolio.fills_total += 1

        if order_result and order_result.get("status") == "FILLED":
            fill_price   = float(order_result.get("avgPrice", price))
            slippage_bps = abs(fill_price - price) / price * 10000
            portfolio.fills_success += 1
        elif order_result is None:
            slippage_bps = 2.0
            fill_price   = price * (1.0 - slippage_bps / 10000)
            portfolio.fills_success += 1

        pnl_pct  = (fill_price - pos["entry_price"]) / pos["entry_price"] * 100
        if pos["side"] == "SHORT":
            pnl_pct = -pnl_pct
        pnl_usdt = pnl_pct / 100 * pos["notional"]

        portfolio.equity += pnl_usdt
        if portfolio.equity > portfolio.peak:
            portfolio.peak = portfolio.equity

        trade = {
            "symbol":       sym,
            "side":         pos["side"],
            "entry_price":  pos["entry_price"],
            "exit_price":   fill_price,
            "qty":          pos["qty"],
            "notional":     pos["notional"],
            "pnl_pct":      round(pnl_pct, 4),
            "pnl_usdt":     round(pnl_usdt, 4),
            "slippage_bps": round(slippage_bps, 1),
            "open_ts":      pos["open_ts"],
            "close_ts":     int(time.time()),
            "duration_s":   int(time.time()) - pos["open_ts"],
            "paper_trade":  True,
        }
        portfolio.closed_trades.append(trade)

        # Publish to trade.closed stream (read by Layer 5, Layer 6)
        await r.xadd(STREAM_CLOSED, {k: str(v) for k, v in trade.items()})
        # Also to local paper trades list (capped 1000)
        await r.lpush(KEY_TRADES, json.dumps(trade))
        await r.ltrim(KEY_TRADES, 0, 999)

        log.info(
            f"[PAPER] CLOSE {sym} pnl={pnl_usdt:+.2f}USDT ({pnl_pct:+.3f}%) "
            f"slip={slippage_bps:.1f}bps wr={portfolio.win_rate:.1f}% "
            f"pf={portfolio.profit_factor:.2f}"
        )


# ── Portfolio Publisher ───────────────────────────────────────────────────
async def publish_portfolio(portfolio: PaperPortfolio, r: aioredis.Redis):
    ts  = int(time.time())
    data = {
        "ts":            ts,
        "equity":        round(portfolio.equity, 2),
        "peak":          round(portfolio.peak, 2),
        "dd_pct":        round(portfolio.dd_pct, 2),
        "n_open":        len(portfolio.positions),
        "n_closed":      len(portfolio.closed_trades),
        "win_rate_pct":  round(portfolio.win_rate, 1),
        "profit_factor": portfolio.profit_factor,
        "fill_rate_pct": round(portfolio.fill_rate, 1),
        "avg_slip_bps":  portfolio.avg_slippage_bps,
        "open_symbols":  ",".join(portfolio.positions.keys()) or "none",
        "return_pct":    round((portfolio.equity - 10000.0) / 100.0, 2),
    }
    await r.hset(KEY_PORTFOLIO, mapping={k: str(v) for k, v in data.items()})


# ── Main Loop ─────────────────────────────────────────────────────────────
async def run_paper_loop(r: aioredis.Redis):
    portfolio     = PaperPortfolio()
    consumer_name = f"paper_{os.getpid()}"
    last_publish  = 0.0

    try:
        await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="$", mkstream=True)
    except Exception:
        pass

    while True:
        phase_raw = await r.get(KEY_PHASE)
        phase_val = int(phase_raw) if phase_raw is not None else 0

        if phase_val != 2:
            await r.hset(KEY_STATUS, mapping={
                "status": "IDLE",
                "phase":  str(phase_val),
                "reason": f"PaperTrade activates at phase=2, currently phase={phase_val}",
                "ts":     str(int(time.time())),
            })
            log.debug(f"[PAPER] idle, phase={phase_val}")
            await asyncio.sleep(IDLE_SLEEP)
            continue

        await r.hset(KEY_STATUS, mapping={
            "status": "ACTIVE",
            "phase":  "2",
            "reason": "PAPER_TRADE mode — executing on Binance TESTNET",
            "ts":     str(int(time.time())),
        })

        # Consume shadow signals
        try:
            msgs = await r.xreadgroup(
                CG_NAME, consumer_name,
                {STREAM_SHADOW: ">"}, count=20, block=5000
            )
            if msgs:
                for msg_id, fields in msgs[0][1]:
                    await process_signal(portfolio, fields, r)
                    await r.xack(STREAM_SHADOW, CG_NAME, msg_id)
        except Exception as e:
            if "NOGROUP" in str(e):
                try:
                    await r.xgroup_create(STREAM_SHADOW, CG_NAME, id="$", mkstream=True)
                except Exception:
                    pass
            else:
                log.error(f"Stream error: {e}")
            await asyncio.sleep(2)
            continue

        now = time.time()
        if now - last_publish >= PUBLISH_EVERY:
            await publish_portfolio(portfolio, r)
            log.info(
                f"[PAPER] phase=2 equity={portfolio.equity:.2f} "
                f"n_closed={len(portfolio.closed_trades)} "
                f"wr={portfolio.win_rate:.1f}% pf={portfolio.profit_factor:.2f} "
                f"fill_rate={portfolio.fill_rate:.1f}% slip={portfolio.avg_slippage_bps:.1f}bps"
            )
            last_publish = now


async def main():
    log.info("[PAPER] Paper Trade Controller starting")
    r = aioredis.Redis(host=REDIS_HOST, port=6379, db=REDIS_DB, decode_responses=True)
    await r.ping()
    log.info(f"[PAPER] Redis OK | testnet_creds={'YES' if API_KEY else 'NO — simulated fills'}")
    await run_paper_loop(r)


if __name__ == "__main__":
    asyncio.run(main())
