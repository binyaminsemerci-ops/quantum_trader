"""
One-shot shadow trade catch-up.
Reads ALL harvest v2 signals directly from the stream (no consumer group),
applies the V2 close logic, and writes results to the accuracy key.
Run once to backfill trades=0 issue.
"""
import asyncio, json, time, collections
import redis.asyncio as aioredis

STREAM = "quantum:stream:harvest.v2.shadow"
KEY_ACCURACY = "quantum:sandbox:accuracy:latest"
KEY_GATE     = "quantum:sandbox:gate:latest"
KEY_SHADOW_TRADES = "quantum:shadow:trades:closed"
KEY_EQUITY   = "quantum:shadow:equity:series"

MIN_SHADOW_TRADES = 30
MIN_ACCURACY = 55.0
MIN_PF = 1.1

CLOSE_DECISIONS = {"FULL_CLOSE", "PARTIAL_75", "PARTIAL_50", "PARTIAL_25"}

async def main():
    r = aioredis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    
    print(f"[CATCHUP] Reading all messages from {STREAM}...")
    
    # Read ALL messages in stream regardless of consumer group
    msgs = await r.xrange(STREAM, "-", "+", count=5000)
    print(f"[CATCHUP] Found {len(msgs)} messages")
    
    win_window = collections.deque(maxlen=50)
    closed_trades = []
    equity = 10000.0
    peak = 10000.0
    
    for msg_id, fields in msgs:
        decision = fields.get("decision", "").upper()
        if decision not in CLOSE_DECISIONS:
            continue
        
        sym = fields.get("symbol", "")
        side = fields.get("side", "LONG").upper()
        unrealized_pnl = float(fields.get("unrealized_pnl", 0.0))
        initial_risk = float(fields.get("initial_risk", 10.0))
        r_net = float(fields.get("R_net", 0.0))
        
        if not sym:
            continue
        
        is_win = unrealized_pnl > 0.0
        win_window.append(1 if is_win else 0)
        equity += unrealized_pnl
        if equity > peak:
            peak = equity

        trade = {
            "symbol": sym,
            "side": side,
            "entry": 0.0,
            "exit": 0.0,
            "size": round(initial_risk / 10.0, 4),
            "notional": initial_risk,
            "pnl_usdt": round(unrealized_pnl, 4),
            "pnl_pct": round(r_net * 100.0, 3),
            "reason": decision,
            "source": "harvest_v2_catchup",
            "close_ts": int(time.time()),
        }
        closed_trades.append(trade)
    
    n_trades = len(closed_trades)
    n_wins = sum(t["pnl_usdt"] > 0 for t in closed_trades)
    gross_wins = sum(t["pnl_usdt"] for t in closed_trades if t["pnl_usdt"] > 0)
    gross_losses = abs(sum(t["pnl_usdt"] for t in closed_trades if t["pnl_usdt"] <= 0))
    pf = round(gross_wins / gross_losses, 3) if gross_losses > 0 else float(gross_wins > 0)
    
    rolling_acc = sum(win_window) / len(win_window) * 100 if win_window else 0.0
    total_pnl = equity - 10000.0
    n_needed = max(0, MIN_SHADOW_TRADES - n_trades)
    gate_open = (n_trades >= MIN_SHADOW_TRADES and rolling_acc >= MIN_ACCURACY and pf >= MIN_PF)
    gate_str = "OPEN" if gate_open else "CLOSED"
    
    print(f"[CATCHUP] Processed {n_trades} FULL_CLOSE trades")
    print(f"[CATCHUP] Win rate: {n_wins}/{n_trades} = {rolling_acc:.1f}%")
    print(f"[CATCHUP] Profit factor: {pf:.3f}")
    print(f"[CATCHUP] Total P&L: {total_pnl:+.2f} USDT")
    print(f"[CATCHUP] Gate: {gate_str}")
    
    now = int(time.time())
    
    # Write accuracy key (read by DAG 8 C2)
    await r.hset(KEY_ACCURACY, mapping={
        "accuracy_pct":     str(round(rolling_acc, 1)),
        "n_trades":         str(n_trades),
        "profit_factor":    str(pf),
        "gate":             gate_str,
        "min_trades_needed": str(n_needed),
        "paper_pnl_usd":    str(round(total_pnl, 2)),
        "paper_wins":       str(n_wins),
        "paper_losses":     str(n_trades - n_wins),
        "total_shadow_signals": str(len(msgs)),
        "exit_signals":     str(n_trades),
        "source":           "harvest_v2_catchup",
        "ts":               str(now),
    })
    print(f"[CATCHUP] Wrote quantum:sandbox:accuracy:latest")
    
    await r.hset(KEY_GATE, mapping={
        "gate":    gate_str,
        "reason":  f"n={n_trades}/{MIN_SHADOW_TRADES} acc={rolling_acc:.1f}% pf={pf:.2f}",
        "n_trades": str(n_trades),
        "accuracy_pct": str(round(rolling_acc, 1)),
        "profit_factor": str(pf),
        "ts":      str(now),
    })
    print(f"[CATCHUP] Wrote quantum:sandbox:gate:latest  gate={gate_str}")
    
    # Write closed trades list (last 50)
    pipe = r.pipeline()
    for t in closed_trades[-50:]:
        pipe.lpush(KEY_SHADOW_TRADES, json.dumps(t))
    pipe.ltrim(KEY_SHADOW_TRADES, 0, 499)
    await pipe.execute()
    print(f"[CATCHUP] Wrote {min(50, n_trades)} trades to {KEY_SHADOW_TRADES}")
    
    # Also set consumer group pointer to end (avoid re-processing on controller restart)
    try:
        await r.xgroup_setid(STREAM, "shadow_controller", "$")
        print(f"[CATCHUP] XGROUP SETID shadow_controller $ (skip re-processing)")
    except Exception as e:
        print(f"[CATCHUP] XGROUP SETID skip: {e}")
    
    print(f"\n[CATCHUP] DONE — DAG 8 C2 now: n={n_trades} acc={rolling_acc:.1f}% pf={pf:.2f} gate={gate_str}")
    await r.aclose()

asyncio.run(main())
