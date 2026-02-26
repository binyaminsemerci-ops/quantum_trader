#!/usr/bin/env python3
"""
Realized PnL — siste time + historikk
"""
import redis
import json
import subprocess
from datetime import datetime, timezone, timedelta

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

now = datetime.now(timezone.utc)
one_hour_ago = now - timedelta(hours=1)
print(f"=== REALIZED PnL — SISTE TIME ===")
print(f"Fra: {one_hour_ago.strftime('%H:%M:%S UTC')} → {now.strftime('%H:%M:%S UTC')}\n")

# ─── 1. trade.closed stream ───────────────────────────────────
print("── trade.closed stream (alle lukkede handler) ──")
try:
    # Get entries from last hour (stream ID = ms timestamp)
    min_id = str(int(one_hour_ago.timestamp() * 1000)) + "-0"
    entries = r.xrange("quantum:stream:trade.closed", min=min_id, max="+")
    
    if not entries:
        # Try alternative stream names
        for stream in ["quantum:stream:trades.closed", "quantum:stream:trade.close",
                       "quantum:stream:closed", "quantum:trades:closed"]:
            entries = r.xrange(stream, min=min_id, max="+")
            if entries:
                print(f"  Found in: {stream}")
                break
    
    if entries:
        total_pnl = 0.0
        wins = 0
        losses = 0
        trades = []
        
        for entry_id, fields in entries:
            ts_ms = int(entry_id.split("-")[0])
            ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            
            # Parse payload (might be JSON in 'payload' field or flat fields)
            payload_raw = fields.get("payload", "")
            flat = fields
            
            if payload_raw:
                try:
                    flat = json.loads(payload_raw)
                except:
                    pass
            
            symbol = flat.get("symbol", fields.get("symbol", "?"))
            pnl_pct = flat.get("pnl_pct", flat.get("pnl", flat.get("realized_pnl_pct", None)))
            pnl_usd = flat.get("pnl_usd", flat.get("realized_pnl_usd", None))
            r_val = flat.get("r_multiple", flat.get("r_value", flat.get("R", None)))
            side = flat.get("side", flat.get("direction", "?"))
            source = flat.get("source", fields.get("source", "?"))
            
            trades.append({
                "ts": ts,
                "symbol": symbol,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd,
                "r": r_val,
                "side": side,
                "source": source,
                "raw": flat,
            })
        
        # Sort by time
        trades.sort(key=lambda x: x["ts"])
        
        print(f"  Antall lukkede handler: {len(trades)}\n")
        
        for t in trades:
            pnl_str = f"PnL={t['pnl_pct']}%" if t['pnl_pct'] else ""
            r_str = f"R={t['r']}" if t['r'] else ""
            usd_str = f"${t['pnl_usd']}" if t['pnl_usd'] else ""
            print(f"  {t['ts'].strftime('%H:%M:%S')} {t['symbol']:15} {t['side']:5} {pnl_str:10} {r_str:8} {usd_str:10} src={t['source']}")
            
            try:
                if t['pnl_pct']:
                    pct = float(str(t['pnl_pct']).replace('%', ''))
                    total_pnl += pct
                    if pct >= 0:
                        wins += 1
                    else:
                        losses += 1
            except:
                pass
        
        print(f"\n  Wins: {wins}  Losses: {losses}  Win rate: {wins/(wins+losses)*100:.0f}%" if (wins+losses)>0 else "")
        print(f"  Total realized PnL %: {total_pnl:+.2f}%")
    else:
        print("  INGEN entries i quantum:stream:trade.closed siste time")
        # Show stream info
        info = r.xinfo_stream("quantum:stream:trade.closed") if r.exists("quantum:stream:trade.closed") else None
        if info:
            print(f"  Stream length: {info.get('length', '?')}")
            print(f"  Last entry: {info.get('last-generated-id', '?')}")
        else:
            print("  Stream eksisterer ikke")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ─── 2. Scan Redis for closed trade records ───────────────────
print("\n── Redis: closed trade hashes siste time ──")
try:
    # Look for trade result hashes
    patterns = [
        "quantum:trade:result:*",
        "quantum:trade:closed:*",
        "quantum:pnl:*",
        "quantum:result:*",
        "quantum:history:trade:*",
    ]
    
    found_any = False
    for pattern in patterns:
        keys = r.keys(pattern)
        if keys:
            found_any = True
            print(f"  Pattern {pattern}: {len(keys)} keys")
            # Sample latest 5
            for key in sorted(keys)[-5:]:
                data = r.hgetall(key)
                if not data:
                    data = {"value": r.get(key)}
                ts = data.get("timestamp", data.get("ts", data.get("closed_at", "")))
                sym = data.get("symbol", "?")
                pnl = data.get("realized_pnl", data.get("pnl", data.get("pnl_usd", "?")))
                print(f"    {key}: symbol={sym} pnl={pnl} ts={str(ts)[:16]}")
    
    if not found_any:
        print("  Ingen dedikerte trade result keys funnet")
except Exception as e:
    print(f"  ERROR: {e}")

# ─── 3. intent-executor HARVEST SUCCESS logs ─────────────────
print("\n── intent-executor: HARVEST SUCCESS siste time ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-intent-executor', '--since', '1 hour ago', '--no-pager'],
        capture_output=True, text=True, timeout=15
    )
    logs = result.stdout.splitlines()
    
    harvest_lines = [l for l in logs if 'HARVEST SUCCESS' in l or 'trade.closed' in l or 'PnL=' in l]
    
    total_pnl_r = 0.0
    harvest_events = []
    
    for line in harvest_lines:
        if 'HARVEST SUCCESS' in line or 'PnL=' in line:
            # Extract symbol and PnL
            sym = "?"
            pnl = None
            r_val = None
            
            parts = line.split()
            for i, p in enumerate(parts):
                if 'HARVEST' in p and i+1 < len(parts):
                    sym = parts[i+1] if parts[i+1] not in ['SUCCESS','INTENT','CLOSE','SKIP'] else (parts[i+2] if i+2 < len(parts) else "?")
                if p.startswith('PnL='):
                    pnl = p.replace('PnL=', '')
                if p.startswith('R='):
                    r_val = p.replace('R=', '')
            
            # Better parse
            if 'HARVEST SUCCESS:' in line:
                after = line.split('HARVEST SUCCESS:')[1].strip()
                sym = after.split()[0]
            if 'Published trade.closed:' in line:
                after = line.split('trade.closed:')[1].strip()
                sym = after.split()[0]
                for part in after.split():
                    if part.startswith('PnL='):
                        pnl = part.replace('PnL=', '')
                    if part.startswith('R='):
                        r_val = part.replace('R=', '')
            
            # Get timestamp from line
            ts_str = line.split(' ')[2] if len(line.split(' ')) > 2 else ""
            
            harvest_events.append((ts_str, sym, pnl, r_val, line.strip()[-80:]))
            
            if r_val:
                try:
                    total_pnl_r += float(r_val)
                except:
                    pass
    
    if harvest_events:
        print(f"  Closed trades siste time: {len([e for e in harvest_events if 'trade.closed' in str(e)])}")
        seen = set()
        for ts, sym, pnl, r_val, raw in harvest_events:
            key = (ts, sym)
            if key not in seen:
                seen.add(key)
                print(f"  {ts:12} {sym:15} PnL={str(pnl):10} R={str(r_val):6}")
    else:
        print("  Ingen HARVEST SUCCESS i siste time")
    
    # Count totals
    total_harvests = len([l for l in logs if 'HARVEST SUCCESS' in l])
    total_failed = len([l for l in logs if '401' in l or 'Unauthorized' in l])
    
    print(f"\n  Siste time totalt:")
    print(f"  HARVEST SUCCESS: {total_harvests}")
    print(f"  401/Unauthorized: {total_failed}")
    
    # Get current metrics line
    for line in reversed(logs):
        if 'harvest_executed=' in line:
            metrics = line.split('Metrics:')[1].strip() if 'Metrics:' in line else line
            print(f"\n  Kumulativ metrics: {metrics.strip()[-100:]}")
            break
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ─── 4. autonomous-trader exits siste time ───────────────────
print("\n── autonomous-trader: exits siste time ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-autonomous-trader', '--since', '1 hour ago', '--no-pager'],
        capture_output=True, text=True, timeout=15
    )
    logs = result.stdout.splitlines()
    
    exit_lines = [l for l in logs if 'Exit intent published' in l or ('CLOSE' in l and 'PnL=' in l)]
    
    total_r = 0.0
    closed_trades = []
    
    for line in exit_lines:
        if 'CLOSE' in line and 'PnL=' in line:
            sym = "?"
            pnl = None
            r_val = None
            
            parts = line.split()
            for i, p in enumerate(parts):
                if 'CLOSE' in p:
                    sym = parts[i-1] if i > 0 else "?"
                if p.startswith('PnL='):
                    pnl = p.replace('PnL=', '')
                if p.startswith('R='):
                    r_val = p.replace('R=', '')
            
            closed_trades.append((sym, pnl, r_val))
            
            try:
                if r_val:
                    total_r += float(r_val)
            except:
                pass
    
    print(f"  Exit intents publisert: {len(exit_lines)}")
    print(f"  Unike exits med PnL: {len(closed_trades)}")
    if closed_trades:
        print(f"\n  Symbol          PnL        R")
        seen = set()
        for sym, pnl, r_val in closed_trades:
            if sym not in seen:
                seen.add(sym)
                print(f"  {sym:15} {str(pnl):10} {str(r_val)}")
        print(f"\n  Sum R (alle exits): {total_r:+.2f}")
    
    # Stats lines
    stats_lines = [l for l in logs if 'Stats:' in l]
    if stats_lines:
        total_entries = 0
        total_exits = 0
        for sl in stats_lines:
            parts = sl.split('Stats:')[1].strip().split()
            try:
                e = int(parts[0])
                x = int(parts[2].rstrip(','))
                total_entries += e
                total_exits += x
            except:
                pass
        print(f"\n  Siste time totalt: {total_entries} entries, {total_exits} exits")
except Exception as e:
    print(f"  ERROR: {e}")

# ─── 5. Binance open positions akkurat nå ────────────────────
print("\n── Binance open positions (akkurat nå) ──")
try:
    result = subprocess.run(
        ['journalctl', '-u', 'quantum-autonomous-trader', '-n', '50', '--no-pager'],
        capture_output=True, text=True, timeout=10
    )
    logs = result.stdout.splitlines()
    
    monitoring = [l for l in logs if 'Checking' in l and 'positions' in l]
    if monitoring:
        last = monitoring[-1]
        n = last.split('Checking')[1].split('position')[0].strip()
        print(f"  Autonomous trader overvåker: {n} posisjoner")
    
    # Find HOLD/CLOSE decisions
    print("\n  Siste beslutninger:")
    for line in reversed(logs):
        if any(x in line for x in ['HOLD', 'CLOSE', 'EXIT']):
            if 'PnL=' in line or 'R=' in line:
                print(f"  {line.strip()[-80:]}")
        if len([1 for l in logs[-20:] if l == line]) > 10:
            break
    
    for line in reversed(logs):
        if 'Checking' in line and 'position' in line:
            print(f"\n  {line.strip()[-60:]}")
            break
except Exception as e:
    print(f"  ERROR: {e}")
