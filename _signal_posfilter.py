import shutil

filepath = "/opt/quantum/signal_injector.py"
shutil.copy2(filepath, filepath + ".bak_posfilter")

with open(filepath, "r") as f:
    content = f.read()

old = '            change = (close_price - open_price) / open_price\n\n            if change >= MIN_MOVE_PCT:'

new = '''            change = (close_price - open_price) / open_price

            # POSITION FILTER: skip signal if we already have an open position (2026-02-25)
            # Prevents blasting new SELL signals for coins we already shorted -> stops harvest_brain churn
            try:
                _snap_amt = await r.hget(f"quantum:position:snapshot:{symbol}", "position_amt")
                if _snap_amt and abs(float(_snap_amt)) > 0.0:
                    logger.debug(f"  SKIP_OPEN {symbol:20s}  already_in_position={_snap_amt}")
                    continue
            except Exception:
                pass

            if change >= MIN_MOVE_PCT:'''

count = content.count(old)
if count != 1:
    print(f"ERROR: found {count} occurrences (need exactly 1)")
    exit(1)

content = content.replace(old, new)
with open(filepath, "w") as f:
    f.write(content)
print("SUCCESS: position filter added to signal_injector run_cycle")
print("Backup: " + filepath + ".bak_posfilter")
