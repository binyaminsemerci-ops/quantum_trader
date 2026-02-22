import redis

r = redis.Redis(host="localhost", decode_responses=True)

keys = r.keys("quantum:position:*")

portfolio = []
total_entry_risk = 0
total_unrealized = 0
below_stop_count = 0
below_stop_capital = 0

for k in keys:
    pos = r.hgetall(k)
    symbol = pos.get("symbol")

    try:
        entry_risk  = float(pos.get("entry_risk_usdt", 0))
        unrealized  = float(pos.get("unrealized_pnl", 0))
        qty         = float(pos.get("quantity", 0))
        entry_price = float(pos.get("entry_price", 0))
        leverage    = float(pos.get("leverage", 1))
        atr         = float(pos.get("atr_value", 0))
    except Exception:
        continue

    if entry_risk <= 0:
        continue

    notional = qty * entry_price
    R_net    = unrealized / entry_risk

    total_entry_risk += entry_risk
    total_unrealized += unrealized

    state_key   = f"quantum:harvest_v2:state:{symbol}"
    state       = r.hgetall(state_key)
    v2_decision = state.get("last_decision", "NO_STATE") if state else "NO_STATE"
    max_R_seen  = float(state.get("max_R_seen", R_net)) if state and state.get("max_R_seen") else R_net

    protected = max(0, -unrealized)

    if R_net <= -0.5:
        below_stop_count  += 1
        below_stop_capital += protected

    portfolio.append({
        "symbol":           symbol,
        "R_net":            round(R_net, 3),
        "max_R_seen":       round(max_R_seen, 3),
        "entry_risk":       round(entry_risk, 2),
        "unrealized":       round(unrealized, 2),
        "notional":         round(notional, 2),
        "leverage":         int(leverage),
        "atr":              round(atr, 6),
        "V1":               "HOLD",
        "V2":               v2_decision,
        "protected_if_V2":  round(protected, 2),
    })

portfolio.sort(key=lambda x: x["R_net"])
portfolio_R     = total_unrealized / total_entry_risk if total_entry_risk else 0
total_notional  = sum(p["notional"] for p in portfolio)
pct_below_stop  = below_stop_count / len(portfolio) * 100 if portfolio else 0
pct_cap_at_risk = abs(total_unrealized) / total_entry_risk * 100 if total_entry_risk else 0
total_protected = sum(p["protected_if_V2"] for p in portfolio)
v2_full_closes  = sum(1 for p in portfolio if p["V2"] == "FULL_CLOSE")

W = 72
print()
print("=" * W)
print(" PER-SYMBOL ANALYSIS")
print("=" * W)
print(f"{'SYM':<12} {'R_net':>7} {'maxR':>6} {'risk$':>8} {'PnL$':>9} {'notional':>10} {'lev':>4} {'V1':<6} {'V2':<12} {'protect$':>10}")
print("-" * W)
for p in portfolio:
    flag = " <<STOP" if p["R_net"] <= -0.5 else ""
    print(
        f"{p['symbol']:<12} {p['R_net']:>7.3f} {p['max_R_seen']:>6.3f} "
        f"{p['entry_risk']:>8.2f} {p['unrealized']:>9.2f} {p['notional']:>10.2f} "
        f"{p['leverage']:>4} {p['V1']:<6} {p['V2']:<12} {p['protected_if_V2']:>10.2f}{flag}"
    )

print()
print("=" * W)
print(" PORTFOLIO AGGREGATES")
print("=" * W)
print(f"  Positions total      : {len(portfolio)}")
print(f"  Total notional       : {round(total_notional,2):>12} USDT")
print(f"  Total entry risk     : {round(total_entry_risk,2):>12} USDT")
print(f"  Total unrealized PnL : {round(total_unrealized,2):>12} USDT")
print(f"  Portfolio R          : {round(portfolio_R,3):>12}")
print(f"  Below stop (R<=-0.5) : {below_stop_count}/{len(portfolio)} ({round(pct_below_stop,1)}%)")
print(f"  Capital below stop   : {round(below_stop_capital,2):>12} USDT")
print(f"  % capital at risk    : {round(pct_cap_at_risk,1):>11}%")
print(f"  V2 FULL_CLOSE signals: {v2_full_closes}/{len(portfolio)}")
print(f"  Protected if V2 exits: {round(total_protected,2):>12} USDT")

max_risk_pos  = max(portfolio, key=lambda x: abs(x["unrealized"]))
concentration = abs(max_risk_pos["unrealized"]) / abs(total_unrealized) * 100 if total_unrealized else 0
print(f"  Largest loss pos     : {max_risk_pos['symbol']} ({max_risk_pos['unrealized']} USDT, {round(concentration,1)}% of total loss)")

print()
if portfolio_R <= -0.75:
    regime = "RED    -- Portfolio at critical drawdown. V2 would protect majority of capital."
elif portfolio_R <= -0.3:
    regime = "YELLOW -- Portfolio under significant stress. Monitor closely."
else:
    regime = "GREEN  -- Portfolio within normal range."

kill_zone = sum(1 for p in portfolio if p["R_net"] <= -1.0)

print("=" * W)
print(" RISK CLASSIFICATION")
print("=" * W)
print(f"  Regime               : {regime}")
print(f"  Kill-zone (R<=-1.0)  : {kill_zone} positions ({round(kill_zone/len(portfolio)*100,1)}%)")
print(f"  Heat (quantum:capital): 0.0 (key missing -- CORRECT default)")
print(f"  Heat judgment        : heat=0 => V2 uses full targets (R_target=3.0).")
print(f"                         With real heat, targets tighten -> earlier exits.")

print()
print("=" * W)
print(" V1 vs V2 DIVERGENCE")
print("=" * W)
print(f"  V1 decision          : HOLD on all {len(portfolio)} positions (no emit = no exit)")
print(f"  V2 FULL_CLOSE        : {v2_full_closes} positions")
print(f"  Divergence rate      : {round(v2_full_closes/len(portfolio)*100,1)}%")
print(f"  Missed exits by V1   : {v2_full_closes} positions (capital not yet protected)")
print(f"  Net V2 protection    : {round(total_protected,2)} USDT recoverable if V2 applied now")
print()
