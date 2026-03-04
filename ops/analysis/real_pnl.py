import hmac, hashlib, time, requests, datetime

API_KEY = 'w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg'
API_SECRET = 'QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg'
BASE = 'https://testnet.binancefuture.com'

def sign(params):
    qs = '&'.join(f'{k}={v}' for k,v in params.items())
    sig = hmac.new(API_SECRET.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + '&signature=' + sig

def get(endpoint, params={}):
    p = dict(params)
    p['timestamp'] = int(time.time()*1000)
    r = requests.get(BASE+endpoint+'?'+sign(p), headers={'X-MBX-APIKEY': API_KEY}, timeout=10)
    return r.json()

# ── Account snapshot ──────────────────────────────────────────
acct = get('/fapi/v2/account')
total_wb     = float(acct.get('totalWalletBalance', 0))
total_unr    = float(acct.get('totalUnrealizedProfit', 0))
total_mb     = float(acct.get('totalMarginBalance', 0))
print(f"=== ACCOUNT SNAPSHOT ===")
print(f"Wallet Balance (realised deposits + all closed PnL): {total_wb:.4f} USDT")
print(f"Unrealized PnL (open positions):                     {total_unr:.4f} USDT")
print(f"Margin Balance (wallet + unrealized):                {total_mb:.4f} USDT")

# ── Pull income with pagination (no 1000-row limit) ───────────
# Go back 30 days in 7-day chunks to get full picture
now = int(time.time()*1000)
all_income = []
chunk_ms = 7 * 24 * 3600 * 1000
start_30d = now - 30 * 24 * 3600 * 1000

chunk_start = start_30d
while chunk_start < now:
    chunk_end = min(chunk_start + chunk_ms, now)
    batch = get('/fapi/v1/income', {'startTime': chunk_start, 'endTime': chunk_end, 'limit': 1000})
    if isinstance(batch, list):
        all_income.extend(batch)
    chunk_start = chunk_end + 1
    time.sleep(0.1)

# Deduplicate by id if present
seen = set()
deduped = []
for x in all_income:
    key = (x.get('tranId'), x['incomeType'], x['time'])
    if key not in seen:
        seen.add(key)
        deduped.append(x)
all_income = sorted(deduped, key=lambda x: x['time'])

print(f"\n=== 30-DAY INCOME HISTORY (paginated, no truncation) ===")
print(f"Total entries: {len(all_income)}")

if all_income:
    net   = sum(float(x['income']) for x in all_income)
    rlzd  = sum(float(x['income']) for x in all_income if x['incomeType']=='REALIZED_PNL')
    fees  = sum(float(x['income']) for x in all_income if x['incomeType']=='COMMISSION')
    fund  = sum(float(x['income']) for x in all_income if x['incomeType']=='FUNDING_FEE')
    first = datetime.datetime.utcfromtimestamp(all_income[0]['time']/1000)
    last  = datetime.datetime.utcfromtimestamp(all_income[-1]['time']/1000)
    print(f"Period: {first}  →  {last}  (UTC)")
    print(f"Net total:       {net:+.4f} USDT")
    print(f"Realized PnL:    {rlzd:+.4f} USDT")
    print(f"Commissions:     {fees:+.4f} USDT")
    print(f"Funding fees:    {fund:+.4f} USDT")

    # Per-day breakdown
    print(f"\n=== PER-DAY BREAKDOWN ===")
    from collections import defaultdict
    day_net  = defaultdict(float)
    day_rlzd = defaultdict(float)
    day_fees = defaultdict(float)
    for x in all_income:
        d = datetime.datetime.utcfromtimestamp(x['time']/1000).strftime('%Y-%m-%d')
        day_net[d]  += float(x['income'])
        if x['incomeType']=='REALIZED_PNL':
            day_rlzd[d] += float(x['income'])
        if x['incomeType']=='COMMISSION':
            day_fees[d] += float(x['income'])
    for d in sorted(day_net):
        print(f"  {d}  net={day_net[d]:+8.4f}  pnl={day_rlzd[d]:+8.4f}  fees={day_fees[d]:+8.4f}")

    # Per-symbol breakdown (top losers)
    print(f"\n=== TOP 15 LOSERS (realized PnL only) ===")
    sym_pnl = defaultdict(float)
    sym_fees = defaultdict(float)
    for x in all_income:
        s = x.get('symbol','UNKNOWN')
        if x['incomeType']=='REALIZED_PNL':
            sym_pnl[s] += float(x['income'])
        if x['incomeType']=='COMMISSION':
            sym_fees[s] += float(x['income'])
    combined = {s: sym_pnl[s]+sym_fees[s] for s in set(list(sym_pnl.keys())+list(sym_fees.keys()))}
    for sym, val in sorted(combined.items(), key=lambda x: x[1])[:15]:
        print(f"  {sym:<15} net={val:+8.4f}  pnl={sym_pnl[sym]:+8.4f}  fees={sym_fees[sym]:+8.4f}")

print(f"\nDone.")
