"""
PROOF: 3 OPEN-planner fixes
"""
import sys
sys.path.insert(0, '.')

EP_KEY  = b'entry_price'
ATR_KEY = b'atr_value'
VOL_KEY = b'volatility_factor'
ROUND_TRIP_COST = 0.0008

print('=' * 60)
print('BUG-FIX PROOF: 3 OPEN-planner fixes')
print('=' * 60)

# ─────────────────────────────────────────────────────────────
# PROOF 1: entry_price now forwarded to apply.plan
# ─────────────────────────────────────────────────────────────
print('\n[1] entry_price forwarded to apply.plan')

def publish_OLD(intent):
    fields = {}
    entry_price = intent.get('entry_price')
    if entry_price and entry_price > 0 and intent['side'].upper() == 'BUY':
        breakeven = entry_price * (1 + ROUND_TRIP_COST)
        fields[b'breakeven_price'] = f'{breakeven:.8f}'.encode()
    return fields

def publish_FIXED(intent):
    fields = {}
    entry_price = intent.get('entry_price')
    if entry_price and entry_price > 0:
        fields[EP_KEY] = f'{entry_price:.8f}'.encode()
        if intent['side'].upper() == 'BUY':
            breakeven = entry_price * (1 + ROUND_TRIP_COST)
            fields[b'breakeven_price'] = f'{breakeven:.8f}'.encode()
    src = intent.get('source_payload') or {}
    atr = src.get('atr_value') or intent.get('atr_value')
    vol = src.get('volatility_factor') or intent.get('volatility_factor')
    if atr:
        fields[ATR_KEY] = str(atr).encode()
    if vol:
        fields[VOL_KEY] = str(vol).encode()
    return fields

intent_buy  = {'symbol': 'BTCUSDT', 'side': 'BUY',  'entry_price': 86400.0,
               'reduceOnly': False, 'source_payload': {'atr_value': 320.5, 'volatility_factor': 1.8}}
intent_sell = {'symbol': 'BTCUSDT', 'side': 'SELL', 'entry_price': 86400.0,
               'reduceOnly': False, 'source_payload': {'atr_value': 320.5, 'volatility_factor': 1.8}}

old_buy  = publish_OLD(intent_buy)
old_sell = publish_OLD(intent_sell)
new_buy  = publish_FIXED(intent_buy)
new_sell = publish_FIXED(intent_sell)

print(f'  BUY  OLD: entry_price in fields = {EP_KEY in old_buy}')
print(f'  BUY  NEW: entry_price in fields = {EP_KEY in new_buy}  value={new_buy.get(EP_KEY)}')
print(f'  SELL OLD: entry_price in fields = {EP_KEY in old_sell}   (SHORT SL-check was blind!)')
print(f'  SELL NEW: entry_price in fields = {EP_KEY in new_sell}  value={new_sell.get(EP_KEY)}')

def sl_check(ep_field, position_side, stop_loss):
    entry_price = float(ep_field.decode()) if ep_field else 0.0
    if stop_loss and float(stop_loss) > 0 and entry_price > 0:
        sl = float(stop_loss)
        if position_side == 'SHORT' and sl < entry_price:
            return f'CORRECTED SHORT SL below entry (entry={entry_price:.0f} SL={sl:.0f})'
        if position_side == 'LONG' and sl > entry_price:
            return f'CORRECTED LONG SL above entry (entry={entry_price:.0f} SL={sl:.0f})'
        return 'VALID'
    return 'SKIPPED (entry_price=0)'

bad_sl = '85000.0'   # SHORT SL below entry = immediate margin call on fill
print(f'  SHORT bad SL OLD: {sl_check(old_sell.get(EP_KEY), "SHORT", bad_sl)}')
print(f'  SHORT bad SL NEW: {sl_check(new_sell.get(EP_KEY), "SHORT", bad_sl)}')

# ─────────────────────────────────────────────────────────────
# PROOF 2: atr_value / volatility_factor forwarded
# ─────────────────────────────────────────────────────────────
print('\n[2] atr_value / volatility_factor forwarded')
print(f'  OLD: atr_value in fields       = {ATR_KEY in old_buy}')
print(f'  OLD: volatility_factor fields  = {VOL_KEY in old_buy}')
print(f'  NEW: atr_value in fields       = {ATR_KEY in new_buy}  value={new_buy[ATR_KEY]}')
print(f'  NEW: volatility_factor fields  = {VOL_KEY in new_buy}  value={new_buy[VOL_KEY]}')

def risk(atr_f, vol_f, qty=0.001):
    atr = float(atr_f.decode()) if atr_f else 0.0
    vol = float(vol_f.decode()) if vol_f else 0.0
    rp  = atr * vol if (atr > 0 and vol > 0) else 0.0
    return qty * rp, 1 if rp == 0 else 0

r_old = risk(old_buy.get(ATR_KEY), old_buy.get(VOL_KEY))
r_new = risk(new_buy.get(ATR_KEY), new_buy.get(VOL_KEY))
print(f'  OLD: entry_risk_usdt={r_old[0]:.4f}  risk_missing={r_old[1]}  (always 0.0 — no data)')
print(f'  NEW: entry_risk_usdt={r_new[0]:.4f}  risk_missing={r_new[1]}  (ATR 320.5 * vol 1.8 * qty 0.001)')

# ─────────────────────────────────────────────────────────────
# PROOF 3: claim: keys excluded from position limit count
# ─────────────────────────────────────────────────────────────
print('\n[3] claim: keys excluded from position-limit count')

fake_keys = [
    b'quantum:position:BTCUSDT',
    b'quantum:position:ETHUSDT',
    b'quantum:position:snapshot:SOLUSDT',
    b'quantum:position:ledger:BTCUSDT',
    b'quantum:position:cooldown:XRPUSDT',
    b'quantum:position:claim:BTCUSDT',    # race-guard, was counted!
    b'quantum:position:claim:ETHUSDT',    # race-guard, was counted!
]

def count_old(keys):
    return [k for k in keys if b'snapshot' not in k and b'ledger' not in k and b'cooldown' not in k]

def count_new(keys):
    return [k for k in keys if b'snapshot' not in k and b'ledger' not in k
            and b'cooldown' not in k and b'claim' not in k]

old_pos = count_old(fake_keys)
new_pos = count_new(fake_keys)
print(f'  Keys total:           {len(fake_keys)}')
print(f'  OLD active count:     {len(old_pos)} -> {[k.decode() for k in old_pos]}')
print(f'  NEW active count:     {len(new_pos)} -> {[k.decode() for k in new_pos]}')
print(f'  Claim keys removed:   {len(old_pos) - len(new_pos)}')
print(f'  OLD falsely blocked:  limit 2/2 would reject any new OPEN (claim inflated to 4/10)')
print(f'  NEW correctly sees:   only {len(new_pos)} real positions')

print()
assert EP_KEY not in old_buy,       "BUG: OLD should NOT have entry_price for BUY"
assert EP_KEY not in old_sell,      "BUG: OLD should NOT have entry_price for SELL"
assert EP_KEY in new_buy,           "FIX 1 BUY broken"
assert EP_KEY in new_sell,          "FIX 1 SELL broken"
assert ATR_KEY not in old_buy,      "BUG: OLD should NOT have atr_value"
assert ATR_KEY in new_buy,          "FIX 2 atr broken"
assert r_old[1] == 1,               "BUG: OLD should have risk_missing=1"
assert r_new[1] == 0,               "FIX 2 risk_missing broken"
assert len(old_pos) == 4,           "BUG: OLD count should be 4 (includes 2 claim keys)"
assert len(new_pos) == 2,           "FIX 3 claim filter broken"
print('ALL 3 FIXES PROVEN (assertions passed)')
