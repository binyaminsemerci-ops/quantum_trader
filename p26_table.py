#!/usr/bin/env python3
import subprocess

def hget(key, field):
    try:
        out = subprocess.check_output(['redis-cli','HGET',key,field], text=True).strip()
        return out
    except Exception:
        return ''

symbols = ['BTCUSDT','ETHUSDT','SOLUSDT']
print(f'{"SYMBOL":8}  {"ACTION":18}  {"R_net":8}  {"K":8}  {"new_sl":12}')
for s in symbols:
    base = f'quantum:harvest:proposal:{s}'
    action = hget(base,'harvest_action')
    rnet   = hget(base,'R_net') or hget(base,'r_net')
    k      = hget(base,'kill_score') or hget(base,'K')
    newsl  = hget(base,'new_sl_proposed')
    print(f'{s:8}  {action:18}  {rnet:8}  {k:8}  {newsl:12}')
