"""Quick per-symbol LGBM vs XGB comparison on live market data."""
import sys; sys.path.insert(0, '.')
import requests, pandas as pd, numpy as np, joblib, pickle, xgboost as xgb
from pathlib import Path
from backend.shared.unified_features import get_feature_engineer

FEATURES = [
    'returns','log_returns','price_range','body_size','upper_wick','lower_wick',
    'is_doji','is_hammer','is_engulfing','gap_up','gap_down','rsi','macd',
    'macd_signal','macd_hist','stoch_k','stoch_d','roc','ema_9','ema_9_dist',
    'ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist',
    'sma_20','sma_50','adx','plus_di','minus_di','bb_middle','bb_upper',
    'bb_lower','bb_width','bb_position','atr','atr_pct','volatility',
    'volume_sma','volume_ratio','obv','obv_ema','vpt','momentum_5',
    'momentum_10','momentum_20','acceleration','relative_spread'
]

lgbm_path = sorted(Path('ai_engine/models').glob('lightgbm_v*_v3.pkl'))[-1]
xgb_path  = sorted(p for p in Path('ai_engine/models').glob('xgb_v6_*.pkl')
                   if p.stem.count('_') == 3)[-1]
print(f"LGBM: {lgbm_path.name}")
print(f"XGB:  {xgb_path.name}")

lgbm  = joblib.load(lgbm_path)
xgb_m = pickle.load(open(xgb_path, 'rb'))
eng   = get_feature_engineer()

LABELS = ['SELL', 'HOLD', 'BUY']
symbols = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT','AVAXUSDT','LTCUSDT']

print(f"\n{'SYM':<10}  {'LGBM':<28}  {'XGB':<28}")
print('-' * 70)
for sym in symbols:
    r = requests.get('https://api.binance.com/api/v3/klines',
        params={'symbol': sym, 'interval': '1h', 'limit': 210}, timeout=10)
    df = pd.DataFrame(r.json(), columns=[
        'ot','open','high','low','close','volume','ct','qv','nt','tbb','tbq','ig'])
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    feat = eng.compute_features(df)
    row  = feat[FEATURES].dropna().iloc[-1:].values.astype('f')

    lp   = lgbm.predict_proba(row)[0]
    lout = LABELS[np.argmax(lp)]

    dm   = xgb.DMatrix(row)
    xp   = xgb_m.predict(dm)[0]
    xout = LABELS[np.argmax(xp)]

    lgbm_str = f"{lout}  S={lp[0]:.2f} H={lp[1]:.2f} B={lp[2]:.2f}"
    xgb_str  = f"{xout}  S={xp[0]:.2f} H={xp[1]:.2f} B={xp[2]:.2f}"
    print(f"{sym:<10}  {lgbm_str:<28}  {xgb_str:<28}")
