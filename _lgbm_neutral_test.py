import joblib, numpy as np
from pathlib import Path

LABELS = ['SELL', 'HOLD', 'BUY']
m = joblib.load(sorted(Path('ai_engine/models').glob('lightgbm_v*_v3.pkl'))[-1])
print('Model:', sorted(Path('ai_engine/models').glob('lightgbm_v*_v3.pkl'))[-1].name)

# Neutral input (all zeros = mean of training data after scaler)
p = m.predict_proba(np.zeros((1, 49)))[0]
winner = LABELS[np.argmax(p)]
print(f'Zeros (neutral): SELL={p[0]:.3f} HOLD={p[1]:.3f} BUY={p[2]:.3f}  -> {winner}')

# Random market conditions
np.random.seed(42)
X = np.random.randn(1000, 49)
proba = m.predict_proba(X)
cnt = np.bincount(np.argmax(proba, axis=1), minlength=3)
print(f'Random N(0,1) x1000: SELL={cnt[0]} HOLD={cnt[1]} BUY={cnt[2]}')
print(f'Mean probs:          SELL={proba[:,0].mean():.3f} HOLD={proba[:,1].mean():.3f} BUY={proba[:,2].mean():.3f}')

# Bullish inputs (positive momentum, rsi=60, ema positive)
bull = np.zeros((1, 49))
bull[0, 11] = 60   # rsi=60
bull[0, 21] = 1.5  # ema_21_dist=+1.5%
bull[0, 44] = 0.5  # momentum_5=+0.5%
pb = m.predict_proba(bull)[0]
print(f'Bullish input:   SELL={pb[0]:.3f} HOLD={pb[1]:.3f} BUY={pb[2]:.3f}  -> {LABELS[np.argmax(pb)]}')

# Bearish inputs
bear = np.zeros((1, 49))
bear[0, 11] = 35   # rsi=35
bear[0, 21] = -1.5 # ema_21_dist=-1.5%
bear[0, 44] = -0.5 # momentum_5=-0.5%
pbe = m.predict_proba(bear)[0]
print(f'Bearish input:   SELL={pbe[0]:.3f} HOLD={pbe[1]:.3f} BUY={pbe[2]:.3f}  -> {LABELS[np.argmax(pbe)]}')
