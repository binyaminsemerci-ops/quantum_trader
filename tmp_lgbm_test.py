import joblib, numpy as np
m  = joblib.load('/app/models/lgbm_model.pkl')
sc = joblib.load('/app/models/lgbm_scaler.pkl')
print("Model type:", type(m).__name__)

THRESH = 0.003

def classify(p):
    if p > THRESH: return 2   # BUY
    if p < -THRESH: return 0  # SELL
    return 1                  # HOLD

# Test 1: zeros (training mean in raw space)
z = m.predict(sc.transform(np.zeros((1,49))))[0]
print("zeros pred (training mean): {:.5f} -> {}".format(z, ['SELL','HOLD','BUY'][classify(z)]))

# Test 2: random x1000
np.random.seed(42); preds = []
for _ in range(1000):
    preds.append(m.predict(sc.transform(np.random.randn(1,49)))[0])
preds = np.array(preds)
c = [0,0,0]
for p in preds: c[classify(p)] += 1
print("random x1000 mean={:.5f} std={:.5f}".format(preds.mean(), preds.std()))
print("random x1000 classes: SELL={} HOLD={} BUY={}".format(c[0],c[1],c[2]))
print("  -> neutral random input gives {}% SELL".format(round(c[0]/10,1)))

# Test 3: check if model has systematic negative bias
# Compute what % of training mean range falls as SELL
percentiles = np.percentile(preds, [10,25,50,75,90])
print("pred percentiles p10/p25/p50/p75/p90:", [round(x,5) for x in percentiles])

# Test 4: known BUY conditions
feat=['returns','log_returns','price_range','body_size','upper_wick','lower_wick',
      'is_doji','is_hammer','is_engulfing','gap_up','gap_down','rsi','macd',
      'macd_signal','macd_hist','stoch_k','stoch_d','roc','ema_9','ema_9_dist',
      'ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist',
      'sma_20','sma_50','adx','plus_di','minus_di','bb_middle','bb_upper',
      'bb_lower','bb_width','bb_position','atr','atr_pct','volatility',
      'volume_sma','volume_ratio','obv','obv_ema','vpt','momentum_5',
      'momentum_10','momentum_20','acceleration','relative_spread']
xb = np.zeros((1,49)); xb[0,feat.index('rsi')]=25; xb[0,feat.index('returns')]=0.015
p = m.predict(sc.transform(xb))[0]
print("BUY signal (RSI=25,+1.5%): {:.5f} -> {}".format(p, ['SELL','HOLD','BUY'][classify(p)]))

xs = np.zeros((1,49)); xs[0,feat.index('rsi')]=75; xs[0,feat.index('returns')]=-0.015
p = m.predict(sc.transform(xs))[0]
print("SELL signal(RSI=75,-1.5%): {:.5f} -> {}".format(p, ['SELL','HOLD','BUY'][classify(p)]))
feat=['returns','log_returns','price_range','body_size','upper_wick','lower_wick',
      'is_doji','is_hammer','is_engulfing','gap_up','gap_down','rsi','macd',
      'macd_signal','macd_hist','stoch_k','stoch_d','roc','ema_9','ema_9_dist',
      'ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist',
      'sma_20','sma_50','adx','plus_di','minus_di','bb_middle','bb_upper',
      'bb_lower','bb_width','bb_position','atr','atr_pct','volatility',
      'volume_sma','volume_ratio','obv','obv_ema','vpt','momentum_5',
      'momentum_10','momentum_20','acceleration','relative_spread']
def pp(x):
    r=m.predict_proba(sc.transform(x))[0]
    return "SELL={:.3f} HOLD={:.3f} BUY={:.3f}".format(r[0],r[1],r[2])
print("zeros:       ", pp(np.zeros((1,49))))
xb=np.zeros((1,49)); xb[0,feat.index('rsi')]=25; xb[0,feat.index('returns')]=0.015
print("BUY-signal:  ", pp(xb))
xs=np.zeros((1,49)); xs[0,feat.index('rsi')]=75; xs[0,feat.index('returns')]=-0.015
print("SELL-signal: ", pp(xs))
np.random.seed(42); c=[0,0,0]
for _ in range(1000):
    r=m.predict_proba(sc.transform(np.random.randn(1,49)))[0]; c[r.argmax()]+=1
print("random x1000: SELL={} HOLD={} BUY={}".format(*c))
print("n_estimators:", m.n_estimators_)
if hasattr(m, 'n_classes_'):
    print('n_classes:', m.n_classes_, 'classes:', m.classes_)
n = m.n_features_in_
print('n_features:', n)
x = np.zeros((1, n))
if hasattr(m, 'predict_proba'):
    p = m.predict_proba(x)[0]
    print('proba(zeros):', p.round(4))
    preds = np.argmax(m.predict_proba(np.random.randn(100, n)), axis=1)
    from collections import Counter
    print('class dist (100 random):', dict(Counter(preds.tolist())))
    print('action map: 0=SELL 1=HOLD 2=BUY')
else:
    out = m.predict(x)
    print('regression(zeros):', out)
    out100 = m.predict(np.random.randn(100, n))
    print('regression range:', out100.min().round(4), 'to', out100.max().round(4))
    print('threshold needed for BUY/SELL = ±0.3 (currently all HOLD since max ~0.025)')
