import joblib, numpy as np
m = joblib.load('/app/models/lgbm_model.pkl')
print('type:', type(m).__name__)
THRESH = 0.003
feat = ['returns','log_returns','price_range','body_size','upper_wick','lower_wick','is_doji','is_hammer','is_engulfing','gap_up','gap_down','rsi','macd','macd_signal','macd_hist','stoch_k','stoch_d','roc','ema_9','ema_9_dist','ema_21','ema_21_dist','ema_50','ema_50_dist','ema_200','ema_200_dist','sma_20','sma_50','adx','plus_di','minus_di','bb_middle','bb_upper','bb_lower','bb_width','bb_position','atr','atr_pct','volatility','volume_sma','volume_ratio','obv','obv_ema','vpt','momentum_5','momentum_10','momentum_20','acceleration','relative_spread']
def cl(v): return 'SELL' if v<-THRESH else ('BUY' if v>THRESH else 'HOLD')
z = m.predict(np.zeros((1,49)))[0]
print('zeros raw:', round(z,5), '->', cl(z))
np.random.seed(42); preds=[]
for _ in range(1000): preds.append(m.predict(np.random.randn(1,49))[0])
preds=np.array(preds); c={'SELL':0,'HOLD':0,'BUY':0}
for v in preds: c[cl(v)]+=1
print('random x1000:', c, 'mean={:.5f}'.format(preds.mean()))
xb=np.zeros((1,49)); xb[0,feat.index('rsi')]=25; xb[0,feat.index('returns')]=0.015
print('BUY-signal:', round(m.predict(xb)[0],5), '->', cl(m.predict(xb)[0]))
xs=np.zeros((1,49)); xs[0,feat.index('rsi')]=75; xs[0,feat.index('returns')]=-0.015
print('SELL-signal:', round(m.predict(xs)[0],5), '->', cl(m.predict(xs)[0]))
print('n_features:', m.n_features_in_ if hasattr(m,'n_features_in_') else '?')
