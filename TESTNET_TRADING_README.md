# 🚀 Futures Testnet Trading Guide

## Overview

Dette systemet kjører **live futures trading på Binance Testnet** med AI ensemble (XGBoost, LightGBM, N-HiTS, PatchTST). Testnet bruker **FAKE penger** så det er 100% trygt å teste strategier.

## 🎯 Features

- **Futures-spesifikk data**: OHLCV + funding rates + open interest + long/short ratios
- **Leverage trading**: 10-20x leverage med cross margin
- **Dynamic TP/SL**: Basert på volatilitet og market conditions
- **Funding rate awareness**: Detekterer funding rate reversals
- **Liquidation zones**: Unngår crowded liquidation levels
- **24/7 automated trading**: Kontinuerlig testnet trading med AI predictions

## 📋 Setup Steps

### 1. Hent Binance Futures Testnet API Keys

1. Gå til: https://testnet.binancefuture.com
2. Log inn med GitHub/Google
3. Generer API keys (testnet keys er GRATIS og forskjellig fra production!)
4. **VIKTIG**: Testnet keys fungerer KUN på testnet, ikke production

### 2. Sett Environment Variables

Kopier `.env.template` til `.env`:

```bash
cp .env.template .env
```

Rediger `.env` og legg til testnet keys:

```env
# Binance Futures Testnet API Keys (TESTNET ONLY - FAKE MONEY)
BINANCE_TESTNET_API_KEY=your_testnet_api_key_here
BINANCE_TESTNET_SECRET_KEY=your_testnet_secret_key_here
```

### 3. Hent Futures Training Data

```bash
python scripts/fetch_futures_data.py
```

Dette henter:
- **500+ USDT-M perpetual futures** (top liquid pairs)
- **180 dagers historikk** (~750K rader total)
- **Funding rates** (hver 8. time)
- **Open interest** (siste 30 dager)
- **Long/short ratios** (siste 30 dager)

Output: `data/binance_futures_training_data.csv`

ETA: ~30 minutter

### 4. Tren Modeller på Futures Data

```bash
python scripts/train_all_models_futures.py
```

Dette trener:
- **XGBoost**: Gradient boosting på futures patterns (5 min)
- **N-HiTS**: Deep learning time series model (15 min)
- **PatchTST**: Patch-based transformer (20 min)

Total: ~40 minutter

### 5. Start Testnet Trading

```bash
python scripts/testnet_trading.py
```

Dette starter:
- **Live market data scanning** (50 symbols)
- **AI ensemble predictions** (4 models voting)
- **Automated position management**:
  - Entry: Market orders når confidence > 65%
  - Exit: Dynamic TP/SL basert på volatilitet
  - Leverage: 10x (configurable)
  - Max positions: 5 concurrent
  - Position size: 2% av balance per trade

**Logging**: `logs/testnet_trading.log`

## 📊 Monitoring

### Real-time Terminal Output

```
============================================================
🔍 ITERATION #42 - 2025-11-20 22:45:00 UTC
============================================================

📊 ACTIVE POSITIONS (3):
  🟢 BTCUSDT    | LONG  | Entry: $96,532.40 | Current: $96,789.12 | P&L: $+25.68 (+2.66%)
  🔴 ETHUSDT    | SHORT | Entry: $3,421.80  | Current: $3,430.20  | P&L: $-8.40 (-0.82%)
  🟢 SOLUSDT    | LONG  | Entry: $242.15    | Current: $243.90    | P&L: $+7.23 (+2.98%)

🎯 SCANNING FOR SIGNALS (3/5 positions)...

🚨 SIGNAL DETECTED: ARBUSDT
  Action: LONG
  Confidence: 72.3%
  Models: {'xgboost': 'LONG', 'lightgbm': 'LONG', 'nhits': 'LONG', 'patchtst': 'HOLD'}

🎯 OPENING POSITION:
  Symbol: ARBUSDT
  Side: LONG (BUY)
  Quantity: 450.0
  Entry: $1.2340
  TP: $1.2525 (+1.5%)
  SL: $1.2217 (-1.0%)
  Confidence: 72.30%
  Leverage: 10x
  ✅ Position opened! Order ID: 12345678

⏱️  Next scan in 5 minutes...
```

### Log File Analysis

```bash
# Følg live logs
tail -f logs/testnet_trading.log

# Søk etter signaler
grep "SIGNAL DETECTED" logs/testnet_trading.log

# Søk etter åpnede positions
grep "Position opened" logs/testnet_trading.log

# Søk etter P&L updates
grep "P&L:" logs/testnet_trading.log
```

## ⚙️ Configuration

Rediger `scripts/testnet_trading.py`:

```python
# Trading config
self.leverage = 10  # 10x leverage (conservative)
self.position_size_pct = 0.02  # 2% per trade
self.max_positions = 5  # Max concurrent
self.min_confidence = 0.65  # Min AI confidence (65%)
```

**Recommendations**:
- Start med 10x leverage (tryggere)
- Position size 2-3% (risk management)
- Max 3-5 positions (diversification)
- Min confidence 65-70% (kvalitetsfilter)

## 🎓 Futures Trading Concepts

### Funding Rates

- **Positive funding**: Longs betaler shorts (bullish sentiment)
- **Negative funding**: Shorts betaler longs (bearish sentiment)
- **Strategy**: Trade MOT høy funding (fade crowded trades)

### Open Interest

- **Rising OI + Rising Price**: Bullish (new longs entering)
- **Rising OI + Falling Price**: Bearish (new shorts entering)
- **Falling OI**: Position closing (reversal signal)

### Long/Short Ratios

- **High L/S ratio (>2.0)**: Too many longs, potential reversal DOWN
- **Low L/S ratio (<0.5)**: Too many shorts, potential reversal UP
- **Strategy**: Contrarian trading ved ekstreme ratios

### Leverage & Liquidation

- **10x leverage**: 10% move mot deg = liquidation
- **20x leverage**: 5% move mot deg = liquidation
- **Cross margin**: Entire balance som collateral (mer flexible)
- **Isolated margin**: Per-position collateral (mer controlled)

## 📈 Expected Performance

**Testnet Results (estimated)**:
- Win rate: 55-65%
- Avg profit per win: 1.5-3%
- Avg loss per loss: 1-2%
- Max drawdown: 10-15%
- Sharpe ratio: 1.5-2.5

**Note**: Testnet performance ≠ production performance (slippage, latency, psychology).

## 🔒 Safety Features

1. **Testnet only**: Fake money, zero risk
2. **Position limits**: Max 5 concurrent positions
3. **Dynamic TP/SL**: Always set stop losses
4. **Confidence threshold**: Only trade high-confidence signals
5. **Rate limiting**: 1h cooldown per symbol
6. **Leverage cap**: 10x default (conservative)

## 🚨 Troubleshooting

### "Failed to setup futures account"

- Check testnet keys in `.env`
- Verify keys på https://testnet.binancefuture.com
- Ensure `testnet=True` in client initialization

### "No actionable signals found"

- Check model loading: `ensemble.get_model_status()`
- Lower confidence threshold: `self.min_confidence = 0.60`
- Increase symbol list: `symbols[:100]` instead of `symbols[:50]`

### "Position opened but no TP/SL"

- Check order types: `FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET`
- Verify permissions in API key settings (Enable Futures)

### API Rate Limits

- Script har built-in delays (0.5s per symbol)
- Every 10 symbols: 5s sleep
- Max 1200 requests/minute (Binance limit: 2400/min)

## 📚 Further Reading

- [Binance Futures API Docs](https://binance-docs.github.io/apidocs/futures/en/)
- [Testnet Registration](https://testnet.binancefuture.com)
- [Funding Rate Explained](https://www.binance.com/en/support/faq/360033525031)
- [Open Interest Guide](https://www.binance.com/en/blog/futures/open-interest-explained-421499824684903515)

## 🛑 Production Deployment (CAUTION!)

**DO NOT deploy to production without**:
1. Extensive testnet validation (2+ weeks)
2. Risk management audit
3. Proper monitoring setup
4. Gradual capital allocation
5. Kill switch implementation

Testnet er for TESTING. Production trading har:
- Real slippage
- Real latency
- Real liquidations
- Real stress

**Always trade responsibly with capital you can afford to lose.**

---

**Questions?** Check logs first: `logs/testnet_trading.log`

**Issues?** Verify:
1. Testnet keys correct
2. Models trained
3. Futures data fetched
4. Environment variables set
