# 🤖 AI Trading Architecture & Funksjons Prinsipp

## Oversikt: Hvordan AI Trading Fungerer

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUANTUM TRADER AI SYSTEM                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   DATAKILDER │─────▶│ FEATURE ENG. │─────▶│   ML MODEL   │─────▶│   HANDEL     │
│              │      │              │      │              │      │   BESLUTNING │
└──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
     ▼                      ▼                      ▼                      ▼
  Market Data          Indikatorer           XGBoost/RF           BUY/SELL/HOLD
  Sentiment            Tekniske              Prediksjoner         + Confidence
  News/Trends          + Sentiment           + Scoring            + Stop Loss
```

---

## 1️⃣ DATAKILDER (Input Layer)

### Market Data
```python
# Fra: backend/routes/external_data.py
await binance_ohlcv(symbol="BTCUSDT", limit=600)
# Returnerer:
{
  "candles": [
    {
      "timestamp": "2025-11-11T00:00:00Z",
      "open": 43000.0,
      "high": 43500.0,
      "low": 42800.0,
      "close": 43200.0,
      "volume": 1250.5
    },
    # ... 599 flere candles (historiske data)
  ]
}
```

**Data punkter per candle:**
- **Open**: Åpningspris for perioden
- **High**: Høyeste pris i perioden
- **Low**: Laveste pris i perioden
- **Close**: Sluttkurs for perioden
- **Volume**: Handelsvolum (antall coins handlet)

### Sentiment Data
```python
# Fra: backend/utils/twitter_client.py
await twitter_sentiment(symbol="BTC")
# Returnerer:
{
  "score": 0.65,  # Positivitet: -1 (negativt) til +1 (positivt)
  "positive": 0.65,
  "neutral": 0.25,
  "negative": 0.10
}
```

### News/Trending Data
```python
# Fra: backend/routes/coingecko_data.py
trending_coins = await get_trending_coins()
# Brukes for å identifisere "hot" coins med mye oppmerksomhet
```

---

## 2️⃣ FEATURE ENGINEERING (Transformasjon)

### Fra: `ai_engine/feature_engineer.py`

Feature engineering er hjerte av AI systemet. Vi transformerer rå data til **features** som ML-modellen kan lære fra:

```python
def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beregner tekniske indikatorer fra OHLCV data
    
    Input: Raw price data (open, high, low, close, volume)
    Output: DataFrame med 20+ beregnede features
    """
    
    # 1. MOVING AVERAGES (Trend følging)
    df['MA_10'] = df['close'].rolling(10).mean()    # Kort-term trend
    df['MA_50'] = df['close'].rolling(50).mean()    # Medium-term trend
    df['EMA_10'] = df['close'].ewm(span=10).mean()  # Exponential moving average
    
    # 2. MOMENTUM INDICATORS (Kjøps/selgs kraft)
    df['RSI_14'] = _compute_rsi(df['close'], 14)    # Relative Strength Index (0-100)
    df['MACD'] = _compute_macd(df['close'])         # Moving Average Convergence Divergence
    
    # 3. VOLATILITET (Risiko måling)
    df['ATR_14'] = _compute_atr(df, 14)             # Average True Range
    df['Bollinger_Upper'] = MA_20 + (2 * std)       # Upper band
    df['Bollinger_Lower'] = MA_20 - (2 * std)       # Lower band
    
    # 4. VOLUME ANALYSIS (Markedsaktivitet)
    df['Volume_MA_20'] = df['volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_MA_20']
    
    # 5. PRICE PATTERNS (Markedsstruktur)
    df['High_Low_Ratio'] = (df['high'] - df['low']) / df['close']
    df['Close_Open_Ratio'] = (df['close'] - df['open']) / df['open']
    
    return df
```

### Sentiment Features
```python
def add_sentiment_features(df, sentiment_series, news_counts):
    """
    Legger til sentiment og news data som features
    
    Sentiment brukes til å fange "market mood" som tekniske 
    indikatorer ikke kan se.
    """
    df['sentiment'] = sentiment_series          # Twitter/social sentiment
    df['news_count'] = news_counts              # Antal news events
    df['sentiment_ma_5'] = sentiment_series.rolling(5).mean()  # Trend i sentiment
    
    return df
```

### Target Variable (Hva vi prøver å predikere)
```python
def add_target(df, horizon=1, threshold=0.0):
    """
    Beregner "Return" - den faktiske pris endringen vi vil predikere
    
    horizon=1: Predict price change 1 period frem
    threshold=0.0: Positiv return = BUY signal, negativ = SELL
    """
    df['Return'] = (df['close'].shift(-horizon) - df['close']) / df['close']
    
    # Return > 0 → Prisen vil stige (BUY)
    # Return < 0 → Prisen vil falle (SELL)
    # Return ≈ 0 → Ingen klar trend (HOLD)
    
    return df
```

**Eksempel Output:**
```
timestamp           close   MA_10   RSI_14  sentiment  Return    Action
2025-11-11 01:00   43000   42850   65.4    0.65       0.023     BUY
2025-11-11 02:00   43200   42900   68.2    0.70       0.015     BUY
2025-11-11 03:00   43150   42950   62.1    0.55      -0.010     SELL
```

---

## 3️⃣ MACHINE LEARNING MODEL (Scikit-learn)

### Training Pipeline: `ai_engine/train_and_save.py`

```python
async def train_and_save(symbols: List[str], limit: int = 600):
    """
    STEG 1: Samle treningsdata
    """
    all_data = []
    for symbol in symbols:
        # Hent 600 candles (historiske priser)
        candles, sentiment, news = await _fetch_symbol_data(symbol, limit)
        all_data.append((symbol, candles, sentiment, news))
    
    """
    STEG 2: Feature Engineering
    """
    X, y = build_dataset(all_data)
    # X: Array med 20+ features per tidspunkt (RSI, MA, sentiment, etc.)
    # y: Array med faktiske "Returns" (pris endringer)
    
    print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
    # Output: Training data: 5000 samples, 25 features
    
    """
    STEG 3: Data Preprocessing
    """
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Normaliserer features til mean=0, std=1
    # Viktig fordi ML modeller fungerer bedre med normalisert data
    
    """
    STEG 4: Model Selection & Training
    """
    # Primær: XGBoost (Gradient Boosted Decision Trees)
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(
            n_estimators=100,     # Antal trær i ensemble
            max_depth=5,          # Maks dybde per tre
            learning_rate=0.1,    # Hvor fort modellen lærer
            subsample=0.8,        # Sample 80% av data per tre
            colsample_bytree=0.8  # Sample 80% av features per tre
        )
    except ImportError:
        # Fallback: Random Forest (enklere, men fortsatt kraftig)
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
    
    # Tren modellen
    model.fit(X_scaled, y)
    # Modellen lærer sammenhenger mellom features (RSI, MA, etc.) og Returns
    
    """
    STEG 5: Model Evaluation
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    predictions = model.predict(X_scaled)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.6f}")        # Lower is better
    print(f"  R²: {r2:.4f}")          # 0-1, higher is better (1 = perfect)
    
    """
    STEG 6: Save Model & Scaler
    """
    with open(f'{MODEL_DIR}/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'{MODEL_DIR}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Model saved to ai_engine/models/")
```

### Hvorfor XGBoost?

**XGBoost (eXtreme Gradient Boosting)** er den mest brukte algoritmen i trading AI fordi:

1. **Ensemble Learning**: Kombinerer mange små decision trees
   ```
   Tree 1: RSI > 60 → BUY (confidence: 0.3)
   Tree 2: MA_10 > MA_50 → BUY (confidence: 0.2)
   Tree 3: sentiment > 0.5 → BUY (confidence: 0.25)
   ---
   Final: BUY (confidence: 0.75)
   ```

2. **Handles Non-linearity**: Fanger komplekse mønstre
   - Eksempel: "BUY when RSI < 30 AND volume > avg AND sentiment positive"

3. **Feature Importance**: Viser hvilke indikatorer som betyr mest
   ```python
   model.feature_importances_
   # Output:
   # RSI_14: 0.25        ← Viktigste feature
   # MA_10: 0.18
   # volume_ratio: 0.15
   # sentiment: 0.12
   # MACD: 0.10
   # ...
   ```

4. **Robust mot overfitting**: Regularisering innebygd

---

## 4️⃣ INFERENCE (Produksjon) - XGBAgent

### Fra: `ai_engine/agents/xgb_agent.py`

```python
class XGBAgent:
    """
    Real-time trading agent som bruker trained model til å generere signaler
    """
    
    def predict_for_symbol(self, ohlcv_data) -> Dict[str, Any]:
        """
        STEG 1: Feature Extraction
        """
        # Last 120 candles for symbol
        features = self._features_from_ohlcv(ohlcv_data)
        # → DataFrame med 25 features beregnet fra siste data
        
        """
        STEG 2: Preprocessing
        """
        X = features.select_dtypes(include=[np.number]).values
        X_scaled = self.scaler.transform(X)  # Samme scaling som i training
        
        """
        STEG 3: Model Prediction
        """
        if hasattr(self.model, 'predict_proba'):
            # Classification mode (BUY/SELL classes)
            proba = self.model.predict_proba(X_scaled)
            score = float(proba[0][1])  # Probability of BUY class
            
            # Decision thresholds
            if score > 0.55:
                action = "BUY"
            elif score < 0.45:
                action = "SELL"
            else:
                action = "HOLD"
        else:
            # Regression mode (predict Return directly)
            predicted_return = self.model.predict(X_scaled)[0]
            
            if predicted_return > 0.01:    # Expect >1% gain
                action = "BUY"
                score = min(0.99, float(predicted_return))
            elif predicted_return < -0.01:  # Expect >1% loss
                action = "SELL"
                score = min(0.99, float(abs(predicted_return)))
            else:
                action = "HOLD"
                score = 0.0
        
        """
        STEG 4: Risk Management Overlay
        """
        # Additional filters before executing
        if action == "BUY" and score < 0.6:
            action = "HOLD"  # Not confident enough
        
        return {
            "action": action,      # BUY/SELL/HOLD
            "score": score,        # 0.0 - 1.0 confidence
            "features": features   # For logging/debugging
        }
    
    def scan_symbols(self, symbol_ohlcv, top_n=10):
        """
        Scan multiple symbols og rank by volume
        
        Brukes for å finne beste trading opportunities blant mange coins
        """
        # 1. Sort symbols by recent volume
        volumes = [(s, get_volume(df)) for s, df in symbol_ohlcv.items()]
        volumes.sort(key=lambda x: x[1], reverse=True)
        
        # 2. Predict for top N highest volume symbols
        results = {}
        for symbol in volumes[:top_n]:
            prediction = self.predict_for_symbol(symbol_ohlcv[symbol])
            results[symbol] = prediction
        
        return results
        # Output:
        # {
        #   "BTCUSDT": {"action": "BUY", "score": 0.78},
        #   "ETHUSDT": {"action": "HOLD", "score": 0.52},
        #   "SOLUSDT": {"action": "SELL", "score": 0.65}
        # }
```

---

## 5️⃣ STRATEGI PRODUSERING (Strategy Generation)

### Automated Strategy Pipeline

```python
# Fra: backend/routes/ai.py

@router.post("/scan")
async def scan_symbols(symbols: List[str]):
    """
    Real-time scanning endpoint
    
    Kalles hver 3-5 minutter for å oppdatere trading signals
    """
    
    # 1. Load trained agent
    agent = make_default_agent()
    
    # 2. Fetch latest market data
    symbol_ohlcv = {}
    for symbol in symbols:
        candles = await fetch_binance_ohlcv(symbol, limit=120)
        sentiment = await fetch_twitter_sentiment(symbol)
        
        # Enrich candles with sentiment
        for candle in candles:
            candle['sentiment'] = sentiment['score']
        
        symbol_ohlcv[symbol] = candles
    
    # 3. Run AI predictions
    results = agent.scan_symbols(symbol_ohlcv, top_n=10)
    
    # 4. Generate trading signals
    signals = []
    for symbol, prediction in results.items():
        if prediction['action'] != 'HOLD' and prediction['score'] > 0.6:
            signal = {
                "symbol": symbol,
                "action": prediction['action'],
                "confidence": prediction['score'],
                "timestamp": datetime.now(timezone.utc),
                
                # Risk management parameters
                "price_target": calculate_target(symbol, prediction),
                "stop_loss": calculate_stop_loss(symbol, prediction),
                "position_size": calculate_position_size(prediction['score'])
            }
            signals.append(signal)
    
    return signals
```

### Signal til Handel (Execution Flow)

```python
# Fra: backend/utils/scheduler.py

async def _run_execution_cycle():
    """
    Kjører hver 30. minutt for å execute trading signals
    """
    
    # 1. Get latest signals from AI
    signals = await ai_routes.scan_symbols(["BTCUSDT", "ETHUSDT", ...])
    
    for signal in signals:
        # 2. Risk checks
        if not risk_guard.can_trade(signal):
            logger.info(f"Risk check failed for {signal['symbol']}")
            continue
        
        # 3. Position sizing
        account_balance = get_account_balance()
        risk_per_trade = account_balance * 0.02  # 2% risk per trade
        position_size = calculate_size(risk_per_trade, signal['stop_loss'])
        
        # 4. Execute trade
        if signal['action'] == 'BUY':
            order = await exchange.create_market_buy_order(
                symbol=signal['symbol'],
                amount=position_size
            )
        elif signal['action'] == 'SELL':
            order = await exchange.create_market_sell_order(
                symbol=signal['symbol'],
                amount=position_size
            )
        
        # 5. Log trade to database
        trade_log = TradeLog(
            symbol=signal['symbol'],
            side=signal['action'],
            qty=position_size,
            price=order['price'],
            ai_confidence=signal['confidence'],
            ai_reason=f"Model prediction score: {signal['score']:.2f}",
            model_used="XGBoost",
            timestamp=datetime.now(timezone.utc)
        )
        db.add(trade_log)
        db.commit()
        
        logger.info(f"✅ Executed {signal['action']} {position_size} {signal['symbol']}")
```

---

## 6️⃣ LIVE SYSTEM FLOW

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIVE TRADING CYCLE                        │
└─────────────────────────────────────────────────────────────────┘

Every 3 minutes:
  ┌──────────────────────────────────────────┐
  │ 1. MARKET DATA REFRESH                   │
  │    • Fetch latest candles from Binance   │
  │    • Fetch sentiment from Twitter        │
  │    • Fetch trending coins from CoinGecko │
  └──────────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────────┐
  │ 2. FEATURE ENGINEERING                   │
  │    • Calculate RSI, MACD, MA, etc.       │
  │    • Add sentiment features              │
  │    • Normalize with saved scaler         │
  └──────────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────────┐
  │ 3. AI PREDICTION                         │
  │    • Load XGBoost model                  │
  │    • Predict BUY/SELL/HOLD               │
  │    • Calculate confidence score          │
  └──────────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────────┐
  │ 4. SIGNAL GENERATION                     │
  │    • Filter by confidence threshold      │
  │    • Calculate position size             │
  │    • Set stop loss & take profit         │
  └──────────────────────────────────────────┘

Every 30 minutes:
  ┌──────────────────────────────────────────┐
  │ 5. TRADE EXECUTION                       │
  │    • Risk management checks              │
  │    • Execute orders via exchange API     │
  │    • Log trades to database              │
  │    • Update portfolio positions          │
  └──────────────────────────────────────────┘
                    ↓
  ┌──────────────────────────────────────────┐
  │ 6. MONITORING & FEEDBACK                 │
  │    • Track P&L per trade                 │
  │    • Update win rate statistics          │
  │    • Dashboard live updates via WebSocket│
  └──────────────────────────────────────────┘
```

---

## 📊 EKSEMPEL: Full Trading Decision

### Input Data (BTCUSDT kl 03:00)
```python
current_price = 43200
recent_candles = [
  # Last 120 candles (2 hours for 1min candles)
  {"open": 43000, "high": 43500, "low": 42800, "close": 43200, "volume": 1250}
  # ... 119 more
]
sentiment = {"score": 0.72, "positive": 0.72}  # Bullish
```

### Feature Engineering Output
```python
features = {
  # Technical Indicators
  "RSI_14": 68.5,          # Overbought territory (>70 = very overbought)
  "MA_10": 42950,          # Price above short-term MA (bullish)
  "MA_50": 42100,          # Price above long-term MA (strong uptrend)
  "MACD": 150,             # Positive MACD (momentum bullish)
  "ATR_14": 320,           # Average volatility
  "Bollinger_Upper": 44000,
  "Bollinger_Lower": 41500,
  
  # Volume Analysis
  "volume": 1250,
  "volume_ma_20": 1100,
  "volume_ratio": 1.14,    # Above average volume (strong move)
  
  # Sentiment
  "sentiment": 0.72,       # Strong positive sentiment
  "sentiment_ma_5": 0.68,  # Trending more positive
  
  # Price Action
  "high_low_ratio": 0.016,
  "close_open_ratio": 0.0047
}
```

### Model Prediction
```python
# XGBoost processes all features
X_scaled = scaler.transform([features])
predicted_return = model.predict(X_scaled)[0]

# Output: 0.0234 (2.34% expected gain)

# Decision logic
if predicted_return > 0.01:  # Threshold: >1% gain
    action = "BUY"
    confidence = 0.78  # Based on prediction strength
```

### Signal Generation
```python
signal = {
  "symbol": "BTCUSDT",
  "action": "BUY",
  "confidence": 0.78,
  "current_price": 43200,
  
  # Risk management
  "price_target": 44200,      # +2.3% target (predicted_return)
  "stop_loss": 42800,         # -0.9% stop (ATR based)
  "position_size": 0.023,     # BTC amount (2% account risk)
  
  "model_used": "XGBoost",
  "timestamp": "2025-11-11T03:00:00Z",
  
  # Supporting data
  "reasoning": "Strong uptrend (MA_10 > MA_50), positive momentum (MACD), "
               "bullish sentiment (0.72), above-average volume (1.14x)"
}
```

### Trade Execution
```python
# Risk checks pass ✅
# Execute market order
order = await exchange.create_market_buy_order(
    symbol="BTCUSDT",
    amount=0.023  # $993.60 at current price
)

# Log to database
TradeLog(
  symbol="BTCUSDT",
  side="BUY",
  qty=0.023,
  price=43200,
  value=993.60,
  ai_confidence=78.0,
  ai_reason="Model prediction score: 0.78",
  model_used="XGBoost",
  signal_strength=0.78,
  status="filled"
)
```

### Monitoring (30 minutes later)
```python
# Price moved to 43850 (+1.5% gain)
# Decision: Take partial profit or hold for target?

if current_price >= price_target * 0.7:  # 70% of target reached
    # Move stop loss to breakeven
    update_stop_loss(order_id, entry_price)
    
pnl = (43850 - 43200) * 0.023 = $14.95
win = True
```

---

## 🎯 NØKKEL KONSEPTER

### 1. Supervised Learning
Modellen lærer fra historiske data hvor vi vet utfallet:
```
Features (X)                    → Target (y)
[RSI=65, MA_10>MA_50, sent=0.7] → Return = +2.3% ✅ Correct prediction
[RSI=35, MA_10<MA_50, sent=0.2] → Return = -1.8% ✅ Correct prediction
```

### 2. Feature Importance
Ikke alle features er like viktige:
```
Most Important Features:
1. RSI_14 (25%)          ← Strongest predictor
2. MA_10_MA_50_cross (18%)
3. volume_ratio (15%)
4. sentiment (12%)
5. MACD (10%)

Least Important:
20. high_low_ratio (1%)
```

### 3. Ensemble Learning (XGBoost)
Kombinerer mange svake prediktorer til én sterk:
```
Tree 1: "If RSI > 60 → BUY"            (60% accurate)
Tree 2: "If MA_10 > MA_50 → BUY"       (55% accurate)
Tree 3: "If sentiment > 0.5 → BUY"     (52% accurate)
Tree 4: "If volume > avg → BUY"        (58% accurate)
...
Tree 100: Complex pattern              (61% accurate)

Ensemble: All trees combined → BUY     (73% accurate) ✅
```

### 4. Regularization (Overfitting Prevention)
```python
XGBRegressor(
    n_estimators=100,      # Not too many trees
    max_depth=5,           # Not too deep (prevent memorizing)
    learning_rate=0.1,     # Learn slowly
    subsample=0.8,         # Don't use all data each iteration
    colsample_bytree=0.8   # Don't use all features each tree
)
```

Dette forhindrer at modellen "memorerer" training data og kan generalisere til nye situasjoner.

---

## 🔄 KONTINUERLIG FORBEDRING

### Retraining Pipeline
```python
# Scheduled weekly retraining
async def retrain_model():
    # 1. Fetch last 30 days of data
    data = await fetch_historical_data(days=30)
    
    # 2. Add recent trades with actual outcomes
    for trade in recent_trades:
        data.append({
            "features": trade.features,
            "actual_return": trade.pnl / trade.value,
            "predicted_return": trade.ai_confidence
        })
    
    # 3. Retrain with updated data
    new_model = train_model(data)
    
    # 4. Validate performance
    if new_model.score > current_model.score:
        # Deploy new model
        save_model(new_model, "xgb_model.pkl")
        logger.info("✅ Model updated - Performance improved")
```

### Performance Tracking
```python
# Database stores every prediction vs actual outcome
analyze_model_performance()
# Output:
# Win Rate: 68.5% (target: >60%)
# Sharpe Ratio: 1.85 (target: >1.5)
# Max Drawdown: -8.2% (target: <10%)
# Average Return per Trade: +1.23%
```

---

## 📚 SAMMENDRAG

**Hele flyten:**

1. **Data Collection** → Hent market data + sentiment
2. **Feature Engineering** → Beregn 25+ tekniske indikatorer
3. **Training** (scikit-learn/XGBoost) → Lær mønstre fra historiske data
4. **Prediction** → Generer BUY/SELL/HOLD signals med confidence
5. **Risk Management** → Beregn position size, stop loss, target
6. **Execution** → Execute trades via exchange API
7. **Monitoring** → Track P&L, update statistics
8. **Retraining** → Forbedre modellen kontinuerlig

**Scikit-learn rolle:**
- StandardScaler: Data normalisering
- XGBoost/RandomForest: Prediksjon algoritme
- Metrics: Evaluering (MSE, R², accuracy)

**Styrker:**
- ✅ Data-driven decisions (ikke følelser)
- ✅ Fanger komplekse mønstre mennesker ikke ser
- ✅ Konsistent utførelse 24/7
- ✅ Continuous learning fra nye data

**Utfordringer:**
- ⚠️ Krever god historisk data
- ⚠️ Market conditions endrer seg (model drift)
- ⚠️ Ikke 100% accuracy - risk management critical
- ⚠️ Technology/API failures må håndteres
