# 🤖 AI FULL AUTONOMY MIGRATION PLAN

**Dato:** 3. februar 2026  
**Mål:** Fjerne ALLE hardkodede verdier og gi AI full kontroll over trading beslutninger

---

## 🎯 MÅLSETTINGER

### NÅVÆRENDE (Hardkodet)
```
❌ TOP10: Statisk liste med 10 symboler
❌ Leverage: 10x hardkodet i AI Agent fallback
❌ Margin ratio: Implisitt 20% minimum i Portfolio Gate
❌ Position size: Governor beregner basert på MAX_NOTIONAL_PER_TRADE_USDT=200
❌ Stop-loss: Governor/Risk Kernel beregner med hardkodede formler
❌ Take-profit: Governor/Risk Kernel beregner med hardkodede formler
❌ Harvest: Harvest Proposal Publisher med hardkodede R_net thresholds
❌ Kill score: Portfolio Heat Gate med hardkodet k_close_threshold=0.650
```

### ØNSKET (AI Autonomi)
```
✅ TOP10: AI velger beste 10 av 566 symboler dynamisk hver time
✅ Leverage: AI bestemmer per trade (1x - 125x)
✅ Margin ratio: AI bestemmer per trade (minimum 10%)
✅ Position size: AI bestemmer per trade basert på confidence
✅ Stop-loss: AI bestemmer per trade basert på volatilitet
✅ Take-profit: AI bestemmer per trade basert på opportunity
✅ Harvest: AI bestemmer når og hvordan stenge (kun 1 system!)
✅ Kill score: AI justerer thresholds dynamisk basert på marked
```

---

## 🔍 NÅVÆRENDE SYSTEM ANALYSE

### 1. HVEM BESTEMMER HVA NÅ?

#### 1.1 Symbol Selection (TOP10)
**Fil:** `scripts/generate_top10_universe.py`  
**Problem:** Hardkodet scoring formel
```python
score = (
    0.3 * volatility_score +      # Hardkodet vekter
    0.4 * trend_score +
    0.3 * probability_score
)

MIN_SIGMA = 0.005  # Hardkodet filter
MIN_TS = 0.3       # Hardkodet filter
MAX_SYMBOLS = 10   # Hardkodet limit
```

**Hvem bruker:** Intent Bridge (leser `quantum:cfg:universe:top10`)

---

#### 1.2 Leverage Selection
**Fil:** `microservices/ai_engine/service.py` (line 2077)  
**Problem:** Fallback til 10x
```python
leverage = 10.0  # Default fallback (hvis RL agent feiler)
```

**Andre steder:**
- `microservices/execution/config.py`: `MAX_LEVERAGE = 10`
- `microservices/exitbrain_v3_5/adaptive_leverage_config.py`: `LEVERAGE_TIER_1 = 10`

**Hvem bruker:** 
- AI Agent sender leverage i trade intent
- Intent Bridge videresender til apply.plan
- Intent Executor setter leverage via Binance API

---

#### 1.3 Position Sizing
**Fil:** `microservices/governor/main.py`  
**Problem:** Hardkodet max notional
```python
MAX_NOTIONAL_PER_TRADE_USDT = 200  # Hardkodet!
MAX_TOTAL_NOTIONAL_USDT = 2000     # Hardkodet!
```

**Hvem bruker:** Governor (men er stoppet nå!)

**Faktisk:** Intent Bridge og AI Agent bestemmer quantity direkte  
- AI Agent sender `size` eller `quantity` i intent
- Governor skulle ha validert, men kjører ikke

---

#### 1.4 Margin Ratio
**Fil:** `microservices/governor/main.py`  
**Problem:** Implisitt minimum (ikke eksplisitt kodet)
```python
# Ingen eksplisitt MIN_MARGIN_RATIO!
# Men Portfolio Gate P3.3 sjekker margin_ratio > threshold
```

**Beregning:**
```python
margin_ratio = (equity_usd / total_margin_used) * 100
# Hvis ratio < 20% → DENY permit (implisitt i Portfolio Gate logic)
```

---

#### 1.5 Stop-Loss & Take-Profit
**Fil:** `ai_engine/risk_kernel.py` (antagelse, må verifiseres)  
**Problem:** Hardkodet formler
```python
stop_loss = entry_price * (1 - stop_dist_pct)  # stop_dist_pct fra Risk Kernel
take_profit = entry_price * (1 + tp_dist_pct)  # tp_dist_pct fra Risk Kernel
```

**Hvem bruker:**
- AI Agent beregner og sender `stop_loss`, `take_profit` i intent
- Intent Executor plasserer stop-loss og take-profit ordre på Binance

---

#### 1.6 Harvest Decisions (KRITISK!)
**PROBLEM:** Flere systemer kan stenge posisjoner!

##### System 1: Harvest Proposal Publisher
**Fil:** `microservices/harvest_proposal_publisher/main.py`  
**Jobb:** Foreslår stenging basert på R_net, age, etc.
```python
harvest_output = compute_harvest_proposal(
    pos_snapshot=pos,
    market_state=ms,
    p1_proposal=p1,
    theta=theta  # Hardkodede thresholds!
)
# Output: "FULL_CLOSE_PROPOSED", "PARTIAL_CLOSE_30", "UPDATE_SL", "HOLD"
```

**Hardkodede verdier i HarvestTheta:**
```python
class HarvestTheta:
    r_tp_min: float = 2.0      # Minimum R for take-profit
    r_partial: float = 1.5     # R for partial close
    age_tp_sec: float = 3600   # Age threshold
    age_emergency_sec: float = 7200
    # ... mange flere
```

##### System 2: Portfolio Heat Gate
**Fil:** `microservices/portfolio_heat_gate/main.py`  
**Jobb:** Vurderer og nedgraderer harvest forslag
```python
kill_score = regime_flip + ts_drop + pnl_factor

if kill_score >= k_close_threshold:  # 0.650 hardkodet!
    action = "UPDATE_SL"  # Nedgrader CLOSE til UPDATE_SL
else:
    action = harvest_action  # Godkjenn original
```

##### System 3: Apply Layer Action Normalizer
**Fil:** `microservices/apply_layer/main.py`  
**Jobb:** Normaliserer harvest actions til executable commands
```python
if action == "FULL_CLOSE_PROPOSED":
    normalized = "CLOSE"
elif action == "PARTIAL_CLOSE_30":
    normalized = "REDUCE_30"
elif action == "UPDATE_SL":
    normalized = "UPDATE_SL"
elif action == "UNKNOWN":
    normalized = "UPDATE_SL"  # Fallback!
```

##### System 4: Binance Stop-Loss/Take-Profit (Automatisk!)
**Jobb:** Binance selv stenger ved stop-loss eller take-profit pris
```python
# Når posisjon åpnes:
Intent Executor → Binance STOP_MARKET ordre (stop_loss pris)
Intent Executor → Binance TAKE_PROFIT_MARKET ordre (take_profit pris)

# Binance overvåker:
if mark_price <= stop_loss:
    Binance → Automatisk SELL (reduceOnly=true)
elif mark_price >= take_profit:
    Binance → Automatisk SELL (reduceOnly=true)
```

**KONKLUSJON:**
- 🔴 **4 SEPARATE SYSTEMER** kan stenge posisjoner!
- 🔴 Harvest Publisher, Heat Gate, Apply Layer, Binance auto-orders
- 🔴 Koordinering via Redis hashes og calibrated flag
- ⚠️ **Komplekst!** Vanskelig å gi AI full kontroll

---

### 1.7 Kill Score Thresholds
**Fil:** `microservices/portfolio_heat_gate/main.py`  
**Problem:** Hardkodet threshold
```python
K_CLOSE_THRESHOLD = 0.650  # Hardkodet!

if kill_score >= K_CLOSE_THRESHOLD:
    # Blokkerer stenging
```

**Fil:** `microservices/governor/main.py`  
**Også hardkodet:**
```python
KILL_SCORE_CRITICAL = 0.8
KILL_SCORE_OPEN_THRESHOLD = 0.85
KILL_SCORE_CLOSE_THRESHOLD = 0.65
```

---

## 🚨 PROBLEMER IDENTIFISERT

### Problem 1: TOP10 Hardkodet Scoring
```
Nåværende: generate_top10_universe.py med hardkodet formel
Ønsket: AI velger beste 10 basert på egen modell
```

### Problem 2: Leverage Fallback
```
Nåværende: AI Agent fallback til 10x hvis RL feiler
Ønsket: Ingen fallback - AI må ALLTID bestemme
```

### Problem 3: Position Sizing Begrenset
```
Nåværende: MAX_NOTIONAL_PER_TRADE_USDT = 200 (Governor)
Ønsket: AI bestemmer size basert på confidence, ingen hard cap
```

### Problem 4: Margin Ratio Implisitt
```
Nåværende: Portfolio Gate implisitt minimum ~20%
Ønsket: AI bestemmer minimum per trade (men > 10% safety)
```

### Problem 5: Harvest Multi-System Chaos
```
Nåværende: 4 systemer kan stenge (Harvest, Heat Gate, Apply, Binance)
Ønsket: KUN AI bestemmer (via ett system med AI-generert formel)
```

### Problem 6: Kill Score Hardkodet
```
Nåværende: k_close_threshold = 0.650 hardkodet
Ønsket: AI justerer dynamisk basert på markedstilstand
```

---

## 🔧 MIGRASJONSPLAN

### FASE 1: AI Symbol Selection (TOP10 → AI TOP10)

**Mål:** AI velger beste 10 symboler dynamisk

#### Løsning A: AI Model for Symbol Selection
```python
# Ny fil: microservices/ai_agent/symbol_selector.py

class AISymbolSelector:
    def __init__(self, model_path: str):
        self.model = load_ai_model(model_path)
    
    def select_top_symbols(
        self, 
        all_symbols: List[str], 
        market_states: Dict[str, MarketState],
        max_symbols: int = 10
    ) -> List[str]:
        """AI bestemmer beste symboler basert på full analyse"""
        
        # Feature extraction per symbol
        features = []
        for symbol in all_symbols:
            ms = market_states.get(symbol)
            if not ms:
                continue
            
            feature_vector = [
                ms.sigma,           # Volatilitet
                ms.ts,              # Trend styrke
                ms.p_trend,         # Trend sannsynlighet
                ms.volume_24h,      # Volum
                ms.liquidity_score, # Likviditet
                ms.correlation,     # Korrelasjon til andre
                # ... AI bestemmer hvilke features!
            ]
            features.append((symbol, feature_vector))
        
        # AI scoring (ikke hardkodet formel!)
        scores = self.model.predict_symbol_scores(features)
        
        # Sorter og velg top N
        ranked = sorted(zip(all_symbols, scores), key=lambda x: -x[1])
        top_symbols = [sym for sym, score in ranked[:max_symbols]]
        
        return top_symbols
```

#### Alternativ B: RL-basert Symbol Selection
```python
# Bruk RL Agent til å velge symboler
# State: Current portfolio + all market states
# Action: Select 10 symbols
# Reward: Portfolio PnL over next N hours

class RLSymbolSelector:
    def select_top_symbols(
        self,
        all_symbols: List[str],
        market_states: Dict[str, MarketState],
        current_portfolio: Dict[str, Position],
        max_symbols: int = 10
    ) -> List[str]:
        """RL agent velger optimale symboler"""
        
        state = self._build_state(all_symbols, market_states, current_portfolio)
        action = self.rl_agent.act(state)  # RL bestemmer!
        top_symbols = self._decode_action(action, all_symbols)
        
        return top_symbols
```

#### Deployment:
```bash
# 1. Tren AI/RL modell for symbol selection
# 2. Deploy modell til /home/qt/quantum_trader/models/symbol_selector.pkl
# 3. Oppdater generate_top10_universe.py til å bruke AI model
# 4. Slett hardkodede vekter (0.3, 0.4, 0.3)
```

---

### FASE 2: AI Leverage & Sizing (Full Autonomi)

**Mål:** AI bestemmer leverage og size per trade uten begrensninger (utenom safety minimum)

#### Endringer:

**1. Fjern Leverage Fallback**
```python
# Fil: microservices/ai_engine/service.py (line 2077)

# FJERN:
leverage = 10.0  # Default fallback

# ERSTATT MED:
if not leverage or leverage < 1:
    logger.error(f"AI failed to provide leverage for {symbol}, SKIP trade!")
    return None  # Ikke trade hvis AI ikke bestemmer!
```

**2. Fjern MAX_NOTIONAL begrensninger**
```python
# Fil: microservices/governor/main.py

# FJERN:
MAX_NOTIONAL_PER_TRADE_USDT = 200
MAX_TOTAL_NOTIONAL_USDT = 2000

# ERSTATT MED:
# Kun safety checks:
MIN_MARGIN_RATIO_SAFETY = 10.0  # AI kan ikke gå under 10%
MAX_LEVERAGE_SAFETY = 125       # Binance limit

if margin_ratio < MIN_MARGIN_RATIO_SAFETY:
    logger.warning(f"Trade would violate 10% margin safety, DENY")
    return False

if leverage > MAX_LEVERAGE_SAFETY:
    logger.warning(f"Leverage {leverage}x exceeds Binance max, cap to 125x")
    leverage = 125
```

**3. AI bestemmer size basert på confidence**
```python
# Fil: microservices/ai_agent/service.py

def generate_trade_intent(self, symbol: str, signal: Signal) -> TradeIntent:
    """AI bestemmer ALT"""
    
    # AI model output
    ai_decision = self.model.predict(
        symbol=symbol,
        market_state=self.get_market_state(symbol),
        portfolio_state=self.get_portfolio_state()
    )
    
    # AI bestemmer direkte:
    leverage = ai_decision.leverage           # AI output (1-125)
    position_size_usd = ai_decision.size_usd  # AI output (basert på confidence)
    stop_loss = ai_decision.stop_loss         # AI output (basert på volatilitet)
    take_profit = ai_decision.take_profit     # AI output (basert på opportunity)
    
    # INGEN hardkodede verdier!
    # INGEN formler!
    # AI har full autonomi
    
    return TradeIntent(
        symbol=symbol,
        action="BUY" or "SELL",
        size=position_size_usd,
        leverage=leverage,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=ai_decision.confidence  # Metadata for logging
    )
```

---

### FASE 3: AI Harvest Control (Single System)

**Mål:** KUN AI bestemmer når og hvordan stenge posisjoner

#### Problem: 4 Systemer Nå
```
1. Harvest Proposal Publisher (foreslår)
2. Portfolio Heat Gate (vurderer)
3. Apply Layer (normaliserer)
4. Binance Auto-Orders (utfører)
```

#### Løsning: Konsolider til AI-Driven Harvest

**Alternativ A: Deaktiver Alt Unntatt AI**
```bash
# 1. Stopp Harvest Proposal Publisher
systemctl stop quantum-harvest-proposal-publisher
systemctl disable quantum-harvest-proposal-publisher

# 2. Stopp Portfolio Heat Gate
systemctl stop quantum-portfolio-heat-gate
systemctl disable quantum-portfolio-heat-gate

# 3. Apply Layer: Kun prosesser AI-genererte close intents
# (Ikke les fra harvest proposals)

# 4. AI Agent: Overvåk posisjoner og generer CLOSE intents
# Samme flow som OPEN, men action="SELL" med reduceOnly=true
```

**Alternativ B: AI Styrer Harvest Formulas**
```python
# Ny fil: microservices/ai_harvest_controller/main.py

class AIHarvestController:
    """AI bestemmer harvest beslutninger uten hardkodede formler"""
    
    def __init__(self, model_path: str):
        self.model = load_ai_harvest_model(model_path)
    
    def should_close_position(
        self,
        position: Position,
        market_state: MarketState,
        portfolio_state: PortfolioState
    ) -> HarvestDecision:
        """AI analyserer og bestemmer"""
        
        # Feature extraction
        features = {
            "unrealized_pnl": position.unrealized_pnl,
            "R_net": position.unrealized_pnl / position.initial_risk,
            "age_sec": position.age_sec,
            "sigma": market_state.sigma,
            "ts": market_state.ts,
            "trend_reversal": market_state.detect_reversal(),
            "portfolio_exposure": portfolio_state.exposure_pct,
            "portfolio_pnl": portfolio_state.total_pnl,
            # ... AI bestemmer hvilke features!
        }
        
        # AI beslutning (INGEN hardkodede thresholds!)
        decision = self.model.predict_harvest_action(features)
        
        return HarvestDecision(
            action=decision.action,  # "FULL_CLOSE", "PARTIAL_CLOSE", "HOLD", "UPDATE_SL"
            confidence=decision.confidence,
            reason=decision.explanation  # AI forklarer hvorfor
        )
```

**Deployment:**
```bash
# 1. Tren AI harvest model på historiske data
# 2. Deploy til /home/qt/quantum_trader/models/ai_harvest.pkl
# 3. Start ai_harvest_controller service
# 4. Deaktiver gamle harvest services
```

---

### FASE 4: AI Dynamic Kill Score

**Mål:** AI justerer kill score threshold dynamisk

#### Nåværende Problem:
```python
# Hardkodet threshold
k_close_threshold = 0.650
```

#### Løsning: AI Adaptive Threshold
```python
# Fil: microservices/portfolio_heat_gate/main.py

class AIAdaptiveKillScore:
    def __init__(self, model_path: str):
        self.model = load_ai_model(model_path)
    
    def get_adaptive_threshold(
        self,
        portfolio_state: PortfolioState,
        market_regime: str,
        recent_volatility: float
    ) -> float:
        """AI bestemmer threshold basert på kontekst"""
        
        features = {
            "portfolio_exposure": portfolio_state.exposure_pct,
            "portfolio_pnl": portfolio_state.total_pnl,
            "open_positions": portfolio_state.num_positions,
            "market_regime": market_regime,  # "trending", "ranging", "volatile"
            "vix_equivalent": recent_volatility,
            "hour_of_day": datetime.now().hour,  # Market timing
        }
        
        # AI bestemmer threshold (ikke 0.650!)
        threshold = self.model.predict_threshold(features)
        
        # Safety bounds (AI kan ikke gå for ekstrem)
        threshold = max(0.3, min(0.9, threshold))  # 0.3 - 0.9 range
        
        return threshold
```

---

## 📋 IMPLEMENTATION CHECKLIST

### ✅ FASE 1: AI Symbol Selection
- [ ] Tren AI/RL modell for symbol selection
- [ ] Implementer `AISymbolSelector` klasse
- [ ] Integrer i `generate_top10_universe.py`
- [ ] Fjern hardkodede vekter (0.3, 0.4, 0.3)
- [ ] Fjern hardkodede filters (MIN_SIGMA, MIN_TS)
- [ ] Test: AI velger forskjellige symboler basert på marked
- [ ] Deploy til produks

### ✅ FASE 2: AI Leverage & Sizing
- [ ] Fjern leverage fallback i `ai_engine/service.py`
- [ ] Fjern MAX_NOTIONAL i `governor/main.py`
- [ ] Implementer MIN_MARGIN_RATIO_SAFETY = 10%
- [ ] AI model output: leverage, size, sl, tp
- [ ] Test: AI bruker forskjellig leverage per trade
- [ ] Test: AI bruker forskjellig size basert på confidence
- [ ] Verifiser margin ratio aldri < 10%
- [ ] Deploy til produks

### ✅ FASE 3: AI Harvest Control
- [ ] Bestem strategi: Alternativ A (deaktiver alt) eller B (AI formulas)
- [ ] Hvis A: Stopp Harvest Publisher, Heat Gate
- [ ] Hvis B: Tren AI harvest model, deploy service
- [ ] Verifiser: KUN AI genererer CLOSE intents
- [ ] Fjern hardkodede HarvestTheta verdier
- [ ] Test: AI stenger ved riktige tidspunkt (backtest)
- [ ] Deploy til produks

### ✅ FASE 4: AI Dynamic Kill Score
- [ ] Tren AI modell for adaptive threshold
- [ ] Implementer `AIAdaptiveKillScore` klasse
- [ ] Integrer i Portfolio Heat Gate
- [ ] Fjern hardkodet k_close_threshold = 0.650
- [ ] Test: Threshold varierer med markedstilstand
- [ ] Deploy til produks

### ✅ FASE 5: Safety & Monitoring
- [ ] Implementer hard safety limits:
  - MIN_MARGIN_RATIO_SAFETY = 10%
  - MAX_LEVERAGE_SAFETY = 125
  - MAX_POSITION_SIZE_SAFETY = $10,000 (eller 50% av kapital)
- [ ] Logging: AI decisions for audit
- [ ] Alerting: Hvis AI genererer ekstreme verdier
- [ ] Dashboard: Vise AI-beslutninger i real-time
- [ ] Backtesting: Sammenlign AI vs hardkodet
- [ ] Deploy til produks

---

## 🚨 SAFETY GUIDELINES

### Hard Limits (IKKE AI-kontrollert)

```python
# DISSE ER HARDKODET FOR SIKKERHET:

MIN_MARGIN_RATIO = 10.0      # AI kan ikke gå under
MAX_LEVERAGE = 125           # Binance limit
MAX_POSITION_SIZE_PCT = 50   # Maks 50% av kapital per posisjon
MAX_TOTAL_EXPOSURE_PCT = 200 # Maks 200% (2x leverage average)

# Hvis AI foreslår noe som bryter disse:
# → Blokkeres automatisk
# → Logg warning
# → AI får negativ reward (for læring)
```

### AI Autonomi Zone

```python
# DISSE ER AI-KONTROLLERT:

leverage: 1 - 125            # AI bestemmer (innenfor safety range)
position_size: 0 - 50% cap   # AI bestemmer (innenfor safety range)
stop_loss: any               # AI bestemmer (anbefalt < 10% fra entry)
take_profit: any             # AI bestemmer
harvest_timing: any          # AI bestemmer
symbol_selection: any 10     # AI bestemmer
kill_score_threshold: 0.3-0.9 # AI bestemmer (innenfor safety range)
```

---

## 📊 TESTING PLAN

### 1. Backtesting
```bash
# Test AI decisions på historisk data
python backtest_ai_autonomy.py \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --capital 10000 \
  --mode ai_full_autonomy

# Sammenlign med hardkodet:
python backtest_comparison.py \
  --mode-a hardcoded \
  --mode-b ai_autonomy \
  --metrics sharpe,drawdown,profit
```

### 2. Paper Trading
```bash
# Deploy til testnet først
QUANTUM_ENV=testnet systemctl start quantum-ai-agent

# Overvåk i 1 uke:
# - AI leverage decisions
# - AI size decisions
# - AI harvest decisions
# - Margin ratio (skal aldri < 10%)
```

### 3. Gradvis Rollout
```bash
# Dag 1-7: 10% av kapital med AI autonomi
# Dag 8-14: 25% av kapital
# Dag 15-21: 50% av kapital
# Dag 22+: 100% av kapital (hvis metrics gode)
```

---

## 🎯 SUCCESS CRITERIA

### Minimum Requirements
- ✅ AI velger 10 symboler dynamisk (ikke hardkodet)
- ✅ AI bestemmer leverage per trade (1-125x)
- ✅ AI bestemmer size per trade (basert på confidence)
- ✅ Margin ratio ALDRI < 10% (safety enforced)
- ✅ KUN 1 system for harvest (AI-drevet)
- ✅ Ingen hardkodede formler i kritisk beslutningslogikk

### Performance Targets
- Sharpe ratio > 2.0 (vs 1.5 med hardkodet)
- Max drawdown < 15% (vs 20% med hardkodet)
- Win rate > 60% (vs 50% med hardkodet)
- Realized PnL > +30% per år

### Operational Targets
- AI uptime > 99.9%
- Decision latency < 500ms
- Zero margin calls (10% safety enforced)
- Audit trail for alle AI decisions

---

## 📝 DOKUMENTASJON UPDATES

Når ferdig, oppdater:
- `SYSTEM_OVERSIKT_NORSK.md` - Ny AI autonomi seksjon
- `SYSTEM_ARCHITECTURE_COMPLETE_ANALYSIS.md` - Teknisk deep-dive
- `AI_AGENT_QUICK_START.md` - Nye AI capabilities
- `00_START_HERE_PRODUCTION_HYGIENE.md` - Safety guidelines

---

**Status:** 🚧 PLANLEGGINGSFASE  
**Neste steg:** Review plan med bruker, start FASE 1 implementation  
**Estimert tid:** 4-6 uker for full deployment
