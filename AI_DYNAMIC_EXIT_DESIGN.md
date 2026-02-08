# AI-DREVET DYNAMISK EXIT SYSTEM - DESIGN DOK

**Status:** Design Phase  
**Dato:** 2026-02-07  
**Mål:** Erstatte hardkodede R-levels (25%, 50%, 75%) med AI real-time beslutninger

---

## 1. OMSTENDIGHETER AI SKAL VURDERE

### 🎯 Position Growth Rate (R-level momentum)
```
HVIS R går: 1.0 → 2.0 → 4.0 raskt (strong momentum)
  → AI beslutter: HOLD (la det løpe)
  
HVIS R går: 4.0 → 4.1 → 4.0 sakte (svakt momentum)
  → AI beslutter: PARTIAL 60% (ta profitt før reversering)
```

**Datapunkter:**
- R_net historikk (siste 10 målinger)
- R acceleration (dR/dt)
- Peak R vs current R

### 📈 Market Regime Status
```
HVIS entry regime = TRENDING og nå fortsatt TRENDING
  → AI beslutter: HOLD (momentum intakt)
  
HVIS entry regime = TRENDING og nå = RANGING
  → AI beslutter: PARTIAL 50% (regime endret)
  
HVIS entry regime = TRENDING og nå = VOLATILE_REVERSAL
  → AI beslutter: CLOSE 100% (risiko flip)
```

**Datapunkter:**
- Current regime (fra RegimeDetector)
- Entry regime (lagret ved åpning)
- Regime stability score

### 📊 Volatility Changes (ATR momentum)
```
HVIS ATR expanding (høyere highs, høyere lows)
  → AI beslutter: HOLD (trending fortsetter)
  
HVIS ATR contracting (range tightening)
  → AI beslutter: PARTIAL 40% (momentum svekkes)
  
HVIS ATR spike (sudden volatility jump)
  → AI beslutter: CLOSE 100% (panic/uro)
```

**Datapunkter:**
- ATR 14-period
- ATR gradient (expanding/contracting)
- Volatility percentile (vs 30-day history)

### ⏰ Time in Position
```
HVIS fresh position (<1h) og R>2
  → AI beslutter: HOLD (gi det rom)
  
HVIS aging position (3-6h) og R=2-4
  → AI beslutter: PARTIAL 30% (start take profit)
  
HVIS old position (>12h) og R<6
  → AI beslutter: CLOSE 100% (rot kapital)
```

**Datapunkter:**
- Position age_sec
- Avg holding time for profitable trades (historical)

### 🎲 Ensemble Confidence
```
HVIS alle 4 modeller enige + high confidence
  → AI beslutter: HOLD (sterkt signal)
  
HVIS modeller begynt å diverge (2-2 split)
  → AI beslutter: PARTIAL 50% (usikkerhet)
  
HVIS modeller flipped (nå motsatt retning)
  → AI beslutter: CLOSE 100% (signal reversert)
```

**Datapunkter:**
- Current ensemble confidence (0-1)
- Entry ensemble confidence (lagret)
- Model agreement score

### 📦 Portfolio Correlation & Concentration
```
HVIS position er 50% av total portfolio
  → AI beslutter: PARTIAL 60% (reduser konsentrasjon)
  
HVIS mange korrelerte posisjoner (BTC+ETH+SOL alle LONG)
  → AI beslutter: PARTIAL 40% (diversifiser)
```

**Datapunkter:**
- Position value / total portfolio value
- Correlation between open positions

### 💹 Price Action & Momentum
```
HVIS strong momentum candles (grønne bodier, økende volume)
  → AI beslutter: HOLD (momentum fortsetter)
  
HVIS reversal signals (doji, shooting star, bearish engulfing)
  → AI beslutter: PARTIAL/CLOSE (reversal risiko)
  
HVIS consolidation (små bodier, lav volume)
  → AI beslutter: PARTIAL 50% (momentum død)
```

**Datapunkter:**
- Recent candle patterns
- Volume profile
- RSI divergence

---

## 2. ARKITEKTUR

### **Komponent A: AI Engine - evaluate_exit() API**

**Ny endpoint:** `/api/v1/evaluate-exit`

**Input:**
```python
{
    "symbol": "BTCUSDT",
    "side": "LONG",
    "entry_price": 45000.0,
    "current_price": 47250.0,
    "position_qty": 0.01,
    "entry_timestamp": 1738876800,
    "age_sec": 3600,
    "R_net": 2.5,
    "R_history": [1.0, 1.5, 2.0, 2.3, 2.5],  # siste 5 målinger
    "entry_regime": "TRENDING",
    "entry_confidence": 0.82,
    "stop_loss": 44500.0,
    "take_profit": 48000.0
}
```

**AI Logic (i service.py):**
```python
async def evaluate_exit(self, position_data: dict) -> dict:
    """
    AI-driven exit evaluation
    
    Vurderer alle faktorer og returnerer dynamisk exit beslutning
    """
    symbol = position_data["symbol"]
    R_net = position_data["R_net"]
    age_sec = position_data["age_sec"]
    
    # 1. REGIME CHECK
    current_regime = self.regime_detector.get_regime(symbol)
    entry_regime = position_data.get("entry_regime", "TRENDING")
    regime_changed = (current_regime != entry_regime)
    
    # 2. VOLATILITY CHECK
    vse_data = await self.vse.get_structure(symbol)
    atr_gradient = vse_data.get("atr_gradient", 0)  # positive = expanding
    vol_expanding = (atr_gradient > 0.1)
    
    # 3. ENSEMBLE RE-EVALUATION
    current_signal = await self._get_ensemble_signal(symbol)
    entry_confidence = position_data.get("entry_confidence", 0.7)
    confidence_degraded = (current_signal.confidence < entry_confidence - 0.15)
    
    # 4. R-MOMENTUM CHECK
    R_history = position_data.get("R_history", [])
    if len(R_history) >= 3:
        R_acceleration = (R_history[-1] - R_history[-3]) / 2
        momentum_strong = (R_acceleration > 0.3)
    else:
        momentum_strong = False
    
    # 5. TIME DECAY
    age_hours = age_sec / 3600
    position_old = (age_hours > 6)
    
    # 6. SCORING & DECISION
    hold_score = 0
    exit_score = 0
    
    # Positive factors (HOLD)
    if not regime_changed: hold_score += 3
    if vol_expanding: hold_score += 2
    if momentum_strong: hold_score += 3
    if not confidence_degraded: hold_score += 2
    if age_hours < 2 and R_net > 1: hold_score += 2  # let it breathe
    
    # Negative factors (EXIT)
    if regime_changed: exit_score += 4
    if not vol_expanding: exit_score += 2  # contracting
    if confidence_degraded: exit_score += 3
    if position_old and R_net < 5: exit_score += 2  # rotate capital
    if R_net > 6: exit_score += 1  # take profit at extreme levels
    
    # DECISION LOGIC
    if exit_score > hold_score + 3:
        # Strong exit signal
        if regime_changed or confidence_degraded:
            action = "CLOSE"
            percentage = 1.0  # 100%
            reason = "regime_flip+confidence_lost"
        else:
            action = "PARTIAL_CLOSE"
            # Dynamic percentage based on exit_score
            percentage = min(0.75, (exit_score / 12))  # 0.3 - 0.75
            reason = f"exit_score={exit_score}_conditions_weakening"
    
    elif exit_score > hold_score:
        # Moderate exit signal
        action = "PARTIAL_CLOSE"
        percentage = 0.3 + (R_net / 20)  # 30-50% based on R
        reason = f"profit_taking_R={R_net:.1f}"
    
    else:
        # Hold position
        action = "HOLD"
        percentage = 0.0
        reason = f"momentum_strong_hold_score={hold_score}"
    
    return {
        "action": action,
        "percentage": round(percentage, 2),
        "reason": reason,
        "factors": {
            "regime_changed": regime_changed,
            "vol_expanding": vol_expanding,
            "momentum_strong": momentum_strong,
            "confidence_degraded": confidence_degraded,
            "position_old": position_old,
            "hold_score": hold_score,
            "exit_score": exit_score
        },
        "current_regime": current_regime.value,
        "timestamp": int(time.time())
    }
```

**Output:**
```json
{
  "action": "PARTIAL_CLOSE",
  "percentage": 0.37,
  "reason": "regime_weakening+volatility_contracting",
  "factors": {
    "regime_changed": false,
    "vol_expanding": false,
    "momentum_strong": false,
    "confidence_degraded": true,
    "position_old": false,
    "hold_score": 5,
    "exit_score": 9
  },
  "current_regime": "RANGING",
  "timestamp": 1738880400
}
```

---

### **Komponent B: Harvest Brain Integration**

**Modifiser evaluate() funksjon:**
```python
def evaluate(self, position: Position) -> List[HarvestIntent]:
    """Evaluate position - AI-driven exits"""
    intents = []
    
    # ... existing SL check ...
    
    # 🔥 AI-DRIVEN EXIT EVALUATION (replaces hardcoded R-levels)
    if position.R_net >= self.config.min_r:
        # Call AI Engine for dynamic exit decision
        ai_decision = await self._get_ai_exit_decision(position)
        
        if ai_decision["action"] in ["PARTIAL_CLOSE", "CLOSE"]:
            percentage = ai_decision["percentage"]
            qty = position.qty * percentage
            
            exit_side = 'SELL' if position.side == 'LONG' else 'BUY'
            
            intent = HarvestIntent(
                intent_type=f'AI_EXIT_{int(percentage*100)}PCT',
                symbol=position.symbol,
                side=exit_side,
                qty=qty,
                reason=f'[AI] {ai_decision["reason"]} (R={position.R_net:.2f})',
                r_level=position.R_net,
                unrealized_pnl=position.unrealized_pnl,
                correlation_id=f"ai:exit:{position.symbol}:{int(time.time())}",
                trace_id=f"ai:{position.symbol}:exit",
                dry_run=(self.config.harvest_mode == 'shadow'),
                ai_factors=ai_decision["factors"]  # for logging
            )
            intents.append(intent)
            
            logger.info(
                f"[AI-EXIT] {position.symbol} {int(percentage*100)}% @ R={position.R_net:.2f} | "
                f"Reason: {ai_decision['reason']} | "
                f"Scores: hold={ai_decision['factors']['hold_score']} "
                f"exit={ai_decision['factors']['exit_score']}"
            )
    
    return intents

async def _get_ai_exit_decision(self, position: Position) -> dict:
    """Query AI Engine for exit decision"""
    payload = {
        "symbol": position.symbol,
        "side": position.side,
        "entry_price": position.entry_price,
        "current_price": position.current_price,
        "position_qty": position.qty,
        "entry_timestamp": position.entry_timestamp,
        "age_sec": position.age_sec,
        "R_net": position.R_net,
        "R_history": self._get_R_history(position.symbol),
        "entry_regime": position.entry_regime,
        "entry_confidence": position.entry_confidence,
        "stop_loss": position.stop_loss,
        "take_profit": position.take_profit
    }
    
    try:
        response = await self.http_client.post(
            "http://127.0.0.1:8001/api/v1/evaluate-exit",
            json=payload,
            timeout=2.0
        )
        return response.json()
    except Exception as e:
        logger.error(f"[AI-EXIT] Failed to get AI decision: {e}")
        # Fallback to conservative partial close
        return {
            "action": "PARTIAL_CLOSE",
            "percentage": 0.25,
            "reason": "ai_unavailable_fallback",
            "factors": {}
        }
```

---

## 3. OVERVÅKNING & DASHBOARD

### **Redis Event Stream**
AI Engine publiserer hver exit-beslutning til Redis:
```python
# I evaluate_exit() etter beslutning:
self.redis.xadd(
    "quantum:stream:ai.exit.decision",
    {
        "symbol": symbol,
        "action": action,
        "percentage": percentage,
        "reason": reason,
        "R_net": R_net,
        "hold_score": hold_score,
        "exit_score": exit_score,
        "regime": current_regime,
        "timestamp": timestamp
    }
)
```

### **Dashboard Integration (quantumfond.com)**

**Frontend må vise:**

1. **Real-time Exit Decisions Table:**
```
| Time     | Symbol   | R-level | Action         | %    | Reason                              | Hold Score | Exit Score |
|----------|----------|---------|----------------|------|-------------------------------------|------------|------------|
| 01:23:45 | BTCUSDT  | 2.5R    | PARTIAL_CLOSE  | 37%  | regime_weakening+vol_contracting    | 5          | 9          |
| 01:22:10 | ETHUSDT  | 4.2R    | PARTIAL_CLOSE  | 52%  | profit_taking_R=4.2                 | 6          | 8          |
| 01:20:33 | SOLUSDT  | 1.8R    | HOLD           | 0%   | momentum_strong_hold_score=10       | 10         | 3          |
```

2. **Exit Decision Distribution Chart:**
```
Percentage Used:
[====== 30-40% ======] 35 exits
[========= 40-50% =========] 42 exits
[==== 50-60% ====] 28 exits
[== 60-70% ==] 15 exits
[= 70-80% =] 8 exits
```

3. **Reason Categories Breakdown:**
```
regime_weakening: 45%
profit_taking: 30%
confidence_degraded: 15%
volatility_spike: 7%
position_old: 3%
```

4. **Performance Attribution:**
```
Exit Timing Analysis:
✅ Optimal exits (captured >80% of peak): 62%
⚠️  Early exits (missed >20% profit): 25%
❌ Late exits (gave back >10% profit): 13%

AI Exit vs Hardcoded Comparison:
AI Dynamic:     +15.2% monthly return
Hardcoded R:    +11.8% monthly return
Improvement:    +3.4% (28% better)
```

### **Grafana Metrics**

**Metrics å tracke:**
```python
# Exit decision metrics
ai_exit_decisions_total{action="PARTIAL_CLOSE|CLOSE|HOLD"}
ai_exit_percentage_avg
ai_exit_hold_score_avg
ai_exit_exit_score_avg

# Performance metrics
ai_exit_profit_captured_pct  # hvor mye av peak profit
ai_exit_timing_score  # optimal/early/late distribution
ai_exit_vs_hardcoded_diff_pct  # performance comparison

# Reason distribution
ai_exit_reason_total{reason="regime_weakening|profit_taking|..."}

# Regime correlation
ai_exit_by_regime_total{regime="TRENDING|RANGING|VOLATILE"}
```

**Grafana Dashboards:**

**Panel 1: Exit Decision Stream (Real-time)**
- Live feed av siste 20 exit decisions
- Viser symbol, R-level, %, reason

**Panel 2: Hold vs Exit Score Distribution**
- Scatter plot: hold_score (x) vs exit_score (y)
- Farge: grønn (HOLD), gul (PARTIAL), rød (CLOSE)

**Panel 3: Performance Attribution**
- Line chart: cumulative return (AI exits vs Hardcoded)
- Distance between lines = improvement

**Panel 4: Reason Heatmap**
- Y-axis: reasons (regime_weakening, profit_taking, etc.)
- X-axis: time (hourly buckets)
- Color intensity: count of exits

---

## 4. TESTING & ROLLOUT

### **Phase 1: Shadow Mode (1 uke)**
```python
HARVEST_BRAIN_AI_MODE = "shadow"
```
- AI evaluerer og logger beslutninger
- Men IKKE eksekuterer
- Sammenligner AI vs hardcoded decisions
- Manual review av resultatene

### **Phase 2: Hybrid Mode (1 uke)**
```python
HARVEST_BRAIN_AI_MODE = "hybrid"
AI_EXIT_PERCENTAGE = 0.5  # 50% av exits bruker AI
```
- 50% av posisjoner bruker AI exits
- 50% bruker hardcoded R-levels
- A/B testing → måle performance difference

### **Phase 3: Full AI Mode**
```python
HARVEST_BRAIN_AI_MODE = "full"
```
- Alle exits styres av AI
- Hardcoded R-levels kun som fallback hvis AI down

---

## 5. IMPLEMENTERING - FILE CHANGES

### **Nye filer:**
1. `microservices/ai_engine/exit_evaluator.py` - exit logic
2. `microservices/ai_engine/routes/exit.py` - API endpoint

### **Modifiserte filer:**
1. `microservices/ai_engine/service.py` - add evaluate_exit()
2. `microservices/ai_engine/main.py` - register exit route
3. `microservices/harvest_brain/harvest_brain.py` - replace R-level logic
4. `frontend/quantumfond.com/` - add exit decision views (må finne frontend repo)

---

## 6. CONFIG VARIABLER

```bash
# .env additions:
AI_EXIT_ENABLED=true
AI_EXIT_MODE=shadow  # shadow|hybrid|full
AI_EXIT_HYBRID_PCT=0.5  # for hybrid mode
AI_EXIT_FALLBACK_PCT=0.25  # if AI unavailable
AI_EXIT_MIN_CONFIDENCE=0.5  # minimum confidence to trust AI decision
AI_EXIT_TIMEOUT_SEC=2  # timeout for AI Engine call
```

---

## 7. NESTE STEG

1. ✅ Review design med bruker
2. ⏸️ Implement evaluate_exit() i AI Engine
3. ⏸️ Modify Harvest Brain til call AI Engine
4. ⏸️ Add Redis event stream for decisions
5. ⏸️ Create Grafana dashboards
6. ⏸️ Integrate with quantumfond.com frontend
7. ⏸️ Test i shadow mode (1 uke)
8. ⏸️ Deploy hybrid mode (A/B test)
9. ⏸️ Full AI rollout

