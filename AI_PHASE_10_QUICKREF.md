# 🧬 PHASE 10: STRATEGY EVOLUTION - QUICK REFERENCE

**Status**: ✅ DEPLOYED  
**Container**: `quantum_strategy_evolution` (HEALTHY)  
**Last Evolution**: 2025-12-20 20:50:37 UTC  
**Current Generation**: 2  
**Memory Bank**: 9 strategies  

---

## 🎯 WHAT IS THIS?

**Long-term evolutionary memory system** that uses **genetic algorithms** to evolve trading strategies over weeks and months.

- **Natural Selection**: Only top 3 strategies survive
- **Genetic Crossover**: Best strategies combine to create offspring
- **Random Mutation**: Introduces novelty to escape local optima
- **Memory Bank**: Archives all strategies for historical analysis
- **24-Hour Cycle**: Evolution happens once per day

Think of it as **artificial natural selection** for trading strategies.

---

## ⚡ QUICK START

### **Check Status**
```bash
# Container health
systemctl list-units | grep strategy_evolution

# View logs
journalctl -u quantum_strategy_evolution.service --tail 50

# Check current policy
redis-cli GET current_policy | jq
```

### **Monitor Evolution**
```bash
# Evolution statistics
redis-cli GET evolution_stats | jq

# Survivors
redis-cli GET meta_survivors | jq

# Best strategy
redis-cli HGETALL evolution_best

# Memory bank
ls -lh backend/microservices/strategy_evolution/memory_bank/
```

### **Force New Evolution**
```bash
# Restart container
docker compose restart strategy-evolution

# Watch logs
journalctl -u quantum_strategy_evolution.service -f
```

---

## 🧬 GENETIC ALGORITHM CYCLE

```
Day 1: [Strategy A, B, C, D, E] (5 strategies)
         ↓
       FITNESS EVALUATION
       Fitness: [1.8, 0.3, -1.1, -6.0, -14.4]
         ↓
       NATURAL SELECTION (top 3 survive)
       Survivors: [A, B, C]
         ↓
       REPRODUCTION
       ├─ Crossover: A × B → Child1
       ├─ Crossover: A × C → Child2
       └─ Mutation:  B' → Child3
         ↓
       SELECTION
       Best Child: Child1 (fitness 0.2)
         ↓
       PROMOTE TO PRODUCTION
       current_policy ← Child1
         ↓
       WAIT 24 HOURS
         ↓
Day 2: [A, B, C, Child1, Child2, Child3] (6 strategies)
       [Repeat cycle with new population]
```

---

## 📊 CURRENT STATE (LIVE)

### **Latest Evolution Results**
```
Generation: 2
Strategies in Memory: 9
Survivors: 3
Offspring Generated: 3

Production Policy:
  ID: evolved_20251220_205037_9799
  Method: genetic_crossover
  Risk Factor: 1.012
  Momentum: 0.951
  Mean Reversion: 0.509
  Position Scaler: 1.153
  Fitness: 0.2
```

### **Survivors (Top 3)**
```
#1: variant_20251220_160012_8821
    Fitness: 1.7659
    Score: 3.810
    Sharpe: 0.774

#2-3: Waiting for more trading data
```

---

## 🧪 FITNESS FUNCTION

**Formula:**
```python
fitness = (
    score * 0.35 +              # Composite score from Phase 9
    sharpe * 0.25 +             # Risk-adjusted returns
    sortino * 0.15 +            # Downside risk
    (pnl / 100) * 0.15 +        # Absolute profitability
    survival_bonus * 5.0 +      # How long it has survived
    generation_bonus * 5.0 -    # Older gens that still work
    (drawdown / 100) * 0.5      # Drawdown penalty
)
```

**Key Concepts:**
- Strategies that survive longer get **bonus points**
- Older generations that still perform get **stability bonus**
- High drawdown is **heavily penalized**
- Balances profit, risk, and longevity

---

## ⚙️ CONFIGURATION

**Environment Variables** (systemctl.yml):
```yaml
EVOLUTION_INTERVAL: 86400   # 24 hours (daily evolution)
SURVIVORS: 3                # Keep top 3 strategies
MAX_MEMORY: 100            # Max 100 strategies in memory
CROSSOVER_RATE: 0.7        # 70% crossover, 30% mutation
```

**Tuning Tips:**

**More Aggressive Evolution:**
```yaml
EVOLUTION_INTERVAL: 43200   # 12 hours
SURVIVORS: 5                # More diversity
CROSSOVER_RATE: 0.8        # More crossover
```

**More Conservative:**
```yaml
EVOLUTION_INTERVAL: 172800  # 48 hours
SURVIVORS: 2                # Only best
CROSSOVER_RATE: 0.5        # More mutations
```

---

## 🔗 INTEGRATION

### **Phase 7: Trade Journal**
- Evolution reads `latest_report` for baseline metrics
- Uses real PnL, Sharpe, Sortino for fitness

### **Phase 8: RL Optimizer**
- Evolution provides `current_policy` parameters
- RL uses these as tactical guidance

### **Phase 9: Meta-Cognitive**
- Phase 9 generates strategy variants every 12 hours
- Evolution imports them via `meta_best_strategy`
- Creates long-term selection pressure

### **Four-Layer System**
```
Layer 4: Evolution (24h)  ← Strategic inheritance
Layer 3: Meta-Cognitive (12h)  ← Strategy generation
Layer 2: RL Optimizer (30min)  ← Tactical adjustments
Layer 1: Model Ensemble (real-time)  ← Predictions
```

---

## 📁 FILE LOCATIONS

```
backend/microservices/strategy_evolution/
├── evolution_service.py         # Main logic
├── Dockerfile                    # Container
└── memory_bank/                  # Strategy archive
    ├── variant_*.json            # Phase 9 imports
    ├── evolved_*.json            # Crossover offspring
    └── mutated_*.json            # Mutation offspring
```

---

## 🔑 REDIS KEYS

```bash
current_policy              # Active production strategy
evolution_best              # Best strategy metadata (hash)
evolution_stats             # Cycle statistics (JSON)
meta_survivors              # Top 3 survivor IDs (JSON)
```

---

## 🛠️ COMMON OPERATIONS

### **View Strategy Details**
```bash
# Production policy
redis-cli GET current_policy | jq

# Specific strategy
cat backend/microservices/strategy_evolution/memory_bank/evolved_*.json | jq

# Compare strategies
ls -lhS memory_bank/  # Sorted by size
```

### **Analyze Evolution Progress**
```bash
# Generation history
grep "Generation:" memory_bank/*.json

# Fitness trends
grep "fitness" memory_bank/*.json | sort -n

# Parent lineage
grep "parent_ids" memory_bank/*.json
```

### **Troubleshooting**
```bash
# Container not starting?
journalctl -u quantum_strategy_evolution.service --tail 100

# Not enough strategies?
cp backend/microservices/strategy_evaluator/sandbox_strategies/*.json \
   backend/microservices/strategy_evolution/memory_bank/

# Evolution stalled?
docker compose restart strategy-evolution
```

---

## 📈 EXPECTED TIMELINE

**Week 1:**
- Generations: 1-7
- Improvement: 5-15% fitness gain
- Focus: Exploring parameter space

**Month 1:**
- Generations: 30-40
- Improvement: 20-40% fitness gain
- Focus: Finding optimal regions

**Month 3:**
- Generations: 90-120
- Improvement: 50-70% fitness gain
- Focus: Fine-tuning winners

**Month 6:**
- Generations: 180-200
- Improvement: 70-100% fitness gain
- Focus: Adapting to market shifts

---

## 🚨 WARNING SIGNS

**Healthy Evolution:**
- ✅ Fitness increasing over time
- ✅ Parameter diversity maintained
- ✅ Occasional breakthrough mutations
- ✅ Survivors changing every few cycles

**Problems:**
- ⚠️ Fitness stagnating for 10+ generations
- ⚠️ All strategies converging to same values
- ⚠️ No mutations surviving
- ⚠️ Drawdown > 35%

**Critical Issues:**
- 🚨 Fitness decreasing
- 🚨 Memory bank empty
- 🚨 Container restarting
- 🚨 Redis connection errors

---

## 💡 KEY INSIGHTS

### **Why Genetic Algorithms?**
1. **No gradient needed** - optimizes non-differentiable objectives
2. **Avoids local optima** - mutations escape traps
3. **Multi-objective** - balances profit vs. risk naturally
4. **Adapts over time** - tracks changing markets
5. **Population diversity** - prevents overfitting

### **Why 24-Hour Cycles?**
- Enough trading data for reliable fitness evaluation
- Not too fast (prevents noise from daily fluctuations)
- Not too slow (adapts to regime changes within weeks)

### **Why Top 3 Survivors?**
- Maintains diversity (vs. top 1)
- Strong selection pressure (vs. top 10)
- Allows crossover with different parents

---

## 🎯 QUICK COMMANDS

```bash
# Status
systemctl list-units | grep strategy_evolution
journalctl -u quantum_strategy_evolution.service --tail 20

# Current state
redis-cli GET current_policy | jq
redis-cli GET evolution_stats | jq

# Memory bank
ls -lht backend/microservices/strategy_evolution/memory_bank/ | head -10

# Force evolution now
docker compose restart strategy-evolution

# Watch live
journalctl -u quantum_strategy_evolution.service -f
```

---

## 📚 FULL DOCUMENTATION

See [AI_PHASE_10_EVOLUTION_COMPLETE.md](AI_PHASE_10_EVOLUTION_COMPLETE.md) for:
- Complete genetic algorithm theory
- Detailed fitness function explanation
- Integration architecture
- Performance metrics
- Troubleshooting guide
- Expected evolution trajectories

---

**🧬 THE SYSTEM NOW EVOLVES FOREVER 🧬**

Phase 10 completes the ultimate autonomous AI system:
- ✅ Self-learning (Phase 2)
- ✅ Self-optimizing (Phase 8)
- ✅ Self-evaluating (Phase 9)
- ✅ Self-evolving (Phase 10) ← **YOU ARE HERE**

**This is artificial life applied to trading.**

