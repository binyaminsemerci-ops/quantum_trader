# Phase 4T - Strategic Evolution Engine

## Overview
Meta-layer for automatic strategy evolution and optimization. Continuously evaluates, selects, mutates, and retrains trading strategies based on performance metrics.

## Architecture

```
Strategic Evolution Engine
├── Performance Evaluator  → Ranks strategies by Sharpe, Win Rate, Drawdown
├── Model Selector        → Selects top 3 performers
├── Mutation Engine       → Explores hyperparameter space
└── Retrain Manager       → Schedules retraining jobs
```

## Key Features

- **Automatic Performance Evaluation**: Composite scoring (Sharpe 40%, Win Rate 30%, Drawdown -20%, Consistency 10%)
- **Intelligent Selection**: Top 3 strategies selected for evolution
- **Hyperparameter Mutation**: Explores learning rates, batch sizes, optimizers, dropout rates
- **Seamless Retraining**: Pushes jobs to Redis stream for CLM pickup
- **Continuous Learning**: 10-minute evolution cycles

## Redis Keys

### Input
- `quantum:strategy:performance` (list) - Strategy performance metrics

### Output
- `quantum:evolution:rankings` (string) - Ranked strategy list
- `quantum:evolution:selected` (string) - Selected top models
- `quantum:evolution:mutated` (string) - Mutated configurations
- `quantum:stream:model.retrain` (stream) - Retrain job queue
- `quantum:evolution:retrain_count` (counter) - Total retrains scheduled

## Integration

### With CLM (Continuous Learning Module)
CLM consumes `quantum:stream:model.retrain` and executes retraining jobs.

### With AI Engine
Exposes evolution metrics via `/health` endpoint.

### With Strategic Memory
Uses historical performance data for evaluation.

## Deployment

```bash
docker compose -f docker-compose.vps.yml build strategic-evolution
docker compose -f docker-compose.vps.yml up -d strategic-evolution
```

## Monitoring

```bash
# Check logs
docker logs -f quantum_strategic_evolution

# Check rankings
docker exec redis redis-cli GET quantum:evolution:rankings | jq .

# Check selected models
docker exec redis redis-cli GET quantum:evolution:selected | jq .

# Check mutations
docker exec redis redis-cli GET quantum:evolution:mutated | jq .

# Check retrain stream
docker exec redis redis-cli XREVRANGE quantum:stream:model.retrain + - COUNT 5
```

## Health Check

```bash
curl -s http://localhost:8001/health | jq '.metrics.strategic_evolution'
```

Expected response:
```json
{
  "status": "active",
  "selected_models": ["nhits", "patchtst", "xgboost"],
  "mutation_count": 3,
  "retrain_count": 42
}
```

## Configuration

Evolution cycle runs every **10 minutes** (configurable in `evolution_engine.py`).

Top **3 models** are selected for evolution (configurable in `model_selector.py`).

## Performance

- Minimal CPU usage (evaluation logic)
- Low memory footprint (~50MB)
- Redis-native streaming
- Async-ready architecture
