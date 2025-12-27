# Training Worker Service

## Overview
Retraining Worker Service listens for model retraining jobs from Strategic Evolution (Phase 4T) and executes actual model training.

## Architecture

```
Strategic Evolution (Phase 4T)
    ↓ (schedules retrain jobs)
Redis Stream: quantum:stream:model.retrain
    ↓ (worker reads)
Retraining Worker
    ↓ (executes training)
Model Trainer
    ↓ (saves models)
/app/models/*.pkl
```

## Streams

### Input (Read):
- `quantum:stream:model.retrain` - Incoming retrain jobs

### Output (Write):
- `quantum:stream:learning.retraining.started` - Job started events
- `quantum:stream:learning.retraining.completed` - Job completion events
- `quantum:stream:learning.retraining.failed` - Job failure events

## Job Format

```json
{
  "model": "xgboost",
  "learning_rate": "0.002",
  "optimizer": "adam"
}
```

## Features

- **Consumer Group**: Uses Redis consumer groups for reliable job processing
- **Acknowledgement**: ACKs messages only after successful processing
- **Error Handling**: Publishes failures to separate stream
- **Model Persistence**: Saves trained models to `/app/models`
- **Monitoring**: Updates Redis counters for tracking

## Usage

### Local Development
```bash
cd microservices/training_worker
python retrain_worker.py
```

### Docker
```bash
docker compose -f docker-compose.vps.yml build retraining-worker
docker compose -f docker-compose.vps.yml up -d retraining-worker
```

### Verify
```bash
# Check logs
docker logs --tail 50 quantum_retraining_worker

# Check completed jobs
docker exec quantum_redis redis-cli XLEN quantum:stream:learning.retraining.completed

# Check latest completion
docker exec quantum_redis redis-cli XREVRANGE quantum:stream:learning.retraining.completed + - COUNT 1
```

## Integration

### With Strategic Evolution (Phase 4T):
- Evolution engine schedules retrain jobs → Worker executes them

### With Model Federation (Phase 4U):
- Worker saves new models → Federation loads and evaluates them

### With CLM v3:
- Shares `/app/models` volume for model file access

## Configuration

Environment variables:
- `REDIS_URL` - Redis connection URL (default: `redis://redis:6379/0`)
- `LOG_LEVEL` - Logging level (default: `INFO`)

## TODO: Production Training

Current implementation uses **mock training** (simulated).

For production, implement actual training in `model_trainer.py`:

```python
def _train_gradient_boosting(self, model, learning_rate, optimizer):
    # 1. Load historical data from database/Redis
    # 2. Prepare features and labels
    # 3. Initialize XGBoost/LightGBM with hyperparameters
    # 4. Train model
    # 5. Validate on holdout set
    # 6. Save trained model
    pass
```

## Metrics

Redis keys:
- `quantum:evolution:retrain_count` - Total jobs scheduled
- `quantum:evolution:retrain_completed_count` - Successfully completed jobs

## Status

✅ Implemented: Job listener, stream processing, error handling  
⚠️ TODO: Replace mock training with actual model training logic  
✅ Integrated: Docker compose, volume mounts, health checks
