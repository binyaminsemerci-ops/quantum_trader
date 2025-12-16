"""
RL v3 API Routes
================

REST API endpoints for RL v3 (PPO) system.

Endpoints:
- GET /api/v1/rl/v3/predict - Get PPO prediction
- POST /api/v1/rl/v3/train - Trigger training
- GET /api/v1/rl/v3/status - Get system status
- GET /api/v1/rl/v3/experiences - Get collected experiences
- POST /api/v1/rl/v3/shadow_mode - Toggle shadow mode

Author: Quantum Trader AI Team
Date: December 2, 2025
"""

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import structlog
import asyncio
from datetime import datetime

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/rl/v3", tags=["RL v3"])

# Global variable for tracking training progress
_training_progress: Dict[str, Any] = {
    "active": False,
    "progress": 0.0,
    "message": "No training in progress",
    "started_at": None,
    "metrics": {}
}


class PredictRequest(BaseModel):
    """Request model for RL v3 prediction."""
    
    price_change_1m: float = Field(default=0.0, description="1-minute price change")
    price_change_5m: float = Field(default=0.0, description="5-minute price change")
    price_change_15m: float = Field(default=0.0, description="15-minute price change")
    volatility: float = Field(default=0.02, description="Market volatility")
    rsi: float = Field(default=50.0, description="RSI indicator")
    macd: float = Field(default=0.0, description="MACD indicator")
    position_size: float = Field(default=0.0, description="Current position size")
    position_side: float = Field(default=0.0, description="Position side (1=long, -1=short, 0=none)")
    balance: float = Field(default=10000.0, description="Account balance")
    equity: float = Field(default=10000.0, description="Account equity")
    regime: str = Field(default="TREND", description="Market regime")
    trend_strength: float = Field(default=0.5, description="Trend strength (0-1)")
    volume_ratio: float = Field(default=1.0, description="Volume ratio")
    spread: float = Field(default=0.001, description="Bid-ask spread")
    time_of_day: float = Field(default=0.5, description="Time of day (0-1)")


class PredictResponse(BaseModel):
    """Response model for RL v3 prediction."""
    
    action: str = Field(..., description="Action name")
    action_code: int = Field(..., description="Action code (0-5)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    value: float = Field(..., description="State value estimate")


class StatusResponse(BaseModel):
    """Response model for RL v3 status."""
    
    model_config = {"protected_namespaces": ()}
    
    active: bool = Field(..., description="Whether RL v3 is active")
    shadow_mode: bool = Field(..., description="Whether in shadow mode")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    experiences_collected: int = Field(..., description="Number of experiences collected")
    model_path: str = Field(..., description="Path to model file")


class TrainResponse(BaseModel):
    """Response model for training."""
    
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, obs: PredictRequest):
    """
    Get RL v3 (PPO) prediction for given observation.
    
    Args:
        request: FastAPI request
        obs: Observation data
        
    Returns:
        PPO-based trading decision
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        # Build observation dict
        obs_dict = obs.dict()
        
        # Get prediction
        result = rl_subscriber_v3.manager.predict(obs_dict)
        
        # Map action code to name
        action_name = rl_subscriber_v3._map_action_to_name(result['action'])
        
        return PredictResponse(
            action=action_name,
            action_code=result['action'],
            confidence=result['confidence'],
            value=result['value']
        )
        
    except Exception as e:
        logger.error("[RL v3 API] Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/status", response_model=StatusResponse)
async def get_status(request: Request):
    """
    Get RL v3 system status.
    
    Args:
        request: FastAPI request
        
    Returns:
        System status information
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            return StatusResponse(
                active=False,
                shadow_mode=True,
                model_loaded=False,
                experiences_collected=0,
                model_path=""
            )
        
        from pathlib import Path
        model_path = Path(rl_subscriber_v3.rl_manager.config.model_path)
        
        # Get metrics from store
        metrics = rl_subscriber_v3.metrics
        
        return StatusResponse(
            active=True,
            shadow_mode=rl_subscriber_v3.shadow_mode,
            model_loaded=model_path.exists(),
            experiences_collected=metrics.total_predictions if hasattr(metrics, 'total_predictions') else 0,
            model_path=str(model_path)
        )
        
    except Exception as e:
        logger.error("[RL v3 API] Status check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.post("/train", response_model=TrainResponse)
async def train(request: Request, num_episodes: int = 10):
    """
    Trigger RL v3 training.
    
    Args:
        request: FastAPI request
        num_episodes: Number of training episodes
        
    Returns:
        Training status
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        # Train in background (this is synchronous for now)
        logger.info("[RL v3 API] Starting training", num_episodes=num_episodes)
        
        metrics = rl_subscriber_v3.manager.train(num_episodes=num_episodes)
        
        # Save trained model
        rl_subscriber_v3.save_model()
        
        logger.info(
            "[RL v3 API] Training complete",
            avg_reward=metrics['avg_reward'],
            final_reward=metrics['final_reward']
        )
        
        return TrainResponse(
            status="success",
            message=f"Training complete: {num_episodes} episodes, avg_reward={metrics['avg_reward']:.2f}"
        )
        
    except Exception as e:
        logger.error("[RL v3 API] Training failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.get("/experiences")
async def get_experiences(request: Request, limit: int = 100):
    """
    Get collected experiences.
    
    Args:
        request: FastAPI request
        limit: Maximum number of experiences to return
        
    Returns:
        List of collected experiences
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        experiences = rl_subscriber_v3.get_experiences()
        
        return {
            "total": len(experiences),
            "experiences": experiences[-limit:] if limit else experiences
        }
        
    except Exception as e:
        logger.error("[RL v3 API] Failed to get experiences", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get experiences: {str(e)}")


@router.post("/shadow_mode")
async def toggle_shadow_mode(request: Request, enabled: bool):
    """
    Toggle shadow mode.
    
    Args:
        request: FastAPI request
        enabled: Whether to enable shadow mode
        
    Returns:
        Updated status
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        rl_subscriber_v3.shadow_mode = enabled
        
        logger.info("[RL v3 API] Shadow mode toggled", enabled=enabled)
        
        return {
            "status": "success",
            "shadow_mode": enabled,
            "message": f"Shadow mode {'enabled' if enabled else 'disabled'}"
        }
        
    except Exception as e:
        logger.error("[RL v3 API] Failed to toggle shadow mode", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to toggle shadow mode: {str(e)}")


# ========================================
# PRODUCTION TRAINING WITH REAL DATA
# ========================================

class TrainProductionRequest(BaseModel):
    """Request model for production training."""
    
    symbol: str = Field(default="BTC/USDT", description="Trading pair")
    timeframe: str = Field(default="1h", description="Candle timeframe")
    lookback_hours: int = Field(default=720, description="Hours of historical data")
    num_episodes: int = Field(default=100, description="Number of training episodes")


class TrainProductionResponse(BaseModel):
    """Response model for production training."""
    
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Training metrics")


@router.post("/train_production", response_model=TrainProductionResponse)
async def train_production(
    request: Request,
    background_tasks: BackgroundTasks,
    train_req: TrainProductionRequest
):
    """
    Train RL v3 on real historical data from Binance.
    
    Args:
        request: FastAPI request
        background_tasks: FastAPI background tasks
        train_req: Training parameters
        
    Returns:
        Training status with metrics
    """
    global _training_progress
    
    try:
        # Check if training already in progress
        if _training_progress["active"]:
            return TrainProductionResponse(
                status="in_progress",
                message=f"Training already in progress: {_training_progress['message']}"
            )
        
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        # Start background training
        background_tasks.add_task(
            _run_production_training,
            rl_subscriber_v3,
            train_req.symbol,
            train_req.timeframe,
            train_req.lookback_hours,
            train_req.num_episodes
        )
        
        logger.info(
            "[RL v3 API] Production training started",
            symbol=train_req.symbol,
            timeframe=train_req.timeframe,
            episodes=train_req.num_episodes
        )
        
        return TrainProductionResponse(
            status="started",
            message=f"Training started on {train_req.symbol} with {train_req.num_episodes} episodes"
        )
        
    except Exception as e:
        logger.error("[RL v3 API] Failed to start production training", error=str(e))
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


async def _run_production_training(
    rl_subscriber_v3,
    symbol: str,
    timeframe: str,
    lookback_hours: int,
    num_episodes: int
):
    """
    Background task for production training with real data.
    """
    global _training_progress
    
    try:
        _training_progress = {
            "active": True,
            "progress": 0.0,
            "message": "Initializing...",
            "started_at": datetime.utcnow().isoformat(),
            "metrics": {}
        }
        
        # Import here to avoid circular imports
        from backend.domains.learning.rl_v3.market_data_provider import RealMarketDataProvider
        from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3
        from backend.domains.learning.rl_v3.ppo_trainer_v3 import PPOTrainer
        from backend.domains.learning.rl_v3.ppo_buffer_v3 import PPOBuffer
        import numpy as np
        
        _training_progress["message"] = "Fetching historical data..."
        _training_progress["progress"] = 5.0
        
        # Create market data provider
        provider = RealMarketDataProvider(
            symbol=symbol,
            timeframe=timeframe,
            lookback_hours=lookback_hours
        )
        
        # Create environment with real data
        env = TradingEnvV3(
            config=rl_subscriber_v3.manager.config,
            market_data_provider=provider
        )
        
        _training_progress["message"] = "Training in progress..."
        _training_progress["progress"] = 10.0
        
        # Training loop
        agent = rl_subscriber_v3.manager.agent
        trainer = PPOTrainer(agent, rl_subscriber_v3.manager.config)
        
        total_rewards = []
        policy_losses = []
        value_losses = []
        
        for episode in range(num_episodes):
            buffer = PPOBuffer(
                rl_subscriber_v3.manager.config.buffer_size,
                rl_subscriber_v3.manager.config.state_dim,
                rl_subscriber_v3.manager.config.gamma,
                rl_subscriber_v3.manager.config.lambda_gae
            )
            
            state = env.reset()
            episode_reward = 0.0
            
            # Collect trajectory
            for _ in range(rl_subscriber_v3.manager.config.buffer_size):
                action, log_prob, value = agent.act(state)
                next_state, reward, done, info = env.step(action)
                
                buffer.store(state, action, log_prob, reward, value, done)
                episode_reward += reward
                
                if done:
                    buffer.finish_path(last_value=0.0)
                    state = env.reset()
                else:
                    state = next_state
            
            # Finish any incomplete trajectory
            if buffer.ptr > buffer.path_start:
                _, _, last_value = agent.act(state)
                buffer.finish_path(last_value=last_value)
            
            # Update agent
            policy_loss, value_loss, entropy = trainer.update(buffer)
            
            # Track metrics
            total_rewards.append(episode_reward)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            
            # Update progress
            progress = 10.0 + (episode + 1) / num_episodes * 80.0
            _training_progress["progress"] = progress
            _training_progress["message"] = f"Episode {episode+1}/{num_episodes}"
            _training_progress["metrics"] = {
                "current_reward": float(episode_reward),
                "avg_reward": float(np.mean(total_rewards)),
                "policy_loss": float(policy_loss),
                "value_loss": float(value_loss)
            }
            
            logger.info(
                "[RL v3 Training] Progress",
                episode=episode+1,
                total=num_episodes,
                reward=episode_reward,
                avg_reward=np.mean(total_rewards)
            )
        
        # Save model
        _training_progress["message"] = "Saving model..."
        _training_progress["progress"] = 95.0
        
        rl_subscriber_v3.save_model()
        
        # Complete
        _training_progress["active"] = False
        _training_progress["progress"] = 100.0
        _training_progress["message"] = "Training complete"
        _training_progress["metrics"] = {
            "avg_reward": float(np.mean(total_rewards)),
            "final_reward": float(total_rewards[-1]),
            "best_reward": float(np.max(total_rewards)),
            "avg_policy_loss": float(np.mean(policy_losses)),
            "avg_value_loss": float(np.mean(value_losses))
        }
        
        logger.info(
            "[RL v3 Training] Complete",
            avg_reward=np.mean(total_rewards),
            final_reward=total_rewards[-1]
        )
        
    except Exception as e:
        logger.error("[RL v3 Training] Failed", error=str(e))
        _training_progress["active"] = False
        _training_progress["message"] = f"Training failed: {str(e)}"


@router.get("/training_progress")
async def get_training_progress():
    """
    Get current training progress.
    
    Returns:
        Training progress information
    """
    return _training_progress


# ========================================
# BENCHMARK VALIDATION
# ========================================

class BenchmarkResponse(BaseModel):
    """Response model for benchmark."""
    
    rl_v3_metrics: Dict[str, float] = Field(..., description="RL v3 metrics")
    buy_hold_metrics: Dict[str, float] = Field(..., description="Buy & Hold metrics")
    moving_avg_metrics: Dict[str, float] = Field(..., description="Moving Average metrics")
    random_metrics: Dict[str, float] = Field(..., description="Random agent metrics")
    winner: str = Field(..., description="Best performing strategy")


@router.post("/benchmark", response_model=BenchmarkResponse)
async def benchmark(
    request: Request,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    lookback_hours: int = 720,
    num_episodes: int = 10
):
    """
    Benchmark RL v3 against baseline strategies.
    
    Compares:
    - RL v3 (PPO agent)
    - Buy & Hold
    - Moving Average crossover
    - Random actions
    
    Metrics:
    - Total return
    - Sharpe ratio
    - Max drawdown
    - Win rate
    
    Args:
        request: FastAPI request
        symbol: Trading pair
        timeframe: Candle timeframe
        lookback_hours: Hours of historical data
        num_episodes: Number of test episodes
        
    Returns:
        Benchmark results for all strategies
    """
    try:
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        # Import dependencies
        from backend.domains.learning.rl_v3.market_data_provider import RealMarketDataProvider
        from backend.domains.learning.rl_v3.env_v3 import TradingEnvV3
        import numpy as np
        
        logger.info("[RL v3 Benchmark] Starting", symbol=symbol, episodes=num_episodes)
        
        # Create market data provider
        provider = RealMarketDataProvider(
            symbol=symbol,
            timeframe=timeframe,
            lookback_hours=lookback_hours
        )
        
        # Create environment
        env = TradingEnvV3(
            config=rl_subscriber_v3.manager.config,
            market_data_provider=provider
        )
        
        # Run benchmarks
        rl_v3_results = _run_strategy(env, rl_subscriber_v3.manager.agent, num_episodes, strategy="rl_v3")
        buy_hold_results = _run_strategy(env, None, num_episodes, strategy="buy_hold")
        moving_avg_results = _run_strategy(env, None, num_episodes, strategy="moving_avg")
        random_results = _run_strategy(env, None, num_episodes, strategy="random")
        
        # Calculate metrics
        rl_v3_metrics = _calculate_metrics(rl_v3_results)
        buy_hold_metrics = _calculate_metrics(buy_hold_results)
        moving_avg_metrics = _calculate_metrics(moving_avg_results)
        random_metrics = _calculate_metrics(random_results)
        
        # Determine winner (by Sharpe ratio)
        strategies = {
            "RL v3": rl_v3_metrics["sharpe_ratio"],
            "Buy & Hold": buy_hold_metrics["sharpe_ratio"],
            "Moving Average": moving_avg_metrics["sharpe_ratio"],
            "Random": random_metrics["sharpe_ratio"]
        }
        winner = max(strategies, key=strategies.get)
        
        logger.info(
            "[RL v3 Benchmark] Complete",
            winner=winner,
            rl_v3_sharpe=rl_v3_metrics["sharpe_ratio"],
            buy_hold_sharpe=buy_hold_metrics["sharpe_ratio"]
        )
        
        return BenchmarkResponse(
            rl_v3_metrics=rl_v3_metrics,
            buy_hold_metrics=buy_hold_metrics,
            moving_avg_metrics=moving_avg_metrics,
            random_metrics=random_metrics,
            winner=winner
        )
        
    except Exception as e:
        logger.error("[RL v3 Benchmark] Failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


def _run_strategy(env, agent, num_episodes: int, strategy: str) -> List[Dict[str, float]]:
    """Run a strategy for multiple episodes."""
    import numpy as np
    
    results = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_data = {
            "returns": [],
            "equity": [env.equity],
            "actions": []
        }
        
        while not done:
            # Get action based on strategy
            if strategy == "rl_v3" and agent:
                action, _, _ = agent.act(state, deterministic=True)
            elif strategy == "buy_hold":
                action = 1  # Always LONG
            elif strategy == "moving_avg":
                # Simple MA crossover
                if len(episode_data["equity"]) < 2:
                    action = 0  # HOLD
                else:
                    recent_avg = np.mean(episode_data["equity"][-10:]) if len(episode_data["equity"]) >= 10 else episode_data["equity"][-1]
                    action = 1 if env.current_price > recent_avg else 4  # LONG if above MA, else CLOSE
            else:  # random
                action = np.random.randint(0, 6)
            
            state, reward, done, info = env.step(action)
            
            episode_data["returns"].append(reward)
            episode_data["equity"].append(info["equity"])
            episode_data["actions"].append(action)
        
        results.append(episode_data)
    
    return results


def _calculate_metrics(results: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate performance metrics from results."""
    import numpy as np
    
    all_returns = []
    all_equity = []
    
    for episode in results:
        all_returns.extend(episode["returns"])
        all_equity.extend(episode["equity"])
    
    # Total return
    initial_equity = all_equity[0] if len(all_equity) > 0 else 10000.0
    final_equity = all_equity[-1] if len(all_equity) > 0 else 10000.0
    total_return = (final_equity - initial_equity) / initial_equity
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    returns_array = np.array(all_returns)
    sharpe_ratio = (np.mean(returns_array) / (np.std(returns_array) + 1e-8)) * np.sqrt(252) if len(returns_array) > 0 else 0.0
    
    # Max drawdown
    equity_array = np.array(all_equity)
    running_max = np.maximum.accumulate(equity_array)
    drawdown = (equity_array - running_max) / running_max
    max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0.0
    
    # Win rate
    wins = np.sum(returns_array > 0) if len(returns_array) > 0 else 0
    win_rate = wins / len(returns_array) if len(returns_array) > 0 else 0.0
    
    return {
        "total_return": float(total_return),
        "sharpe_ratio": float(sharpe_ratio),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "avg_return": float(np.mean(returns_array)) if len(returns_array) > 0 else 0.0
    }


# ========================================
# GRADUAL ROLLOUT SYSTEM
# ========================================

class RolloutRequest(BaseModel):
    """Request model for gradual rollout."""
    
    capital_percentage: float = Field(..., description="Percentage of capital to allocate (1-100)")
    max_position_size: Optional[float] = Field(default=None, description="Max position size in USD")
    enable_risk_guard: bool = Field(default=True, description="Enable RiskGuard protection")


class RolloutResponse(BaseModel):
    """Response model for gradual rollout."""
    
    status: str = Field(..., description="Rollout status")
    message: str = Field(..., description="Status message")
    config: Dict[str, Any] = Field(..., description="Rollout configuration")


@router.post("/rollout", response_model=RolloutResponse)
async def configure_rollout(
    request: Request,
    rollout_req: RolloutRequest
):
    """
    Configure gradual rollout for RL v3.
    
    Stages:
    - 1%: Initial testing with minimal capital
    - 5%: Confidence building phase
    - 10%: Standard operating allocation
    
    Args:
        request: FastAPI request
        rollout_req: Rollout configuration
        
    Returns:
        Rollout status and configuration
    """
    try:
        # Validate percentage
        if rollout_req.capital_percentage < 0 or rollout_req.capital_percentage > 100:
            raise HTTPException(
                status_code=400,
                detail="Capital percentage must be between 0 and 100"
            )
        
        # Get RL v3 subscriber from app state
        rl_subscriber_v3 = getattr(request.app.state, 'rl_subscriber_v3', None)
        
        if rl_subscriber_v3 is None:
            raise HTTPException(status_code=503, detail="RL v3 not initialized")
        
        # Configure rollout
        rollout_config = {
            "capital_percentage": rollout_req.capital_percentage,
            "max_position_size": rollout_req.max_position_size,
            "enable_risk_guard": rollout_req.enable_risk_guard,
            "configured_at": datetime.utcnow().isoformat(),
            "stage": _get_rollout_stage(rollout_req.capital_percentage)
        }
        
        # Store in app state (would typically save to database)
        if not hasattr(request.app.state, 'rl_v3_rollout'):
            request.app.state.rl_v3_rollout = {}
        
        request.app.state.rl_v3_rollout = rollout_config
        
        # Disable shadow mode if percentage > 0
        if rollout_req.capital_percentage > 0:
            rl_subscriber_v3.shadow_mode = False
        else:
            rl_subscriber_v3.shadow_mode = True
        
        logger.info(
            "[RL v3 Rollout] Configured",
            capital_percentage=rollout_req.capital_percentage,
            stage=rollout_config["stage"],
            shadow_mode=rl_subscriber_v3.shadow_mode
        )
        
        return RolloutResponse(
            status="success",
            message=f"Rollout configured: {rollout_config['stage']} ({rollout_req.capital_percentage}%)",
            config=rollout_config
        )
        
    except Exception as e:
        logger.error("[RL v3 Rollout] Failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Rollout configuration failed: {str(e)}")


@router.get("/rollout/status")
async def get_rollout_status(request: Request):
    """
    Get current rollout configuration.
    
    Args:
        request: FastAPI request
        
    Returns:
        Current rollout status
    """
    rollout_config = getattr(request.app.state, 'rl_v3_rollout', None)
    
    if rollout_config is None:
        return {
            "configured": False,
            "message": "No rollout configured (shadow mode)"
        }
    
    return {
        "configured": True,
        "config": rollout_config
    }


def _get_rollout_stage(capital_percentage: float) -> str:
    """Determine rollout stage based on capital percentage."""
    if capital_percentage == 0:
        return "Shadow Mode (0%)"
    elif capital_percentage <= 1:
        return "Stage 1: Initial Testing (1%)"
    elif capital_percentage <= 5:
        return "Stage 2: Confidence Building (5%)"
    elif capital_percentage <= 10:
        return "Stage 3: Standard Operation (10%)"
    else:
        return f"Stage 4: Expanded Operation ({capital_percentage}%)"

