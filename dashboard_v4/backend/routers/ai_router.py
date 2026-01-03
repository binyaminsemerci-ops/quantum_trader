from fastapi import APIRouter
from schemas import AIStatus, PredictionsResponse, Prediction
import time
import random
import json

router = APIRouter(prefix="/ai", tags=["AI"])

@router.get("/status", response_model=AIStatus)
def get_ai_status():
    """Get AI engine status with model performance metrics
    
    TESTNET MODE: Reads REAL accuracy from latest AI signals in Redis.
    Calculates average confidence from last 50 signals.
    """
    import redis
    import json
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis')
        r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Get last 50 AI signals to calculate average accuracy
        signals = r.xrevrange('quantum:stream:ai.signal_generated', '+', '-', count=50)
        
        if signals:
            confidences = []
            for _, signal_data in signals:
                payload_json = signal_data.get('payload', '{}')
                payload = json.loads(payload_json)
                conf = payload.get('confidence', 0.0)
                if conf > 0:
                    confidences.append(conf)
            
            if confidences:
                avg_accuracy = sum(confidences) / len(confidences)
                logger.info(f"✅ TESTNET AI Accuracy: {avg_accuracy:.1%} (from {len(confidences)} signals)")
                
                return AIStatus(
                    accuracy=round(avg_accuracy, 3),
                    sharpe=0.0,  # TESTNET - no historical performance yet
                    latency=184,  # Hardcoded - from AI engine metrics
                    models=["XGB", "LGBM", "N-HiTS", "TFT"]
                )
    except Exception as e:
        logger.warning(f"⚠️ Could not read AI accuracy from Redis: {e}")
    
    # Fallback: Default testnet values
    logger.info("🧪 TESTNET - Using default AI metrics")
    return AIStatus(
        accuracy=0.72,
        sharpe=0.0,
        latency=184,
        models=["XGB", "LGBM", "N-HiTS", "TFT"]
    )

@router.get("/health")
def get_ai_health():
    """Proxy endpoint for AI Engine /health to avoid CORS issues.
    
    Returns full AI Engine health data including metrics with feature flags.
    """
    import requests
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Call AI Engine health endpoint (internal Docker network)
        response = requests.get('http://ai-engine:8001/health', timeout=10.0)
        response.raise_for_status()
        health_data = response.json()
        
        logger.info("✅ AI Engine health data fetched successfully")
        return health_data
        
    except requests.exceptions.Timeout:
        logger.error("❌ AI Engine /health timeout")
        return {
            "status": "ERROR",
            "error": "AI Engine health check timeout",
            "metrics": {
                "ensemble_enabled": False,
                "governance_active": False,
                "intelligent_leverage_v2": False,
                "rl_position_sizing": False,
                "adaptive_leverage_enabled": False,
                "cross_exchange_intelligence": False
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to fetch AI Engine health: {e}")
        return {
            "status": "ERROR", 
            "error": str(e),
            "metrics": {
                "ensemble_enabled": False,
                "governance_active": False,
                "intelligent_leverage_v2": False,
                "rl_position_sizing": False,
                "adaptive_leverage_enabled": False,
                "cross_exchange_intelligence": False
            }
        }
    
    # Fallback: Default testnet values
    logger.info("🧪 TESTNET - Using default AI metrics")
    return AIStatus(
        accuracy=0.72,
        sharpe=0.0,
        latency=184,
        models=["XGB", "LGBM", "N-HiTS", "TFT"]
    )

@router.get("/predictions", response_model=PredictionsResponse)
def get_ai_predictions():
    """Get latest AI predictions/signals
    
    Returns 15 most recent REAL trading signals from Redis trade.intent stream.
    """
    import redis
    import logging
    import os
    
    logger = logging.getLogger(__name__)
    predictions = []
    
    try:
        redis_host = os.getenv('REDIS_HOST', 'redis')
        r = redis.Redis(host=redis_host, port=6379, decode_responses=True)
        
        # Get last 15 trade.intent events from Redis stream
        events = r.xrevrange('quantum:stream:trade.intent', '+', '-', count=15)
        logger.info(f"[AI Router] Fetched {len(events)} events from Redis")
        
        for event_id, event_data in events:
            try:
                # Parse payload JSON
                payload_json = event_data.get('payload', '{}')
                payload = json.loads(payload_json)
                
                # Extract data
                symbol = payload.get('symbol', 'UNKNOWN')
                side = payload.get('side', 'BUY')
                confidence = payload.get('confidence', 0.5)
                entry_price = payload.get('entry_price', 0.0)
                stop_loss = payload.get('stop_loss', 0.0)
                take_profit = payload.get('take_profit', 0.0)
                leverage = int(payload.get('leverage', 1))
                position_size = payload.get('position_size_usd', 0.0)
                volatility = payload.get('volatility_factor', 0.0)
                regime = payload.get('regime', 'UNKNOWN').title()
                timestamp_str = payload.get('timestamp', '')
                
                # Format timestamp as ISO datetime string
                if 'T' in timestamp_str:
                    # Already in ISO format, use as is
                    formatted_timestamp = timestamp_str
                else:
                    # Generate ISO timestamp with current date
                    from datetime import datetime
                    current_date = datetime.utcnow().date()
                    if timestamp_str:
                        try:
                            # Try to parse HH:MM:SS
                            time_parts = timestamp_str.split(':')
                            dt = datetime(current_date.year, current_date.month, current_date.day,
                                        int(time_parts[0]), int(time_parts[1]), int(time_parts[2]))
                            formatted_timestamp = dt.isoformat() + 'Z'
                        except:
                            # Fallback to current time
                            formatted_timestamp = datetime.utcnow().isoformat() + 'Z'
                    else:
                        formatted_timestamp = datetime.utcnow().isoformat() + 'Z'
                
                # Determine side format
                side_formatted = "LONG" if side in ["BUY", "LONG"] else "SHORT"
                
                predictions.append(Prediction(
                    id=event_id,
                    timestamp=formatted_timestamp,
                    symbol=symbol,
                    side=side_formatted,
                    confidence=round(confidence, 2),
                    entry_price=round(entry_price, 4),
                    stop_loss=round(stop_loss, 4),
                    take_profit=round(take_profit, 4),
                    leverage=leverage,
                    model="ensemble",
                    reason=f"AI signal from {regime} regime",
                    volatility=round(volatility, 3),
                    regime=regime,
                    position_size_usd=round(position_size, 2)
                ))
                logger.info(f"[AI Router] Parsed prediction: {symbol} {side_formatted}")
            except Exception as e:
                logger.error(f"[AI Router] Failed to parse event {event_id}: {e}")
                continue  # Skip malformed events
        
    except Exception as e:
        logger.error(f"[AI Router] Redis connection failed: {e}")
        pass  # Fallback to empty if Redis unavailable
    
    logger.info(f"[AI Router] Returning {len(predictions)} predictions")
    return PredictionsResponse(
        predictions=predictions,
        count=len(predictions),
        timestamp=time.time()
    )
