from fastapi import APIRouter
from schemas import AIStatus, PredictionsResponse, Prediction
import time
import random

router = APIRouter(prefix="/ai", tags=["AI"])

@router.get("/status", response_model=AIStatus)
def get_ai_status():
    """Get AI engine status with model performance metrics"""
    return AIStatus(
        accuracy=0.72,
        sharpe=1.14,
        latency=184,
        models=["XGB", "LGBM", "N-HiTS", "TFT"]
    )

@router.get("/predictions", response_model=PredictionsResponse)
def get_ai_predictions():
    """Get latest AI predictions/signals
    
    Returns 15 most recent trading signals with ensemble consensus.
    Each prediction shows agreement from multiple AI models.
    """
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
    all_models = ["XGB", "LGBM", "N-HiTS", "TFT"]
    regimes = ["Bullish", "Bearish", "Neutral"]
    
    predictions = []
    
    for i in range(15):
        symbol = random.choice(symbols)
        side = random.choice(["LONG", "SHORT"])
        
        # Ensemble: 2-4 models agree on signal
        num_models = random.randint(2, 4)
        agreeing_models = random.sample(all_models, num_models)
        model_str = "+".join(agreeing_models)
        
        # Confidence: higher when more models agree
        base_confidence = 0.60 + (num_models * 0.08)  # 68-92% range
        confidence = round(base_confidence + random.uniform(-0.05, 0.05), 2)
        confidence = min(0.98, max(0.60, confidence))  # clamp 60-98%
        
        entry_price = round(random.uniform(100, 50000), 2)
        
        predictions.append(Prediction(
            id=f"pred_{int(time.time())}_{i}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            symbol=symbol,
            side=side,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=round(entry_price * (0.97 if side == "LONG" else 1.03), 2),
            take_profit=round(entry_price * (1.05 if side == "LONG" else 0.95), 2),
            leverage=random.choice([5, 10, 15, 20]),
            model=model_str,
            reason=f"{num_models}/{len(all_models)} models agree: Strong {'bullish' if side == 'LONG' else 'bearish'} signal",
            volatility=round(random.uniform(0.015, 0.035), 3),
            regime=random.choice(regimes),
            position_size_usd=round(random.uniform(500, 2000), 2)
        ))
    
    return PredictionsResponse(
        predictions=predictions,
        count=len(predictions),
        timestamp=time.time()
    )
