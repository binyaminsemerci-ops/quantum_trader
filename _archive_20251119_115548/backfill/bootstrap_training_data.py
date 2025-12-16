"""
Bootstrap initial training data with varied samples
for the AI model to learn from.
"""
import asyncio
import sys
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.database import SessionLocal
from backend.models.ai_training import AITrainingSample


def create_bootstrap_samples():
    """Create realistic bootstrap samples with varied outcomes"""
    
    # Sample features (realistic technical indicators)
    base_features = {
        "Close": 3000.0,
        "Volume": 1000000.0,
        "EMA_10": 2990.0,
        "EMA_50": 2950.0,
        "RSI": 55.0,
        "MACD": 15.0,
        "MACD_signal": 12.0,
        "BB_upper": 3100.0,
        "BB_middle": 3000.0,
        "BB_lower": 2900.0,
        "ATR": 50.0,
        "volume_sma_20": 950000.0,
        "price_change_pct": 1.5,
        "high_low_range": 100.0
    }
    
    samples = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=48)
    
    # 1. Winning BUY trades (bullish scenarios)
    for i in range(10):
        features = base_features.copy()
        features["RSI"] = 40 + i * 2  # Oversold to neutral
        features["MACD"] = 5 + i * 3   # Positive momentum
        features["price_change_pct"] = 0.5 + i * 0.3
        
        sample = AITrainingSample(
            symbol="ETHUSDT",
            timestamp=base_time + timedelta(hours=i),
            predicted_action="BUY",
            prediction_score=0.65 + i * 0.02,
            prediction_confidence=0.7 + i * 0.015,
            features=json.dumps(features),
            executed=True,
            execution_side="BUY",
            entry_price=3000.0 + i * 10,
            entry_quantity=0.1,
            entry_time=base_time + timedelta(hours=i),
            outcome_known=True,
            exit_price=3000.0 + i * 10 + 30,  # +1% profit
            exit_time=base_time + timedelta(hours=i, minutes=30),
            realized_pnl=1.0,  # 1% profit
            hold_duration_seconds=1800,
            target_label=1.0,
            target_class="WIN"
        )
        samples.append(sample)
    
    # 2. Losing BUY trades (false signals)
    for i in range(5):
        features = base_features.copy()
        features["RSI"] = 70 + i  # Overbought
        features["MACD"] = -2 - i * 2  # Negative divergence
        features["price_change_pct"] = -0.5 - i * 0.2
        
        sample = AITrainingSample(
            symbol="BTCUSDT",
            timestamp=base_time + timedelta(hours=12 + i),
            predicted_action="BUY",
            prediction_score=0.55 + i * 0.01,
            prediction_confidence=0.6,
            features=json.dumps(features),
            executed=True,
            execution_side="BUY",
            entry_price=65000.0,
            entry_quantity=0.01,
            entry_time=base_time + timedelta(hours=12 + i),
            outcome_known=True,
            exit_price=64350.0,  # -1% loss
            exit_time=base_time + timedelta(hours=12 + i, minutes=45),
            realized_pnl=-1.0,
            hold_duration_seconds=2700,
            target_label=-1.0,
            target_class="LOSS"
        )
        samples.append(sample)
    
    # 3. Winning SELL trades (bearish scenarios)
    for i in range(8):
        features = base_features.copy()
        features["RSI"] = 65 + i * 2  # Overbought
        features["MACD"] = -5 - i * 2  # Negative momentum
        features["price_change_pct"] = -0.8 - i * 0.2
        
        sample = AITrainingSample(
            symbol="SOLUSDT",
            timestamp=base_time + timedelta(hours=20 + i),
            predicted_action="SELL",
            prediction_score=0.70,
            prediction_confidence=0.75,
            features=json.dumps(features),
            executed=True,
            execution_side="SELL",
            entry_price=150.0,
            entry_quantity=1.0,
            entry_time=base_time + timedelta(hours=20 + i),
            outcome_known=True,
            exit_price=148.5,  # +1% profit (short)
            exit_time=base_time + timedelta(hours=20 + i, minutes=40),
            realized_pnl=1.0,
            hold_duration_seconds=2400,
            target_label=1.0,
            target_class="WIN"
        )
        samples.append(sample)
    
    # 4. Correct HOLD signals (no trade = avoided loss)
    for i in range(7):
        features = base_features.copy()
        features["RSI"] = 48 + i  # Neutral
        features["MACD"] = 1 - i * 0.5  # Weak signal
        features["price_change_pct"] = 0.1
        
        sample = AITrainingSample(
            symbol="ADAUSDT",
            timestamp=base_time + timedelta(hours=30 + i),
            predicted_action="HOLD",
            prediction_score=0.50,
            prediction_confidence=0.6,
            features=json.dumps(features),
            executed=False,
            outcome_known=True,
            target_label=0.0,
            target_class="NEUTRAL"
        )
        samples.append(sample)
    
    return samples


async def main():
    print("[ROCKET] BOOTSTRAPPING TRAINING DATA\n")
    
    db = SessionLocal()
    try:
        # Check existing samples
        existing = db.query(AITrainingSample).count()
        print(f"[CHART] Existing samples: {existing}")
        
        # Create bootstrap samples
        samples = create_bootstrap_samples()
        print(f"📦 Creating {len(samples)} bootstrap samples...")
        
        # Add to database
        for sample in samples:
            db.add(sample)
        
        db.commit()
        
        # Verify
        total = db.query(AITrainingSample).count()
        print(f"\n[OK] SUCCESS!")
        print(f"   Total samples now: {total}")
        print(f"   New samples added: {total - existing}")
        
        # Show distribution
        buy_count = db.query(AITrainingSample).filter(AITrainingSample.predicted_action == "BUY").count()
        sell_count = db.query(AITrainingSample).filter(AITrainingSample.predicted_action == "SELL").count()
        hold_count = db.query(AITrainingSample).filter(AITrainingSample.predicted_action == "HOLD").count()
        
        print(f"\n[CHART_UP] Signal distribution:")
        print(f"   BUY:  {buy_count} samples")
        print(f"   SELL: {sell_count} samples")
        print(f"   HOLD: {hold_count} samples")
        
        # Show outcomes
        wins = db.query(AITrainingSample).filter(AITrainingSample.target_class == "WIN").count()
        losses = db.query(AITrainingSample).filter(AITrainingSample.target_class == "LOSS").count()
        neutral = db.query(AITrainingSample).filter(AITrainingSample.target_class == "NEUTRAL").count()
        
        print(f"\n[MONEY] Outcome distribution:")
        print(f"   WIN:     {wins} samples")
        print(f"   LOSS:    {losses} samples")
        print(f"   NEUTRAL: {neutral} samples")
        
        print(f"\n[TARGET] Model can now learn:")
        print(f"   [OK] When to BUY (bullish indicators)")
        print(f"   [OK] When to SELL (bearish indicators)")
        print(f"   [OK] When to HOLD (weak signals)")
        print(f"   [OK] Win vs Loss patterns")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
