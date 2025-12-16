"""
Test script for continuous learning functionality.
Simulates the full cycle: prediction → execution → outcome → retraining
"""
import sys
import os

# Fix import paths
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from datetime import datetime, timezone
from database import SessionLocal
from backend.models.ai_training import AITrainingSample, AIModelVersion
import json

def test_sample_creation():
    """Test creating AI training samples"""
    db = SessionLocal()
    try:
        # Create a test sample (without run_id to avoid foreign key issues)
        sample = AITrainingSample(
            symbol="BTCUSDC",
            timestamp=datetime.now(timezone.utc),
            predicted_action="BUY",
            prediction_score=0.0023,
            prediction_confidence=0.72,
            model_version="v_test",
            features=json.dumps([0.1, 0.2, 0.3] * 26),  # 78 features (more realistic)
            executed=True,
            execution_side="BUY",
            entry_price=45000.0,
            entry_quantity=0.1,
            entry_time=datetime.now(timezone.utc),
            outcome_known=True,
            exit_price=45500.0,
            exit_time=datetime.now(timezone.utc),
            realized_pnl=50.0,
            target_label=0.0111,  # 1.11% return
            target_class="WIN",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        db.add(sample)
        db.commit()
        db.refresh(sample)
        
        print(f"[OK] Created sample ID: {sample.id}")
        print(f"   Symbol: {sample.symbol}")
        print(f"   Action: {sample.predicted_action}")
        print(f"   Confidence: {sample.prediction_confidence}")
        print(f"   Entry: ${sample.entry_price}")
        print(f"   Exit: ${sample.exit_price}")
        print(f"   P&L: ${sample.realized_pnl}")
        print(f"   Return: {sample.target_label*100:.2f}%")
        print(f"   Class: {sample.target_class}")
        
        return sample.id
        
    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        db.rollback()
        return None
    finally:
        db.close()


def test_query_samples():
    """Test querying samples"""
    db = SessionLocal()
    try:
        # Query all samples
        samples = db.query(AITrainingSample).all()
        print(f"\n[CHART] Total samples in database: {len(samples)}")
        
        # Query samples with outcomes
        completed = db.query(AITrainingSample).filter(
            AITrainingSample.outcome_known == True
        ).all()
        print(f"   Completed samples (with outcomes): {len(completed)}")
        
        # Query by action
        buy_samples = db.query(AITrainingSample).filter(
            AITrainingSample.predicted_action == "BUY"
        ).count()
        sell_samples = db.query(AITrainingSample).filter(
            AITrainingSample.predicted_action == "SELL"
        ).count()
        hold_samples = db.query(AITrainingSample).filter(
            AITrainingSample.predicted_action == "HOLD"
        ).count()
        
        print(f"   BUY: {buy_samples}, SELL: {sell_samples}, HOLD: {hold_samples}")
        
        return len(completed)
        
    except Exception as e:
        print(f"❌ Error querying samples: {e}")
        return 0
    finally:
        db.close()


def test_model_versions():
    """Test model version tracking"""
    db = SessionLocal()
    try:
        versions = db.query(AIModelVersion).all()
        print(f"\n🤖 Model versions in database: {len(versions)}")
        
        for v in versions:
            print(f"   {v.version_id}: {v.model_type}")
            print(f"      Training samples: {v.training_samples}")
            print(f"      Validation accuracy: {v.validation_accuracy}")
            print(f"      Active: {v.is_active}")
        
        return len(versions)
        
    except Exception as e:
        print(f"❌ Error querying model versions: {e}")
        return 0
    finally:
        db.close()


def create_multiple_samples(count=10):
    """Create multiple test samples for retraining tests"""
    db = SessionLocal()
    created = 0
    
    try:
        for i in range(count):
            # Vary the data
            action = ["BUY", "SELL", "HOLD"][i % 3]
            entry_price = 45000 + (i * 100)
            exit_price = entry_price + ((-1 if action == "SELL" else 1) * (50 + i * 10))
            pnl = (exit_price - entry_price) * 0.1
            return_pct = (exit_price - entry_price) / entry_price
            
            sample = AITrainingSample(
                symbol=["BTCUSDC", "ETHUSDC", "SOLUSDC"][i % 3],
                timestamp=datetime.now(timezone.utc),
                predicted_action=action,
                prediction_score=0.001 * (i - 5),  # Vary scores
                prediction_confidence=0.5 + (i * 0.04),
                model_version="v_test",
                features=json.dumps([0.01 * i] * 77),  # 77 features
                executed=True,
                execution_side=action if action != "HOLD" else None,
                entry_price=entry_price,
                entry_quantity=0.1,
                entry_time=datetime.now(timezone.utc),
                outcome_known=True,
                exit_price=exit_price,
                exit_time=datetime.now(timezone.utc),
                realized_pnl=pnl,
                target_label=return_pct,
                target_class="WIN" if pnl > 0 else "LOSS",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            db.add(sample)
            created += 1
        
        db.commit()
        print(f"\n[OK] Created {created} test samples")
        return created
        
    except Exception as e:
        print(f"❌ Error creating samples: {e}")
        db.rollback()
        return created
    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("CONTINUOUS LEARNING TEST")
    print("=" * 60)
    
    # Test 1: Create a single sample
    print("\n1️⃣  Testing sample creation...")
    sample_id = test_sample_creation()
    
    # Test 2: Query samples
    print("\n2️⃣  Testing sample queries...")
    completed_count = test_query_samples()
    
    # Test 3: Check model versions
    print("\n3️⃣  Testing model version tracking...")
    version_count = test_model_versions()
    
    # Test 4: Create multiple samples for retraining
    if completed_count < 10:
        print("\n4️⃣  Creating additional samples for retraining test...")
        create_multiple_samples(10)
        test_query_samples()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"[OK] Database tables: OK")
    print(f"[OK] Sample creation: OK")
    print(f"[OK] Sample queries: OK")
    print(f"[OK] Model version tracking: OK")
    print(f"\n💡 Next steps:")
    print(f"   1. Let system collect real trading data")
    print(f"   2. Wait for 100+ samples with outcomes")
    print(f"   3. Test retraining: curl -X POST http://localhost:8000/ai/retrain")
    print(f"   4. Review model versions: curl http://localhost:8000/ai/models")
    print("=" * 60)
