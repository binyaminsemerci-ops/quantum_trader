# backend/seed_trades.py
import os
from datetime import datetime, timedelta
from database import SessionLocal, engine, Base
from models import Trade, TradeLog
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def drop_and_recreate_tables():
    """Drop existing tables and recreate with fresh schema"""
    logger.info("Dropping and recreating database tables...")
    try:
        # Drop all tables first
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Dropped existing tables")
        
        # Create fresh tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Created fresh tables with correct schema")
        return True
    except Exception as e:
        logger.error(f"❌ Error recreating tables: {e}")
        return False

def create_sample_data():
    """Create sample data with proper integer IDs"""
    
    # Create sample trades with integer IDs
    sample_trades = []
    for i in range(1, 21):
        trade = Trade(
            symbol=["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT"][i % 4],
            side="buy" if i % 2 == 0 else "sell",
            qty=round(0.1 + (i * 0.05), 3),
            price=50000 + (i * 1000),
            pnl=round((i - 10) * 23.45, 2),
            fee=round(0.001 * (50000 + i * 1000), 2),
            timestamp=datetime.now() - timedelta(minutes=i * 10),
            status="filled",
        )
        sample_trades.append(trade)

    # Create sample trade logs
    sample_logs = []
    for i in range(1, 21):
        log = TradeLog(
            symbol=["BTCUSDT", "ETHUSDT", "ADAUSDT"][i % 3],
            side="buy" if i % 2 == 0 else "sell",
            qty=round(0.1 + (i * 0.03), 3),
            price=48000 + (i * 800),
            status="executed" if i % 3 == 0 else "pending",
            reason="AI signal" if i % 2 == 0 else "Manual trade",
            timestamp=datetime.now() - timedelta(minutes=i * 5),
        )
        sample_logs.append(log)

    return sample_trades, sample_logs

def seed_database():
    """Seed database with sample data"""
    # First drop and recreate tables with correct schema
    if not drop_and_recreate_tables():
        return False
    
    # Create database session
    db = SessionLocal()
    
    try:
        logger.info("🌱 Creating sample data...")
        
        # Create and add sample data
        trades, logs = create_sample_data()
        
        # Add to database
        db.add_all(trades)
        db.add_all(logs)
        db.commit()
        
        # Verify data was added
        trade_count = db.query(Trade).count()
        log_count = db.query(TradeLog).count()
        
        logger.info(f"✅ Database seeded successfully!")
        logger.info(f"   - {trade_count} trades added")
        logger.info(f"   - {log_count} trade logs added")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error seeding database: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_data():
    """Verify that data exists in database"""
    db = SessionLocal()
    try:
        trade_count = db.query(Trade).count()
        log_count = db.query(TradeLog).count()
        
        logger.info(f"📊 Database verification:")
        logger.info(f"   - Trades: {trade_count}")
        logger.info(f"   - Trade Logs: {log_count}")
        
        if trade_count > 0:
            # Show sample trade
            sample_trade = db.query(Trade).first()
            logger.info(f"   - Sample trade: {sample_trade.symbol} {sample_trade.side} {sample_trade.qty} @ {sample_trade.price}")
        
        return trade_count > 0 and log_count > 0
        
    except Exception as e:
        logger.error(f"❌ Error verifying data: {e}")
        return False
    finally:
        db.close()

# Test AI signal generation
if __name__ == "__main__":
    logger.info("🚀 Testing AI signal generation...")
    
    ai_engine.activate()  # Activate AI engine
    
    for symbol in ai_engine.symbols:
        signal = ai_engine.generate_signal(symbol)
        if signal:
            logger.info(f"✅ Generated signal: {signal.signal_type} for {signal.symbol} with confidence {signal.confidence}")
        else:
            logger.warning(f"⚠️ No signal generated for {symbol}")
    
    ai_engine.deactivate()  # Deactivate AI engine
