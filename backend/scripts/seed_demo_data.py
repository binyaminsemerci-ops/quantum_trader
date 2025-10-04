#!/usr/bin/env python3
"""
Seed script for demo data in Quantum Trader database.

This script populates the database with sample trade data for demo purposes.
Run this after setting up the database with `alembic upgrade head`.

Usage:
    python scripts/seed_demo_data.py
"""

import sys
import os
from datetime import datetime, timezone, timedelta

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database import SessionLocal, TradeLog
from sqlalchemy.exc import IntegrityError


def create_demo_trades():
    """Create sample trade data for demo purposes."""
    db = SessionLocal()
    
    try:
        # Sample trades with different outcomes
        demo_trades = [
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "qty": 0.01,
                "price": 43500.00,
                "status": "FILLED",
                "reason": "Strong bullish signal detected",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=2)
            },
            {
                "symbol": "BTCUSDT", 
                "side": "SELL",
                "qty": 0.01,
                "price": 44200.00,
                "status": "FILLED",
                "reason": "Take profit target reached",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=1, minutes=30)
            },
            {
                "symbol": "ETHUSDT",
                "side": "BUY", 
                "qty": 0.5,
                "price": 2650.00,
                "status": "FILLED",
                "reason": "Technical breakout confirmed",
                "timestamp": datetime.now(timezone.utc) - timedelta(hours=1)
            },
            {
                "symbol": "ADAUSDT",
                "side": "BUY",
                "qty": 100.0,
                "price": 0.385,
                "status": "CANCELLED",
                "reason": "Market conditions changed",
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=45)
            },
            {
                "symbol": "SOLUSDT",
                "side": "BUY",
                "qty": 2.0,
                "price": 145.50,
                "status": "PARTIALLY_FILLED",
                "reason": "DCA strategy entry",
                "timestamp": datetime.now(timezone.utc) - timedelta(minutes=20)
            }
        ]
        
        # Create trade records
        for trade_data in demo_trades:
            trade = TradeLog(**trade_data)
            db.add(trade)
        
        db.commit()
        print(f"✅ Created {len(demo_trades)} demo trades successfully!")
        
        # Show what we created
        trades = db.query(TradeLog).all()
        print(f"\n📊 Total trades in database: {len(trades)}")
        for trade in trades[-5:]:  # Show last 5
            print(f"  {trade.timestamp.strftime('%H:%M')} | {trade.symbol} | {trade.side} | {trade.status}")
            
    except IntegrityError as e:
        db.rollback()
        print(f"❌ Error creating demo data: {e}")
    finally:
        db.close()


def create_demo_settings():
    """Create sample settings for demo purposes."""
    from database import Settings
    
    db = SessionLocal()
    
    try:
        # Check if settings already exist
        existing_settings = db.query(Settings).first()
        if existing_settings:
            print("⚠️ Settings already exist, skipping demo settings creation")
            return
            
        # Create demo settings (masked for security)
        demo_settings = Settings(
            api_key="demo_api_key_12345",
            api_secret="demo_secret_67890"
        )
        
        db.add(demo_settings)
        db.commit()
        print("✅ Created demo settings successfully!")
        
    except IntegrityError as e:
        db.rollback()
        print(f"❌ Error creating demo settings: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    print("🚀 Seeding Quantum Trader with demo data...")
    
    # Check if database exists and is accessible
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        print("💡 Make sure to run 'alembic upgrade head' first to create the schema")
        sys.exit(1)
    
    create_demo_trades()
    create_demo_settings()
    print("\n🎉 Demo data seeding complete!")