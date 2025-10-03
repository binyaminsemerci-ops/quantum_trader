#!/usr/bin/env python3
"""
Start the Enhanced Continuous Learning Engine with Live Data Feeds
Dette starter AI-systemet som lærer kontinuerlig fra Twitter, nyheter og markedsdata
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.continuous_learning_engine import ContinuousLearningEngine
from backend.utils.logging import get_logger

logger = get_logger(__name__)

def start_enhanced_learning():
    """Start the continuous learning engine with enhanced data feeds."""
    
    # Default symbols for learning (most liquid crypto pairs)
    symbols = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", 
        "ADAUSDT", "AVAXUSDT", "MATICUSDT", "DOTUSDT", "LINKUSDT"
    ]
    
    logger.info("🚀 Starting Enhanced Continuous Learning Engine...")
    logger.info(f"📈 Monitoring symbols: {', '.join(symbols)}")
    
    # Create and start the engine
    engine = ContinuousLearningEngine(
        symbols=symbols,
        twitter_update_interval=60,     # Twitter every 1 minute
        market_update_interval=30,      # Market data every 30 seconds  
        training_interval=3600,         # Retrain every hour
        sentiment_threshold=0.3,        # Sentiment impact threshold
        enhanced_fetch_interval=300     # Enhanced data every 5 minutes
    )
    
    logger.info("🤖 Starting AI learning loops...")
    
    try:
        # Start the engine
        engine.start()
        logger.info("✅ Enhanced Learning Engine Started Successfully!")
        logger.info("📊 Real-time AI strategy evolution from live data feeds active")
        logger.info("🐦 Twitter sentiment analysis: ACTIVE")
        logger.info("📈 Market data feeds: ACTIVE")  
        logger.info("🧠 Continuous model training: ACTIVE")
        logger.info("📡 Enhanced data sources: ACTIVE")
        
        # Keep running
        while engine.is_running:
            # Show learning status
            status = engine.get_learning_status()
            logger.info(f"📚 Learning Status: {status['symbols_monitored']} symbols, "
                       f"{status['data_points']} data points, "
                       f"Accuracy: {status['model_accuracy']:.1%}")
            
            import time
            time.sleep(60)  # Status update every minute
            
    except KeyboardInterrupt:
        logger.info("🛑 Stopping Enhanced Learning Engine...")
        engine.stop()
        logger.info("✅ Enhanced Learning Engine stopped")
        
    except Exception as e:
        logger.error(f"❌ Error in Enhanced Learning Engine: {e}")
        if hasattr(engine, 'stop'):
            engine.stop()

if __name__ == "__main__":
    start_enhanced_learning()