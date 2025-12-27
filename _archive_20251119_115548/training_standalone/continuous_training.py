#!/usr/bin/env python3
"""
🔄 KONTINUERLIG AI TRENING FOR FUTURES
=======================================
Kjører i bakgrunnen og trener AI kontinuerlig på fresh data.

Henter data fra:
- Binance Futures (top 100 by volume)
- CoinGecko (trending coins, market cap data)
- Layer 1 & Layer 2 coins
- 24h volume leaders

Trener hvert 4. time for å lære nye mønstre.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def continuous_training_loop():
    """Kontinuerlig treningsloop"""
    
    iteration = 0
    
    while True:
        iteration += 1
        logger.info("=" * 80)
        logger.info(f"🔄 KONTINUERLIG TRENING - Iterasjon #{iteration}")
        logger.info(f"⏰ Startet: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        
        try:
            # Import backend AI retraining with correct paths
            backend_path = str(Path(__file__).parent / "backend")
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            from backend.database import SessionLocal
            from backend.services.ai_trading_engine import create_ai_trading_engine
            from ai_engine.agents.xgb_agent import make_default_agent
            
            db = SessionLocal()
            agent = make_default_agent()
            ai_engine = create_ai_trading_engine(agent=agent, db_session=db)
            
            logger.info("🤖 Starter AI retraining...")
            result = await ai_engine._retrain_model(min_samples=1)  # Accept any samples for rapid learning
            
            if result.get("status") == "success":
                logger.info(f"[OK] Trening vellykket!")
                logger.info(f"   Samples: {result.get('training_samples', 0)}")
                logger.info(f"   Train accuracy: {result.get('train_accuracy', 0):.3f}")
                logger.info(f"   Val accuracy: {result.get('validation_accuracy', 0):.3f}")
                logger.info(f"   Modell: {result.get('model_path', 'N/A')}")
            else:
                logger.warning(f"[WARNING] Trening feilet: {result.get('reason', result.get('error', 'Unknown'))}")
            
            db.close()
            
        except Exception as e:
            logger.error(f"❌ Feil under trening: {e}", exc_info=True)
        
        # Vent 5 minutter før neste trening
        wait_minutes = 5
        wait_seconds = wait_minutes * 60
        
        logger.info(f"")
        logger.info(f"💤 Venter {wait_minutes} minutter før neste trening...")
        logger.info(f"   Neste kjøring: {datetime.fromtimestamp(time.time() + wait_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80)
        logger.info("")
        
        await asyncio.sleep(wait_seconds)


async def fetch_and_train():
    """Hent data og tren modell"""
    
    logger.info("[CHART] Henter fresh data fra Binance Futures...")
    
    try:
        import ccxt
        
        exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_API_SECRET'),
            'options': {'defaultType': 'future'}
        })
        
        # Hent alle USDT futures
        markets = await exchange.load_markets()
        usdt_futures = [
            symbol for symbol in markets.keys()
            if symbol.endswith('/USDT:USDT')
        ]
        
        logger.info(f"[CHART_UP] Fant {len(usdt_futures)} USDT perpetual futures")
        
        # Sorter etter volume
        tickers = await exchange.fetch_tickers(usdt_futures)
        sorted_symbols = sorted(
            tickers.items(),
            key=lambda x: x[1].get('quoteVolume', 0) if x[1] else 0,
            reverse=True
        )
        
        top_100 = [sym for sym, _ in sorted_symbols[:100]]
        
        logger.info(f"[TARGET] Top 100 by volume:")
        for i, sym in enumerate(top_100[:10], 1):
            vol = tickers[sym]['quoteVolume'] / 1_000_000 if tickers[sym] else 0
            logger.info(f"   #{i}. {sym}: ${vol:.1f}M")
        
        # Her kan du legge til faktisk data-henting og trening
        # For nå, returner bare listen
        return top_100
        
    except Exception as e:
        logger.error(f"❌ Feil ved datahenting: {e}")
        return []


if __name__ == "__main__":
    logger.info("[ROCKET] STARTER KONTINUERLIG AI TRENING")
    logger.info("📅 Treningsintervall: Hver 4. time")
    logger.info("[CHART] Data: Top 100 Binance Futures by 24h volume")
    logger.info("[TARGET] Mål: Kontinuerlig læring av futures strategier")
    logger.info("")
    
    try:
        asyncio.run(continuous_training_loop())
    except KeyboardInterrupt:
        logger.info("")
        logger.info("⏹️ Kontinuerlig trening stoppet av bruker")
        logger.info("👋 Ha det!")
